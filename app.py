# app.py — Gig Explorer (France): robust CSV, FR seasons, genres, dynamic map
# Notebook-style: every chart includes an automatic interpretation in English.

import io, os, unicodedata, time, re, requests, math
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# ===== Meta: data source & about block =====
SOURCE_URL = "https://www.data.gouv.fr/datasets/liste-des-festivals-en-france//"  
DATASET_NAME = "festivals-global-festivals-pl"   
ACCESS_DATE = "2025-10-25"

def show_data_meta():
    st.sidebar.markdown(
        f"**Data source:** [{DATASET_NAME}]({SOURCE_URL})  \n"
        f"**Accessed:** {ACCESS_DATE}"
    )

def about_block():
    with st.expander("About this data / Limitations"):
        st.markdown(
            "- Source: "
            f"[{DATASET_NAME}]({SOURCE_URL})"
            f" (accessed: {ACCESS_DATE})\n"
            "- Cleaning: encoding normalization, French month parsing, split multi-genre values ('/', '&', ';'), and basic dedup.\n"
            "- Limits: coverage varies by region and month; potential duplicates from aggregator feeds; geocoding via OSM used only when coordinates are missing."
        )

# ------------------------------- Page ----------------------------------------
st.set_page_config(page_title="Gig Explorer - France Concerts & Festivals", layout="wide")
st.title("Gig Explorer - France Concerts & Festivals")
st.caption("From the big picture to 'where to go next': interactive analysis of concerts & festivals in France.")

# ----------------------------- Sidebar ---------------------------------------
with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_main")
    default_csv = "festivals-global-festivals-pl.csv"
    use_local = st.checkbox(f"Use local file: {default_csv}", value=True)

    st.divider()
    sep_choice = st.selectbox("Delimiter", ["Auto", ",", ";", "\\t", "|"], index=0)
    enc_choice = st.selectbox("Encoding", ["Auto", "utf-8", "latin-1"], index=0)

    st.divider()
    st.header("Story mode")
    story_mode = st.radio(
        "Choose a narrative path:",
        ["Overview", "Seasons", "Hotspots", "Top Genres", "Map"],
        index=0
    )

    st.divider()
    st.header("Map options")
    allow_geocode = st.checkbox(
        "Enable on-the-fly geocoding if no Latitude/Longitude (needs internet; rate-limited)",
        value=False
    )

# ----------------------- Robust CSV loader -----------------------------------
@st.cache_data(show_spinner=True)
def read_csv_robust(source, sep_opt, enc_opt):
    encs = ["utf-8", "latin-1"] if enc_opt == "Auto" else [enc_opt]

    def _read(sep, enc, force_python=False):
        kw = {"encoding": enc, "on_bad_lines": "skip"}
        if sep is None:
            kw.update({"sep": None, "engine": "python"})
        else:
            kw.update({"sep": sep})
            if force_python:
                kw.update({"engine": "python"})
        return pd.read_csv(source, **kw)

    if sep_opt == "Auto":
        for e in encs:
            try:
                return _read(None, e)
            except Exception:
                pass
        for e in encs:
            for s in [",", ";", "\t", "|"]:
                for force in (False, True):
                    try:
                        return _read(s, e, force_python=force)
                    except Exception:
                        pass
    else:
        s = {",": ",", ";": ";", "\\t": "\t", "|": "|"}.get(sep_opt, sep_opt)
        for e in encs:
            for force in (False, True):
                try:
                    return _read(s, e, force_python=force)
                except Exception:
                    pass

    raise ValueError("Couldn't parse CSV. Try Delimiter=';' and/or Encoding='latin-1'.")

def load_df(uploaded, use_local, sep_opt, enc_opt, default_csv):
    if uploaded is not None:
        return read_csv_robust(io.BytesIO(uploaded.getvalue()), sep_opt, enc_opt)
    if use_local:
        if os.path.exists(default_csv):
            return read_csv_robust(default_csv, sep_opt, enc_opt)
        st.error(f"Local file '{default_csv}' not found next to app.py.")
        st.stop()
    st.info("Upload a CSV or tick 'Use local file' to begin.")
    st.stop()

df = load_df(uploaded, use_local, sep_choice, enc_choice, default_csv)

# Auto-fix typical FR issues
try:
    needs_fix = (df.shape[1] == 1) or any(";" in str(c) for c in df.columns)
    if needs_fix:
        src = io.BytesIO(uploaded.getvalue()) if uploaded is not None else default_csv
        df = read_csv_robust(src, ";", "latin-1")
        # (No banner shown)
except Exception:
    pass

# Clean BOM + occasional mojibake
df.columns = df.columns.str.replace(r'^\ufeff', '', regex=True)

def looks_mojibake(s: pd.Series) -> bool:
    sample = ' '.join(map(str, s.head(50).tolist()))
    return any(x in sample for x in ["Ã", "Â", "¤", "", ""])

def fix_series_mojibake(s: pd.Series) -> pd.Series:
    try:
        return s.astype(str).apply(lambda x: x.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore'))
    except Exception:
        return s

if looks_mojibake(pd.Series(df.columns)):
    df.columns = [c.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore') for c in df.columns]

for c in df.select_dtypes(include=['object']).columns:
    if looks_mojibake(df[c]):
        df[c] = fix_series_mojibake(df[c]).str.strip()


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().replace("-", " ").replace("_", " ")

cols = list(df.columns)
lower = [_norm(c) for c in cols]

def guess(keys):
    for k in keys:
        nk = _norm(k)
        for i, nm in enumerate(lower):
            if nk in nm:
                return cols[i]
    return None

date_guess   = guess(["date","jour","day","datetime","debut","début","date debut","date début"])
period_guess = guess(["periode","période","saison","dates","période de déroulement","periode de deroulement"])
city_guess   = guess(["city","ville","commune"])
region_guess = guess(["region","région","depart","departement","dept","département","province"])
genre_guess  = guess(["genre","style","discipline","esthetique","esthétique","type de musique","musique"])
price_guess  = guess(["price","tarif","ticket","prix","cout","coût"])
lat_guess    = guess(["lat","latitude"])
lon_guess    = guess(["lon","lng","long","longitude"])

st.subheader("Map your columns")
c1, c2, c3, c4 = st.columns(4)
with c1:
    date_col   = st.selectbox("Date (if exact)", ["- none -"] + cols, index=(cols.index(date_guess)+1 if date_guess in cols else 0))
    period_col = st.selectbox("Period / Season text (French months)", ["- none -"] + cols, index=(cols.index(period_guess)+1 if period_guess in cols else 0))
with c2:
    city_col   = st.selectbox("City", ["- none -"] + cols, index=(cols.index(city_guess)+1 if city_guess in cols else 0))
    region_col = st.selectbox("Region/Dept", ["- none -"] + cols, index=(cols.index(region_guess)+1 if region_guess in cols else 0))
with c3:
    genre_col  = st.selectbox("Style / Genre", ["- none -"] + cols, index=(cols.index(genre_guess)+1 if genre_guess in cols else 0))
    # Removed price selectbox
with c4:
    lat_col = st.selectbox("Latitude (optional)", ["- none -"] + cols, index=(cols.index(lat_guess)+1 if lat_guess in cols else 0))
    lon_col = st.selectbox("Longitude (optional)", ["- none -"] + cols, index=(cols.index(lon_guess)+1 if lon_guess in cols else 0))

# Always set price_col to none (removed price mapping)
price_col = "- none -"

# Casts (safe)
if date_col != "- none -":
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if lat_col != "- none -":
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
if lon_col != "- none -":
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

# ------------------------------- Filters -------------------------------------
# ------------------------------- Filters -------------------------------------
show_filters = (
    story_mode != "Overview" and (
        (date_col != "- none -" and df[date_col].notna().any()) or
        (region_col != "- none -") or
        (city_col != "- none -")
    )
)
df_f = df.copy()
if show_filters:
    st.markdown("---")
    st.subheader("Filters")
    fc1, fc2, fc3 = st.columns(3)

    if date_col != "- none -" and df[date_col].notna().any():
        dmin, dmax = pd.to_datetime(df[date_col].min()), pd.to_datetime(df[date_col].max())
        with fc1:
            dr = st.date_input("Date range", (dmin.date(), dmax.date()))
            dmin, dmax = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df_f = df_f[(df_f[date_col] >= dmin) & (df_f[date_col] <= dmax)]

    with fc2:
        if region_col != "- none -":
            regions = sorted(df[region_col].dropna().astype(str).unique())
            region_sel = st.multiselect("Region/Dept", regions)
        else:
            region_sel = None
    if region_sel:
        df_f = df_f[df_f[region_col].astype(str).isin(region_sel)]

    with fc3:
        if city_col != "- none -":
            cities = sorted(df[city_col].dropna().astype(str).unique())
            city_sel = st.multiselect("City", cities)
        else:
            city_sel = None
    if city_sel:
        df_f = df_f[df_f[city_col].astype(str).isin(city_sel)]

# ------------------------------ KPIs -----------------------------------------
if story_mode != "Overview":
    st.markdown("---")
    st.subheader("Key metrics (filtered)")
    k1, k2, k3, k4 = st.columns(4)

    if date_col != "- none -":
        k1.metric("Upcoming events", int(df_f[df_f[date_col] >= pd.Timestamp(datetime.now().date())].shape[0]))
    else:
        k1.metric("Upcoming events", "-")

    if region_col != "- none -":
        k2.metric("Regions covered", int(df_f[region_col].dropna().nunique()))
    elif city_col != "- none -":
        k2.metric("Cities covered", int(df_f[city_col].dropna().nunique()))
    else:
        k2.metric("Areas covered", "-")

    if genre_col != "- none -" and df_f[genre_col].notna().any():
        try:
            gsplit = (
                df_f[genre_col].astype(str)
                .str.replace(r"\s*&\s*|/|\|", ",", regex=True)
                .str.replace(";", ",")
                .str.split(",")
            )
            distinct_genres = pd.Series([x.strip() for lst in gsplit if isinstance(lst, list) for x in lst if x and x.strip()]).str.title().nunique()
            k3.metric("Distinct genres", int(distinct_genres))
        except Exception:
            k3.metric("Distinct genres", "-")
    else:
        k3.metric("Distinct genres", "-")

    k4.metric("Events (filtered)", int(df_f.shape[0]))

analysis_mode = st.checkbox("Show detailed interpretations", value=True)
def explain(md: str):
    if analysis_mode:
        st.markdown(md)

# ----------------- Utilities for seasons (FR text) ---------------------------
FR_MONTHS = {
    "janvier":1, "fevrier":2, "février":2, "mars":3, "avril":4, "mai":5, "juin":6,
    "juillet":7, "aout":8, "août":8, "septembre":9, "octobre":10, "novembre":11, "decembre":12, "décembre":12
}
FR_MONTH_RE = re.compile("|".join(sorted(FR_MONTHS.keys(), key=len, reverse=True)), flags=re.IGNORECASE)

def extract_months_from_text(s: pd.Series) -> pd.Series:
    counts = {m:0 for m in range(1,13)}
    for val in s.dropna().astype(str):
        found = FR_MONTH_RE.findall(val)
        for mname in found:
            counts[FR_MONTHS[mname.lower()]] += 1
    return pd.Series(counts)

def month_order_names():
    return ["January","February","March","April","May","June","July","August","September","October","November","December"]

def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b

# -------------------- Geocoding (optional, cached) ---------------------------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def geocode_cities_osm(cities: list) -> pd.DataFrame:
    out, sess = [], requests.Session()
    headers = {"User-Agent": "GigExplorer/1.0 (educational project)"}
    total = len(cities)
    pb = st.progress(0, text="Geocoding cities…")
    for i, city in enumerate(cities, start=1):
        try:
            r = sess.get("https://nominatim.openstreetmap.org/search",
                         params={"q": f"{city}, France", "format":"json", "limit":1},
                         headers=headers, timeout=12)
            if r.ok and r.json():
                j = r.json()[0]
                out.append({"city": city, "lat": float(j["lat"]), "lon": float(j["lon"])})
            else:
                out.append({"city": city, "lat": None, "lon": None})
        except Exception:
            out.append({"city": city, "lat": None, "lon": None})
        time.sleep(1.05)  # respect OSM rate limits
        pb.progress(int(i/total*100), text=f"Geocoding {i}/{total}")
    pb.empty()
    return pd.DataFrame(out)

# ------------------------------ Story sections -------------------------------
st.markdown("---")

if story_mode == "Overview":
    st.subheader("Welcome - Start your journey")
    # Hero band (simple HTML/CSS for a pleasing banner)
    st.markdown(
        '''
        <div style="padding: 24px 28px; border-radius: 14px; background: linear-gradient(135deg, #1f2937, #111827); color: #fff;">
          <h2 style="margin: 0 0 8px 0;">You love music, concerts, and nights out?</h2>
          <p style="margin: 0; font-size: 16px; line-height: 1.6;">
            <strong>Welcome to Gig Explorer</strong> - your friendly gateway to France’s concerts & festivals.
            Discover new places to dance, sing, and celebrate with your crew. Pick a tab, follow the vibe, and plan your next night.
          </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.markdown("")
    # Optional aesthetic image strip (remote images; will silently fail if offline)
    colA, colB, colC = st.columns(3)
    with colA:
        st.image("https://images.unsplash.com/photo-1514525253161-7a46d19cd819?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Live crowd energy")
    with colB:
        st.image("https://images.unsplash.com/photo-1506157786151-b8491531f063?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Festival nights")
    with colC:
        st.image("https://images.unsplash.com/photo-1483412033650-1015ddeb83d1?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="City stages")
    st.markdown(
        "_Tip: Use the sidebar to choose **Seasons**, **Hotspots**, **Top Genres**, or **Map** when you're ready._"
    )

elif story_mode == "Seasons":
    st.subheader("Seasons - When the scene explodes")
    # Hero band for Seasons
    st.markdown(
        '''
        <div style="padding: 20px 24px; border-radius: 14px; background: linear-gradient(135deg, #0b3b5c, #0a2a43); color: #fff; margin-bottom: 8px;">
          <h3 style="margin: 0 0 6px 0;">When's the best time to go?</h3>
          <p style="margin: 0; font-size: 15px; line-height: 1.6;">
            Seasons shape the scene - summer festival waves, quieter winter months. Browse the monthly timeline to spot peak periods and plan your next gig run.
          </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    cA, cB, cC = st.columns(3)
    with cA:
        st.image("https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Summer festival glow")
    with cB:
        st.image("https://images.unsplash.com/photo-1483412033650-1015ddeb83d1?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Autumn & indoor vibes")
    with cC:
        st.image("https://images.unsplash.com/photo-1518972559570-7cc1309f3229?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Seasonal moods")
    st.markdown(
        "**Welcome to the seasons view.** This page looks at the monthly rhythm: when activity swells, when it slows down, and which months quietly surprise.\n\n"
        "**Let’s explore.**"
    )

    shown = False
    # Exact dates
    if date_col != "- none -" and df_f[date_col].notna().any():
        t = df_f.copy()
        t["month_name"] = t[date_col].dt.month_name()
        by_month = t["month_name"].value_counts().reindex(month_order_names()).fillna(0).astype(int)
        if by_month.sum() > 0:
            fig = px.bar(by_month.reset_index().rename(columns={"index":"Month","month_name":"Events"}),
                         x="Month", y="Events")
            st.plotly_chart(fig, use_container_width=True)
            # Richer interpretation for months (exact dates)
            total = int(by_month.sum())
            ordered = by_month.sort_values(ascending=False)
            top1_m, top1_v = ordered.index[0], int(ordered.iloc[0])
            top2_m, top2_v = (ordered.index[1], int(ordered.iloc[1])) if ordered.size > 1 else (None, 0)
            low_m, low_v = ordered.index[-1], int(ordered.iloc[-1])
            share_top1 = pct(top1_v, total)
            share_top2 = pct(top2_v, total) if total else 0
            summer = int(by_month.loc[["June","July","August"]].sum())
            winter = int(by_month.loc[["December","January","February"]].sum())
            explain(
                f"**Interpretation.** We count **{total} events** with a clear seasonal pulse. "
                f"**{top1_m}** peaks at **{top1_v}** ({share_top1:.1f}%), followed by **{top2_m}** "
                f"with **{top2_v}** ({share_top2:.1f}%). Summer brings **{summer}** events, "
                f"while winter (Dec–Feb) totals **{winter}**. There is still some activity in **December** and a small rebound in **January**, "
                f"but the main wave sits in early summer. If you want the widest choice, target **{top1_m}**; "
                f"for quieter trips, look around **{low_m}**."
            )
            shown = True

    # French period text
    if not shown and period_col != "- none -" and df_f[period_col].notna().any():
        counts = extract_months_from_text(df_f[period_col])
        month_labels = month_order_names()
        month_df = pd.DataFrame({
            "Month": [month_labels[i-1] for i in range(1,13)],
            "Events": [int(counts[i]) for i in range(1,13)]
        })
        fig = px.bar(month_df, x="Month", y="Events")
        st.plotly_chart(fig, use_container_width=True)
        # Richer interpretation for months (text proxy)
        total = int(month_df["Events"].sum())
        ordered = month_df.sort_values("Events", ascending=False)
        top1 = ordered.iloc[0]; top2 = ordered.iloc[1] if len(ordered) > 1 else None
        low  = ordered.iloc[-1]
        summer = int(month_df.loc[month_df["Month"].isin(["June","July","August"]), "Events"].sum())
        winter = int(month_df.loc[month_df["Month"].isin(["December","January","February"]), "Events"].sum())
        msg = (
            f"**Interpretation.** The chart shows a clear seasonal pattern. "
            f"**{top1['Month']}** stands out with **{int(top1['Events'])}** events, "
        )
        if top2 is not None:
            msg += f"followed by **{top2['Month']}** ({int(top2['Events'])}). "
        msg += (
            f"Across the year, summer concentrates **{summer}** events, while winter (Dec–Feb) is softer with **{winter}**. "
            f"There is still visible activity in December and a small rebound in January, but the main wave is in early summer."
        )
        explain(msg)
        shown = True

    if not shown:
        st.info("Map either an exact 'Date' or a 'Period/Season text' column containing French month names.")

elif story_mode == "Hotspots":
    st.subheader("Hotspots - Where to go")
    # Hero band for Hotspots
    st.markdown(
        '''
        <div style="padding: 20px 24px; border-radius: 14px; background: linear-gradient(135deg, #3b0764, #1e1b4b); color: #fff; margin-bottom: 8px;">
          <h3 style="margin: 0 0 6px 0;">Where's the party?</h3>
          <p style="margin: 0; font-size: 15px; line-height: 1.6;">
            From Brittany coasts to Rhône-Alpes arenas, find the regions and cities that pack the calendar. Scroll the charts to uncover hubs you might be missing.
          </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        '''
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 18px;">
          <figure style="margin:0;">
            <img src="https://images.unsplash.com/photo-1492684223066-81342ee5ff30?q=80&w=1200&auto=format&fit=crop" style="width:100%; height:240px; object-fit:cover; border-radius:12px;" alt="Crowd & lights - Brittany" />
            <figcaption style="text-align:center; font-size:12px; margin-top:6px;">Rennes - Brittany</figcaption>
          </figure>
          <figure style="margin:0;">
            <img src="https://images.unsplash.com/photo-1506157786151-b8491531f063?q=80&w=1200&auto=format&fit=crop" style="width:100%; height:240px; object-fit:cover; border-radius:12px;" alt="Nice festival lights - Provence-Alpes-Côte d’Azur" />
            <figcaption style="text-align:center; font-size:12px; margin-top:6px;">Nice - Provence-Alpes-Côte d’Azur</figcaption>
          </figure>
          <figure style="margin:0;">
            <img src="https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?q=80&w=1200&auto=format&fit=crop" style="width:100%; height:240px; object-fit:cover; border-radius:12px;" alt="Open‑air crowd - Occitanie" />
            <figcaption style="text-align:center; font-size:12px; margin-top:6px;">Montpellier - Occitanie</figcaption>
          </figure>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        "**Welcome to the hotspots view.** This page shows the regions and cities with the most events. Use the charts to find big hubs or smaller local scenes.\n\n"
        "**Let’s explore.**"
    )
    if region_col != "- none -" and df_f[region_col].notna().any():
        top_r = df_f[region_col].astype(str).value_counts().reset_index()
        top_r.columns = ["Region/Dept","Events"]
        st.plotly_chart(px.bar(top_r, x="Events", y="Region/Dept", orientation="h"), use_container_width=True)
        # Richer interpretation for regions
        ordered = top_r.sort_values("Events", ascending=False)
        top1_r = ordered.iloc[0]; top2_r = ordered.iloc[1] if len(ordered) > 1 else None; top3_r = ordered.iloc[2] if len(ordered) > 2 else None
        total_r = int(ordered["Events"].sum())
        msg_r = (
            f"**Interpretation.** A few regions clearly anchor the scene. "
            f"Leader: **{top1_r['Region/Dept']}** with **{int(top1_r['Events'])}** events. "
        )
        if top2_r is not None:
            msg_r += f"Next: **{top2_r['Region/Dept']}** (**{int(top2_r['Events'])}**). "
        if top3_r is not None:
            msg_r += f"Third: **{top3_r['Region/Dept']}** (**{int(top3_r['Events'])}**). "
        if top2_r is not None and top3_r is not None:
            trio = int(top1_r['Events'] + top2_r['Events'] + top3_r['Events'])
            msg_r += f"Together, the top‑3 gather **{pct(trio, total_r):.1f}%** of all events. "
        msg_r += (
            "If you want the most options and bigger line‑ups, start with these hubs; "
            "for slower, local nights, explore mid‑table regions."
        )
        explain(msg_r)

        if city_col != "- none -" and df_f[city_col].notna().any():
            st.write("Top cities (within current filters)")
            top_c = df_f[city_col].astype(str).value_counts().head(40).reset_index()
            top_c.columns = ["City","Events"]
            st.plotly_chart(px.bar(top_c, x="Events", y="City", orientation="h"), use_container_width=True)
            # Richer interpretation for cities
            total_c = int(top_c["Events"].sum())
            top1_c = top_c.iloc[0]
            gap = (int(top1_c["Events"]) / int(top_c.iloc[1]["Events"])) if top_c.shape[0] > 1 and int(top_c.iloc[1]["Events"]) > 0 else 1
            msg_c = (
                f"**City lens.** **{top1_c['City']}** leads with **{int(top1_c['Events'])}** events. "
            )
            if gap >= 1.5 and top_c.shape[0] > 1:
                msg_c += "There is a clear first‑place gap. "
            top10_share = pct(int(top_c.head(10)['Events'].sum()), total_c)
            msg_c += (
                f"The top‑10 cities gather **{top10_share:.1f}%** of the listed events. "
                "For party‑heavy trips, aim for the top of the list; for discovery and smaller venues, explore smaller cities further down the list."
            )
            explain(msg_c)
        # Treemap Region→City
        if city_col != "- none -" and df_f[city_col].notna().any():
            agg_rc = df_f.groupby([region_col, city_col]).size().reset_index(name="Events")
            fig_tree = px.treemap(agg_rc, path=[region_col, city_col], values="Events")
            st.plotly_chart(fig_tree, use_container_width=True)
            explain(
                "**Treemap view.** Each rectangle area equals the event count. Regions with many medium tiles indicate distributed activity across multiple cities; one very large tile reveals a single anchor city. Use this to choose between variety (many mid‑sized tiles) and convenience (one dominant hub)."
            )
            # Focused reading for Île-de-France only
            try:
                idf_mask = agg_rc[region_col].astype(str) == "Île-de-France"
                idf = agg_rc[idf_mask].sort_values("Events", ascending=False)
                if not idf.empty:
                    total_idf = int(idf["Events"].sum())
                    top_row = idf.iloc[0]
                    top_city_name = str(top_row[city_col])
                    top_city_share = (int(top_row["Events"]) / total_idf) * 100 if total_idf else 0
                    other_cities = max(0, idf.shape[0] - 1)
                    explain(
                        f"**Reading the treemap - Île-de-France.** **{top_city_name}** holds about **{top_city_share:.1f}%** of the region’s events. "
                        f"The remaining **{other_cities}** cities share the rest, showing how activity is distributed across the whole region."
                    )
            except Exception:
                pass
    elif city_col != "- none -" and df_f[city_col].notna().any():
        top_c = df_f[city_col].astype(str).value_counts().head(40).reset_index()
        top_c.columns = ["City","Events"]
        st.plotly_chart(px.bar(top_c, x="Events", y="City", orientation="h"), use_container_width=True)
        explain("**Interpretation.** City activity is uneven: expect tighter clusters around metro areas and university towns.")
    else:
        st.info("Map 'Region/Dept' or 'City' to see hotspots.")

elif story_mode == "Top Genres":
    st.subheader("Top Genres - What's the vibe?")
    # Hero band for Genres
    st.markdown(
        '''
        <div style="padding: 20px 24px; border-radius: 14px; background: linear-gradient(135deg, #064e3b, #022c22); color: #fff; margin-bottom: 8px;">
          <h3 style="margin: 0 0 6px 0;">What's your vibe?</h3>
          <p style="margin: 0; font-size: 15px; line-height: 1.6;">
            From live music to cinema nights, book fairs, digital and visual arts - discover what’s trending and how the cultural mix shifts across time and places. Tune the filters to your taste.
          </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    gA, gB, gC = st.columns(3)
    with gA:
        st.image("https://images.unsplash.com/photo-1511379938547-c1f69419868d?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Pop & rock energy")
    with gB:
        st.image("https://images.unsplash.com/photo-1483412033650-1015ddeb83d1?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Electronic nights")
    with gC:
        st.image("https://images.unsplash.com/photo-1519681393784-d120267933ba?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Jazz & acoustic")
    st.markdown(
        "**Welcome to the Top Genres view.** This page shows which categories dominate and how they shift across time and regions."
    )
    st.markdown("**Let’s explore.**")
    if genre_col != "- none -" and df_f[genre_col].notna().any():
        g = (
            df_f[genre_col].astype(str)
            .str.replace(r"\s*&\s*", ",", regex=True)
            .str.replace(r"\s*/\s*", ",", regex=True)
            .str.replace(r"\s*\|\s*", ",", regex=True)
            .str.replace(";", ",")
            .str.split(",")
        )
        flat = pd.Series([x.strip() for lst in g if isinstance(lst, list) for x in lst if x and x.strip()])
        flat = flat.str.strip().str.replace(r"\s+", " ", regex=True).str.title()
        topg = flat.value_counts().reset_index()
        topg.columns = ["Genre/Style", "Events"]
        st.plotly_chart(px.bar(topg.head(40), x="Events", y="Genre/Style", orientation="h"), use_container_width=True)

        # Focused interpretation for genres (Music vs Spectacle Vivant)
        total_g = int(topg["Events"].sum())
        top1 = topg.iloc[0]
        top2 = topg.iloc[1] if len(topg) > 1 else None
        msgg = (
            f"**Interpretation.** We observe **{int(topg.shape[0])} genres/styles**. "
            f"**{top1['Genre/Style']}** is clearly in first place with **{int(top1['Events'])}** events."
        )
        if top2 is not None:
            msgg += f" **{top2['Genre/Style']}** comes next with **{int(top2['Events'])}**, and there is a visible gap after these two."
        msgg += " This shows the offer is concentrated in the leading categories, while the others are more specialized."
        explain(msgg)

        # Stacked genre by month (if possible)
        if date_col != "- none -":
            tmp = df_f[[date_col, genre_col]].dropna().copy()
            tmp["month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
            tmp["__split"] = tmp[genre_col].astype(str).str.replace(r"\s*&\s*|/|\|", ",", regex=True).str.replace(";", ",").str.split(",")
            ex = tmp.explode("__split")
            ex["Genre"] = ex["__split"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
            by_mg = ex.groupby(["month","Genre"]).size().reset_index(name="Events")
            fig_mg = px.bar(by_mg, x="month", y="Events", color="Genre", barmode="stack")
            fig_mg.update_layout(xaxis_title="", yaxis_title="Events")
            st.plotly_chart(fig_mg, use_container_width=True)
            explain(
                "**Temporal mix.** Stacked bars reveal how presence shifts month to month. "
                "Look for seasonal surges (for example, open‑air festivals in summer) and quieter off‑season periods. "
                "A stable color pattern means balanced programming; sharp spikes reveal seasonal categories."
            )
        # Heatmap genre by region (if possible)
        if region_col != "- none -":
            ex2 = df_f[[region_col, genre_col]].dropna().copy()
            ex2["__split"] = ex2[genre_col].astype(str).str.replace(r"\s*&\s*|/|\|", ",", regex=True).str.replace(";", ",").str.split(",")
            ex2 = ex2.explode("__split")
            ex2["Genre"] = ex2["__split"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
            topG = ex2["Genre"].value_counts().head(12).index.tolist()
            mat = ex2[ex2["Genre"].isin(topG)].pivot_table(index="Genre", columns=region_col, values=genre_col, aggfunc="count").fillna(0)
            fig_hg = px.imshow(mat, aspect="auto", labels=dict(x="Region/Dept", y="Genre", color="Events"))
            st.plotly_chart(fig_hg, use_container_width=True)
            explain(
                "**Regional flavor.** The heatmap shows which regions over‑index on specific genres. "
                "Bright cells in only a few columns point to localized scenes (regional specialities). "
                "Rows that are bright across many columns indicate nationwide appeal. Use this view to pick regions that match your taste."
            )
            # Concrete example for Nouvelle-Aquitaine
            try:
                region_example = "Nouvelle-Aquitaine"
                if region_example in mat.columns:
                    s = mat[region_example].sort_values(ascending=False)
                    if s.shape[0] >= 2:
                        g1, v1 = s.index[0], int(s.iloc[0])
                        g2, v2 = s.index[1], int(s.iloc[1])
                        explain(
                            f"**Example - Nouvelle‑Aquitaine.** **{g1}** has about **{v1}** events, while **{g2}** has about **{v2}**. "
                            "This highlights how one genre can stand out inside a region."
                        )
            except Exception:
                pass
    else:
        st.info("Map a 'Style / Genre' column to see this view.")

elif story_mode == "Map":
    st.subheader("Map - Find it on the ground")
    # Hero band for Map
    st.markdown(
        '''
        <div style="padding: 20px 24px; border-radius: 14px; background: linear-gradient(135deg, #7f1d1d, #431407); color: #fff; margin-bottom: 8px;">
          <h3 style="margin: 0 0 6px 0;">Ready to go IRL?</h3>
          <p style="margin: 0; font-size: 15px; line-height: 1.6;">
            Pan, zoom, and pinpoint gigs near you. Use the filters (region, city, date, genre) to plan the perfect night out - from intimate clubs to open-air festivals.
          </p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    mA, mB, mC = st.columns(3)
    with mA:
        st.image("https://images.unsplash.com/photo-1469474968028-56623f02e42e?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Road to the venue")
    with mB:
        st.image("https://images.unsplash.com/photo-1518972559570-7cc1309f3229?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="On the ground")
    with mC:
        st.image("https://images.unsplash.com/photo-1514525253161-7a46d19cd819?q=80&w=800&auto=format&fit=crop", use_container_width=True, caption="Night crowd")
    st.markdown("**Let’s explore.**")

    map_lat_col = lat_col if lat_col != "- none -" else None
    map_lon_col = lon_col if lon_col != "- none -" else None
    df_map = df_f.copy()

    # Optional geocoding if no coordinates
    if (map_lat_col is None or map_lon_col is None) and allow_geocode and city_col != "- none -":
        st.write("No coordinates detected - geocoding most frequent cities with OpenStreetMap (temporary).")
        city_counts = df_map[city_col].dropna().astype(str).value_counts()
        LIMIT = 300
        top_cities = city_counts.head(LIMIT).index.tolist()
        geo = geocode_cities_osm(top_cities)
        df_map = df_map.merge(geo.rename(columns={"city": city_col}), on=city_col, how="left")
        map_lat_col, map_lon_col = "lat", "lon"
        st.success(f"Geocoded {geo['lat'].notna().sum()} / {len(top_cities)} cities (top by frequency).")

    if map_lat_col and map_lon_col and df_map[[map_lat_col, map_lon_col]].notna().any().any():
        to_plot = df_map.dropna(subset=[map_lat_col, map_lon_col]).copy()
        max_points = st.slider("Max points to render", min_value=500, max_value=10000, value=3000, step=500)
        if to_plot.shape[0] > max_points:
            to_plot = to_plot.head(max_points)
        hover_cols = []
        if city_col   != "- none -": hover_cols.append(city_col)
        if region_col != "- none -": hover_cols.append(region_col)
        if genre_col  != "- none -": hover_cols.append(genre_col)
        if date_col   != "- none -": hover_cols.append(date_col)

        fig_map = px.scatter_mapbox(
            to_plot,
            lat=map_lat_col, lon=map_lon_col,
            hover_name=(city_col if city_col != "- none -" else None),
            hover_data=hover_cols,
            opacity=0.7,
            zoom=4, height=620
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)

        total_rows = df_f.shape[0]
        plotted = to_plot.shape[0]
        pct_plotted = pct(plotted, total_rows)
        explain(
            f"**Interpretation.** The map shows where events cluster. "
            f"Currently displaying **{plotted}** points ({pct_plotted:.1f}% of filtered rows). "
            "Large clusters point to busy areas (often big cities). Use Region/City and Genre filters to focus on places that match your plan."
        )
    else:
        st.info("Provide 'Latitude' and 'Longitude' columns or enable geocoding and map a 'City' column.")