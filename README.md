# Gig Explorer - France Concerts & Festivals

A Streamlit data-storytelling app that answers three user questions:
- **When** to go (Seasons)
- **Where** to go (Hotspots: regions, cities, treemap)
- **What** to expect (Top Genres)
Plus a **Map** to pass from explore to action.

## Data
- **Source:** festivals-global-festivals-pl
(https://www.data.gouv.frdatasetsliste-des-festivals-en-france/)
- **Accessed:** 2025-10-25
- Cleaning: encoding normalization, French month parsing, split multi-genre values (`/`, `&`, `;`), basic dedup.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

## Story arc

Hook → Context → Insights → Implications  
- **Seasons:** monthly pattern; summer concentrates the most events; clear top months.  
- **Hotspots:** top regions and cities; top-3 share; Île-de-France treemap shows distribution across cities.  
- **Top Genres:** concentration in leading categories; regional flavor via heatmap; example for Nouvelle-Aquitaine.  
- **Map:** filter by region, city, date, genre.

## Limitations

The map relies on approximate geocoding when coordinates are missing.