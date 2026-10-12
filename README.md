# EV Road Trip Charging Recommendation

A Streamlit app for recommending fast-charging stops on California road trips. It uses station data, road routes, estimated battery levels and charging times, and nearby amenities. Recommendations are planning estimates; live charger availability is not verified.

## Data and attribution

`california_fast_chargers.csv` is a California fast-charger subset of [Global EV Charging Station Network (2010–2026)](https://www.kaggle.com/datasets/sohails07/global-ev-charging-station-network-2010-2026) by sohails07. The dataset credits [Open Charge Map](https://openchargemap.org/) as its source. Open Charge Map describes its contributor data license as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Station information is a historical snapshot and should be checked before travel.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For Streamlit Community Cloud, deploy `app.py` from this repository. App visibility is a separate setting in Streamlit Community Cloud.
