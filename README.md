# EV Road Trip Charging Recommendation

A Python and Streamlit app for planning fast-charging stops on California road trips. It uses charging-station data, actual driving routes, estimated battery levels and charging times, and nearby amenities. The recommendations are planning estimates; live charger availability is not verified.

## Features

- Enter an origin, destination, starting battery, estimated full range, arrival reserve, and NACS or CCS connector type.
- Check actual road distances to candidate chargers and verify that the next leg can reach the destination or another compatible charger while preserving the selected arrival reserve.
- Compare ranked stops by charging power, route distance, estimated arrival battery, charging time, and nearby amenities.
- View the route and recommended stops on a map, and download the results.

## Data and attribution

`california_fast_chargers.csv` is a California fast-charger subset of [Global EV Charging Station Network (2010–2026)](https://www.kaggle.com/datasets/sohails07/global-ev-charging-station-network-2010-2026) by sohails07. The dataset credits [Open Charge Map](https://openchargemap.org/) as its source. Open Charge Map describes its contributor data license as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Station information is a historical snapshot and should be checked before travel.

Place search uses OpenStreetMap Nominatim, road distances use the public OSRM service, and nearby amenity searches use OpenStreetMap Overpass. These public services may be slow or unavailable. Successful lookups are cached locally to reduce repeat requests.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For Streamlit Community Cloud, deploy `app.py` from this repository. App visibility is a separate setting in Streamlit Community Cloud.

## Checks

```bash
python -m unittest discover -s tests -p 'test_reserve_update.py' -v
python tests/test_reserve_ui.py
python -m unittest discover -s tests -p 'test_speed_cache.py' -v
```

The app assumes an 80% charging target and average effective charging power equal to 65% of the station's rated power. It does not verify live charger availability, current prices, traffic, or vehicle-specific energy use.
