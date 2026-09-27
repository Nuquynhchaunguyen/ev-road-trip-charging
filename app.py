from html import escape, unescape
import json
import math
import os
from pathlib import Path
import re
import time

import folium
import pandas as pd
import requests
import streamlit as st

from ev_logic import DESTINATION, ORIGIN, calculate_recommendations

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "california_fast_chargers.csv"
SNAPSHOT_PATH = BASE_DIR / "notebook_snapshot.json"
SAVED_MAP_PATH = BASE_DIR / "ev_route_map_saved.html"
NOMINATIM_URL = os.environ.get(
    "EV_GEOCODER_URL", "https://nominatim.openstreetmap.org/search"
)
DEFAULT_INPUTS = {
    "origin_name": "San Jose, CA", "destination_name": "Lake Tahoe, CA",
    "battery_percent": 45, "full_range_miles": 300,
    "safety_reserve_pct": 20, "vehicle_connector": "NACS",
}


@st.cache_data(ttl=86400, max_entries=128, show_spinner=False)
def geocode_place(place):
    """Resolve an edited city once, when the user submits the trip form."""
    normalized = " ".join(place.casefold().strip().split())
    known = {
        "san jose, ca": (ORIGIN, "San Jose, California"),
        "lake tahoe, ca": (DESTINATION, "South Lake Tahoe, California"),
    }
    if normalized in known:
        return known[normalized]
    if not normalized:
        raise ValueError("Origin and Destination cannot be empty.")
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": place, "format": "jsonv2", "limit": 1,
                    "countrycodes": "us"},
            headers={"User-Agent": "MSBA286-EV-Charging-Project/1.0"},
            timeout=(5, 15),
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            raise ValueError(f"Location not found: {place}")
        point = (float(results[0]["lat"]), float(results[0]["lon"]))
        if not (math.isfinite(point[0]) and math.isfinite(point[1])
                and -90 <= point[0] <= 90 and -180 <= point[1] <= 180):
            raise ValueError(f"Invalid coordinates for: {place}")
        return point, str(results[0].get("display_name", place))
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not locate {place}: {exc}") from exc
    finally:
        # Public Nominatim permits at most one request per second.
        time.sleep(1.1)


@st.cache_data(show_spinner=False)
def load_data(path, modified_ns):
    # The timestamp invalidates this cache if the CSV is replaced.
    return pd.read_csv(path)


@st.cache_data(ttl=3600, max_entries=512, show_spinner=False)
def fetch_route(start, end, geometry=False):
    """OSRM road distance, including the drive to/from the charger itself."""
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start[1]},{start[0]};{end[1]},{end[0]}"
    )
    try:
        response = requests.get(
            url,
            params={"overview": "full" if geometry else "false",
                    "geometries": "geojson", "steps": "false"},
            headers={"User-Agent": "MSBA286-EV-Charging-Project"},
            timeout=(5, 30),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok" or not payload.get("routes"):
            raise ValueError("OSRM returned no driving route.")
        route = payload["routes"][0]
        distance, duration = float(route["distance"]), float(route["duration"])
        if not all(math.isfinite(v) and v >= 0 for v in [distance, duration]):
            raise ValueError("OSRM returned invalid distance or duration.")
        result = {"distance_miles": distance / 1609.344,
                  "duration_hours": duration / 3600}
        if geometry:
            points = [[float(lat), float(lon)]
                      for lon, lat in route["geometry"]["coordinates"]]
            if len(points) < 2 or any(
                not (-90 <= lat <= 90 and -180 <= lon <= 180)
                for lat, lon in points
            ):
                raise ValueError("OSRM returned invalid route geometry.")
            result["route_points"] = points
        return result
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Driving-route lookup failed: {exc}") from exc
    finally:
        # Runs only on cache misses; pace requests to the public demo service.
        time.sleep(1.0)


@st.cache_data(ttl=900, max_entries=128, show_spinner=False)
def fetch_amenities(lat, lon):
    """The notebook's 500 m query; preserve Type for its quality formula."""
    query = f'''[out:json][timeout:12];(
      nwr(around:500,{lat},{lon})["amenity"~"toilets|restaurant|cafe|fast_food"];
      nwr(around:500,{lat},{lon})["shop"~"supermarket|convenience|department_store|mall"];
    );out center tags;'''
    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query}, timeout=(5, 15),
            headers={"User-Agent": "MSBA286-EV-Charging-Project"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("remark"):
            raise ValueError(str(payload["remark"]))
        amenities, seen = [], set()
        for element in payload["elements"]:
            tags = element.get("tags", {})
            kind = tags.get("amenity") or tags.get("shop") or "unknown"
            name = tags.get("name")
            if not name and kind == "toilets":
                name = "Public Restroom"
            if not name:
                continue
            key = (name.lower(), kind)
            if key not in seen:
                seen.add(key)
                amenities.append({"Name": name, "Type": kind})
        return amenities, "Live OpenStreetMap" if amenities else "No amenities returned"
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        return [], f"Amenity lookup unavailable: {exc}"


def formatted(value, digits=1, suffix=""):
    return "Unknown" if pd.isna(value) else f"{float(value):.{digits}f}{suffix}"


def reach_label(value):
    return "UNKNOWN" if pd.isna(value) else ("YES" if bool(value) else "NO")


def popup_html(row, rank, destination_name):
    # Treat names from the dataset and API as text, never as executable HTML.
    label = lambda value: escape(str(value))
    return f"""
    <div style="font:14px/1.4 Arial;min-width:260px;max-width:390px">
    <b>Recommendation #{rank}</b><br><br>
    <b>Station:</b> {label(row['AddressTitle'])}<br>
    <b>Town:</b> {label(row['Town'])}<br>
    <b>Operator:</b> {label(row['Operator'])}<br>
    <b>Power:</b> {formatted(row['MaxPowerKW'], 0, ' kW')}<br>
    <b>Recorded Fast Charger Count:</b> {formatted(row['FastChargeCount'], 0)}<br><br>
    <b>Battery at Charger:</b> {formatted(row['EstimatedBatteryAtArrivalPct'], 1, '%')}<br>
    <b>Charge Target:</b> 80%<br>
    <b>Estimated Charge Time:</b> {formatted(row['RealisticChargeTimeMin'], 1, ' min')}<br><br>
    <b>Distance from Route:</b> {formatted(row['DistanceToActualRouteMiles'], 2, ' miles')}<br>
    <b>Distance to {label(destination_name)}:</b> {formatted(row['DistanceToDestinationMiles'], 1, ' miles')}<br>
    <b>Estimated Battery at Destination:</b> {formatted(row['EstimatedBatteryAtDestinationPct'], 1, '%')}<br>
    <b>Can Reach Destination:</b> {reach_label(row['CanReachDestination'])}<br><br>
    <b>Nearby Amenities:</b> {label(row['PopupAmenities'])}<br><br>
    <b>Recommendation Score:</b> {formatted(row['RecommendationScore'])}
    </div>"""


def build_map(frame, route_points, origin_coords, destination_coords,
              origin_name, destination_name):
    route_map = folium.Map(location=[37.9, -120.8], zoom_start=7)
    folium.PolyLine(route_points, weight=5, color="#3388ff",
                    tooltip="Actual driving route").add_to(route_map)
    folium.Marker(origin_coords, popup=f"Start: {escape(origin_name)}",
                  tooltip="Origin").add_to(route_map)
    folium.Marker(destination_coords, popup=f"Destination: {escape(destination_name)}",
                  tooltip="Destination").add_to(route_map)
    bounds = list(route_points)
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        point = [float(row["Latitude"]), float(row["Longitude"])]
        folium.Marker(
            point, tooltip=f"Recommendation #{rank}",
            popup=folium.Popup(popup_html(row, rank, destination_name), max_width=420),
            icon=folium.Icon(color=["green", "blue", "orange"][rank - 1],
                             icon="bolt", prefix="fa"),
        ).add_to(route_map)
        bounds.append(point)
    route_map.fit_bounds(bounds)
    return route_map.get_root().render()


def load_saved_results():
    """Read the notebook's historical popup values without recomputing them."""
    original_map = BASE_DIR / "ev_route_map.html"
    map_path = original_map if original_map.exists() else SAVED_MAP_PATH
    html = map_path.read_text(encoding="utf-8")
    if SNAPSHOT_PATH.exists() and map_path == SAVED_MAP_PATH:
        rows = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["recommendations"]
        return pd.DataFrame(rows), html
    labels = {
        "Station": "AddressTitle", "Town": "Town", "Operator": "Operator",
        "Power": "MaxPowerKW", "Recorded Fast Charger Count": "FastChargeCount",
        "Battery at Charger": "EstimatedBatteryAtArrivalPct",
        "Estimated Charge Time": "RealisticChargeTimeMin",
        "Distance from Route": "DistanceToActualRouteMiles",
        "Distance to Lake Tahoe": "DistanceToDestinationMiles",
        "Estimated Battery at Destination": "EstimatedBatteryAtDestinationPct",
        "Can Reach Destination": "CanReachDestination",
        "Nearby Amenities": "PopupAmenities", "Recommendation Score": "RecommendationScore",
    }
    numbers = {
        "MaxPowerKW", "FastChargeCount", "EstimatedBatteryAtArrivalPct",
        "RealisticChargeTimeMin", "DistanceToActualRouteMiles",
        "DistanceToDestinationMiles", "EstimatedBatteryAtDestinationPct",
        "RecommendationScore",
    }
    records = []
    for rank, popup in re.findall(r"<b>Recommendation #(\d+)</b>(.*?)</div>", html, re.S):
        fields = {
            key: unescape(re.sub(r"<[^>]+>", "", value)).strip()
            for key, value in re.findall(
                r"<b>([^<]+):</b>\s*(.*?)(?=<br>|$)", popup, re.S
            )
        }
        record = {"Rank": int(rank), "AmenitySource": "Saved notebook map"}
        for label, key in labels.items():
            value = fields[label]
            if key in numbers:
                record[key] = float(re.match(r"[-+]?\d+(?:\.\d+)?", value).group())
            elif key == "CanReachDestination":
                record[key] = value == "YES"
            else:
                record[key] = value
        records.append(record)
    if not records:
        raise ValueError("No recommendations found in ev_route_map.html")
    return pd.DataFrame(records).sort_values("Rank"), html


def show_results(frame, map_html, inputs, saved=False, diagnostics=None, route=None,
                 resolved_locations=None):
    st.subheader("Trip Summary")
    safe_range = inputs["battery_percent"] / 100 * inputs["full_range_miles"] * (
        1 - inputs["safety_reserve_pct"] / 100
    )
    st.write(f"**{inputs['origin_name']} → {inputs['destination_name']}**")
    if resolved_locations:
        st.caption("Resolved locations: " + " → ".join(resolved_locations))
    cols = st.columns(4)
    cols[0].metric("Starting Battery", f"{inputs['battery_percent']}%")
    cols[1].metric("Full Range", f"{inputs['full_range_miles']} miles")
    cols[2].metric("Safe Driving Range", f"{safe_range:.1f} miles")
    cols[3].metric("Connector", inputs["vehicle_connector"])
    window_min = (diagnostics or {}).get("min_stop_distance_miles", safe_range * 0.60)
    window_max = (diagnostics or {}).get("max_stop_distance_miles", safe_range)
    st.caption(
        f"Charging window: {window_min:.1f}–{window_max:.1f} miles. "
        "Notebook assumptions: at least 10% battery on arrival; 75 kWh battery; "
        "charge to 80%; effective charging power = 65% of rated power. "
        "Safety reserve is a fraction of the remaining driving range."
    )
    if route:
        st.write(f"**Driving distance:** {route['distance_miles']:.1f} miles · "
                 f"**Driving time:** {route['duration_hours']:.2f} hours")
    if diagnostics and diagnostics.get("charging_needed") is False:
        estimate = diagnostics["estimated_battery_at_destination_without_charge_pct"]
        st.success(
            "No charging stop is needed for this trip. "
            f"Estimated battery at destination: {estimate:.1f}%. "
            "The chargers below are optional places to top up."
        )
    if frame.empty:
        if diagnostics and diagnostics.get("charging_needed") is False:
            st.info("No optional chargers passed the route and battery filters. "
                    "You can reach the destination without stopping to charge.")
        else:
            st.warning("No chargers meet the route, charging-window and 10% arrival filters. "
                       "Adjust the trip inputs and calculate again.")
    else:
        st.subheader(
            "Optional Charging Stops" if diagnostics and not diagnostics.get("charging_needed")
            else "Top 3 Recommended Charging Stops"
        )
        st.caption("These are alternative charging stops, not three stops to visit in sequence.")
        columns = {
            "AddressTitle": "Station", "Town": "Town", "MaxPowerKW": "Power (kW)",
            "FastChargeCount": "Recorded fast chargers",
            "EstimatedBatteryAtArrivalPct": "Battery at charger (%)",
            "RealisticChargeTimeMin": "Charge time (min)",
            "DistanceToActualRouteMiles": "Distance from route (miles)",
            "DistanceToDestinationMiles": "Distance to destination (miles)",
            "EstimatedBatteryAtDestinationPct": "Battery at destination (%)",
            "CanReachDestination": "Can reach destination",
            "RecommendationScore": "Score",
        }
        table = frame[list(columns)].copy()
        table["CanReachDestination"] = table["CanReachDestination"].map(reach_label)
        table.index = range(1, len(table) + 1)
        table.index.name = "Rank"
        st.dataframe(table.rename(columns=columns).round(2), width="stretch")
    st.subheader("Route Map")
    st.iframe(map_html, height=740)
    st.caption("Click a colored marker to view its battery, charging time, amenities and score.")
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        with st.expander(f"#{rank} · {row['AddressTitle']} · Score {row['RecommendationScore']:.1f}"):
            st.write(f"**Nearby amenities:** {row['PopupAmenities']}")
            if "AmenitySource" in row:
                st.caption(str(row["AmenitySource"]))
            st.write(f"**Operator:** {row['Operator']}")
            st.write(f"**Connection types:** {row.get('ConnectionTypes', 'Unknown')}")
    if diagnostics:
        for warning in diagnostics.get("warnings", []):
            st.warning(warning)
        with st.expander("Calculation details"):
            st.json(diagnostics)
    left, right = st.columns(2)
    left.download_button("Download map", map_html, file_name="ev_route_map.html", mime="text/html")
    right.download_button("Download results", frame.to_csv(index=False).encode("utf-8-sig"),
                          file_name="charging_recommendations.csv", mime="text/csv")
    if saved:
        st.caption("Saved values retain the rounding in the original map popups. "
                   "The map needs internet access to load its map tiles and JavaScript libraries.")


def main():
    st.set_page_config(page_title="EV Road Trip Charging Recommendation",
                       page_icon="⚡", layout="wide")
    st.markdown("""<style>
      section[data-testid="stSidebar"] {background: #f2f5f9;}
      .block-container {padding-top: 2.5rem;}
    </style>""", unsafe_allow_html=True)
    st.title("⚡ EV Road Trip Charging Recommendation")
    st.write("A data-driven decision-support tool for identifying "
             "recommended fast-charging stops during long-distance EV travel")
    st.sidebar.header("Trip Inputs")
    with st.sidebar.form("trip_inputs", border=False):
        origin_name = st.text_input("Origin", DEFAULT_INPUTS["origin_name"])
        destination_name = st.text_input("Destination", DEFAULT_INPUTS["destination_name"])
        battery = st.slider("Current Battery (%)", 10, 100, 45)
        full_range = st.number_input("Estimated Full Range (miles)", 100, 500, 300)
        reserve = st.slider("Safety Reserve (%)", 5, 30, 20,
                            help="Percentage of remaining range reserved, as in the notebook.")
        connector = st.selectbox("Connector Type", ["NACS", "CCS"])
        submitted = st.form_submit_button("Find Best Charging Stops")
    st.sidebar.caption("California charger data · Place search: © OpenStreetMap contributors")
    if submitted:
        st.session_state.pop("live_result", None)
        st.session_state["attempted_search"] = True
        inputs = {"origin_name": origin_name.strip(),
                  "destination_name": destination_name.strip(),
                  "battery_percent": battery, "full_range_miles": full_range,
                  "safety_reserve_pct": reserve, "vehicle_connector": connector}
        progress = st.progress(0, text="Finding your trip locations…")
        try:
            origin_coords, resolved_origin = geocode_place(inputs["origin_name"])
            destination_coords, resolved_destination = geocode_place(inputs["destination_name"])
            if origin_coords == destination_coords:
                raise ValueError("Origin and Destination must be different locations.")
            chargers = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
            route = fetch_route(origin_coords, destination_coords, geometry=True)
            frame, diagnostics = calculate_recommendations(
                chargers, battery_percent=battery, full_range_miles=full_range,
                safety_reserve_pct=reserve, vehicle_connector=connector,
                route_points=route["route_points"],
                route_lookup=fetch_route, amenity_lookup=fetch_amenities,
                progress=lambda fraction, message: progress.progress(
                    min(1.0, max(0.0, fraction)), text=message),
                origin_coords=origin_coords,
                destination_coords=destination_coords,
                route_distance_miles=route["distance_miles"],
            )
            map_html = build_map(
                frame, route["route_points"], origin_coords, destination_coords,
                inputs["origin_name"], inputs["destination_name"],
            )
            st.session_state["live_result"] = (
                frame, map_html, inputs, diagnostics, route,
                (resolved_origin, resolved_destination),
            )
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        finally:
            progress.empty()
    if "live_result" in st.session_state:
        frame, map_html, inputs, diagnostics, route, locations = st.session_state["live_result"]
        st.success("Trip calculated for this route.")
        show_results(frame, map_html, inputs, diagnostics=diagnostics,
                     route=route, resolved_locations=locations)
    elif not st.session_state.get("attempted_search"):
        st.caption("Example from the saved notebook · San Jose → Lake Tahoe · "
                   "45% battery · 300-mile range · 20% reserve · NACS")
        try:
            frame, map_html = load_saved_results()
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            st.error(f"Cannot read the saved notebook map: {exc}")
            return
        show_results(frame, map_html, DEFAULT_INPUTS, saved=True)


if __name__ == "__main__":
    main()
