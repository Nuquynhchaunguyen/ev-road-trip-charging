from html import escape
import math
import os
from pathlib import Path
import time

import folium
import pandas as pd
import requests
import streamlit as st

from ev_logic import DESTINATION, ORIGIN, calculate_recommendations
from local_cache import LocalCache

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "california_fast_chargers.csv"
DISK_CACHE = LocalCache(BASE_DIR / ".cache/lookup_cache.sqlite3")
NOMINATIM_URL = os.environ.get(
    "EV_GEOCODER_URL", "https://nominatim.openstreetmap.org/search"
)
DEFAULT_INPUTS = {
    "origin_name": "San Jose, CA", "destination_name": "Lake Tahoe, CA",
    "battery_percent": 45, "full_range_miles": 300,
    "safety_reserve_pct": 20, "vehicle_connector": "NACS",
    "charge_target_pct": 80, "battery_capacity_kwh": 75,
    "electricity_price_per_kwh": .45,
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
    cached = DISK_CACHE.get("geocoder-v1", [NOMINATIM_URL, normalized], 7 * 86400)
    if cached is not None:
        return tuple(cached[0]), cached[1]
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
        result = (point, str(results[0].get("display_name", place)))
        DISK_CACHE.put("geocoder-v1", [NOMINATIM_URL, normalized], result)
        return result
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
    key = [list(start), list(end)]
    kind = "osrm-geometry-v1" if geometry else "osrm-distance-v1"
    cached = DISK_CACHE.get(kind, key, 86400)
    if cached is not None:
        return cached
    unavailable = DISK_CACHE.get("osrm-unavailable-v1", key, 120)
    if unavailable is not None:
        raise ValueError(unavailable)
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
        DISK_CACHE.put(kind, key, result)
        if geometry:
            DISK_CACHE.put("osrm-distance-v1", key, {
                "distance_miles": result["distance_miles"],
                "duration_hours": result["duration_hours"],
            })
        return result
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        message = f"Driving-route lookup failed: {exc}"
        DISK_CACHE.put("osrm-unavailable-v1", key, message)
        raise ValueError(message) from exc
    finally:
        # Runs only on cache misses; pace requests to the public demo service.
        time.sleep(1.0)


def fetch_routes_many(start, ends):
    """Fetch one-to-many actual road distances; no straight-line fallback."""
    values = [DISK_CACHE.get("osrm-distance-v1", [list(start), list(end)], 86400)
              for end in ends]
    missing = [index for index, value in enumerate(values)
               if value is None and DISK_CACHE.get("osrm-unavailable-v1",
                   [list(start), list(ends[index])], 120) is None]
    for offset in range(0, len(missing), 80):
        indexes = missing[offset:offset + 80]
        coords = [start] + [ends[index] for index in indexes]
        locations = ";".join(f"{lon},{lat}" for lat, lon in coords)
        try:
            response = requests.get(
                "https://router.project-osrm.org/table/v1/driving/" + locations,
                params={"sources": "0", "destinations": ";".join(
                    str(i) for i in range(1, len(coords))),
                    "annotations": "distance,duration", "skip_waypoints": "true"},
                headers={"User-Agent": "MSBA286-EV-Charging-Project"},
                timeout=(5, 30),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != "Ok" or payload.get("fallback_speed_cells"):
                raise ValueError("Batch routing did not return verified road distances.")
            distances, durations = payload["distances"][0], payload["durations"][0]
            if len(distances) != len(indexes) or len(durations) != len(indexes):
                raise ValueError("Incomplete batch road-distance response.")
            checked = []
            for distance, duration in zip(distances, durations):
                if distance is None or duration is None:
                    checked.append(None)
                    continue
                distance, duration = float(distance), float(duration)
                if not all(math.isfinite(v) and v >= 0 for v in (distance, duration)):
                    raise ValueError("Invalid batch road-distance response.")
                checked.append({"distance_miles": distance / 1609.344,
                                "duration_hours": duration / 3600})
            for index, value in zip(indexes, checked):
                values[index] = value
                if value is not None:
                    DISK_CACHE.put("osrm-distance-v1",
                                   [list(start), list(ends[index])], value)
                else:
                    DISK_CACHE.put("osrm-unavailable-v1",
                                   [list(start), list(ends[index])],
                                   "Driving-route lookup failed: no verified road route returned.")
        finally:
            time.sleep(1.0)
    return values


@st.cache_data(ttl=900, max_entries=128, show_spinner=False)
def fetch_amenities(lat, lon):
    """The notebook's 500 m query; preserve Type for its quality formula."""
    key = [lat, lon]
    cached = DISK_CACHE.get("amenities-500m-v1", key, 86400)
    if cached is not None:
        return cached[0], "Cached OpenStreetMap (last 24 hours)"
    unavailable = DISK_CACHE.get("amenities-error-v1", key, 120)
    if unavailable is not None:
        return [], unavailable
    query = f'''[out:json][timeout:5];(
      nwr(around:500,{lat},{lon})["amenity"~"toilets|restaurant|cafe|fast_food"];
      nwr(around:500,{lat},{lon})["shop"~"supermarket|convenience|department_store|mall"];
    );out center tags;'''
    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query}, timeout=(3, 8),
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
            amenity_key = (name.lower(), kind)
            if amenity_key not in seen:
                seen.add(amenity_key)
                amenities.append({"Name": name, "Type": kind})
        source = "Live OpenStreetMap" if amenities else "No amenities returned"
        DISK_CACHE.put("amenities-500m-v1", key, [amenities, source])
        return amenities, source
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        source = f"Amenity lookup unavailable: {exc}"
        DISK_CACHE.put("amenities-error-v1", key, source)
        return [], source


def formatted(value, digits=1, suffix=""):
    return "Unknown" if pd.isna(value) else f"{float(value):.{digits}f}{suffix}"


def reach_label(value):
    return "UNKNOWN" if pd.isna(value) else ("YES" if bool(value) else "NO")


def popup_html(row, rank, destination_name):
    # Treat names from the dataset and API as text, never as executable HTML.
    label = lambda value: escape(str(value))
    direct = row['CanReachDestination']
    destination_battery = (
        f"<b>Battery at Destination:</b> {formatted(row['EstimatedBatteryAtDestinationPct'], 1, '%')}<br>"
        if pd.notna(direct) and bool(direct)
        else "<b>Trip continuation:</b> Further charging needed before destination<br>"
    )
    return f"""
    <div style="font:14px/1.4 Arial;min-width:260px;max-width:390px">
    <b>Recommendation #{rank}</b><br><br>
    <b>Station:</b> {label(row['AddressTitle'])}<br>
    <b>Town:</b> {label(row['Town'])}<br>
    <b>Operator:</b> {label(row['Operator'])}<br>
    <b>Power:</b> {formatted(row['MaxPowerKW'], 0, ' kW')}<br>
    <b>Recorded Fast Charger Count:</b> {formatted(row['FastChargeCount'], 0)}<br><br>
    <b>Battery at Charger:</b> {formatted(row['EstimatedBatteryAtArrivalPct'], 1, '%')}<br>
    <b>Charge Target:</b> {formatted(row['ChargeTargetPct'], 0, '%')}<br>
    <b>Estimated Charge Time:</b> {formatted(row['RealisticChargeTimeMin'], 1, ' min')}<br><br>
    <b>Estimated Charging Cost:</b> ${formatted(row['EstimatedChargingCost'], 2)}<br>
    <b>Distance from Route:</b> {formatted(row['DistanceToActualRouteMiles'], 2, ' miles')}<br>
    <b>Distance to {label(destination_name)}:</b> {formatted(row['DistanceToDestinationMiles'], 1, ' miles')}<br>
    {destination_battery}
    <b>Next Step:</b> {label(row['NextLegStatus'])}<br><br>
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




def show_results(frame, map_html, inputs, diagnostics=None, route=None,
                 resolved_locations=None):
    if not frame.empty and (pd.to_numeric(frame["EstimatedBatteryAtArrivalPct"], errors="coerce").isna().any()
                            or (frame["EstimatedBatteryAtArrivalPct"] < inputs["safety_reserve_pct"]).any()):
        st.warning("These results no longer meet your arrival reserve. Calculate the trip again.")
        return
    st.subheader("Trip Summary")
    safe_range = max(0, inputs["battery_percent"] - inputs["safety_reserve_pct"]) / 100 * inputs["full_range_miles"]
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
        f"Battery capacity: {inputs['battery_capacity_kwh']} kWh; "
        "recommended charge target = 80%. Drivers may continue charging if needed. "
        f"The car must retain at least {inputs['safety_reserve_pct']}% battery at every planned stop."
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
            st.warning("No chargers meet the route, charging-window and selected arrival-reserve rules. "
                       "Adjust the trip inputs and calculate again.")
    else:
        heading = ("Optional Charging Stops" if diagnostics and not diagnostics.get("charging_needed")
                   else f"Top {len(frame)} Recommended Next Charging Stop" + ("s" if len(frame) != 1 else ""))
        st.subheader(heading)
        st.caption("Choose one of these reachable next stops. For long trips, the app verifies that later compatible stops are reachable within the battery reserve. Live charger availability is not verified.")
        columns = {
            "AddressTitle": "Station", "Town": "Town",
            "EstimatedBatteryAtArrivalPct": "Battery at charger (%)",
            "RealisticChargeTimeMin": "Time to 80% (min)",
            "EstimatedChargingCost": "Cost to 80% ($)",
            "NextLegStatus": "After this stop",
        }
        table = frame[list(columns)].copy()
        table.index = range(1, len(table) + 1)
        table.index.name = "Rank"
        st.dataframe(table.rename(columns=columns).round(2), width="stretch")
        st.caption(
            f"Estimated cost uses ${inputs['electricity_price_per_kwh']:.2f}/kWh and excludes session, idle, parking, and tax fees. "
            "Check the charging network's live price before starting the session."
        )
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
            st.write(f"**Distance from route:** {row['DistanceToActualRouteMiles']:.2f} miles")
            st.write(f"**Rated power:** {row['MaxPowerKW']:.0f} kW")
            st.write(f"**Recommendation score:** {row['RecommendationScore']:.1f}")
    if diagnostics:
        for warning in diagnostics.get("warnings", []):
            st.warning(warning)
        with st.expander("Calculation details"):
            st.json(diagnostics)
    left, right = st.columns(2)
    left.download_button("Download map", map_html, file_name="ev_route_map.html", mime="text/html")
    right.download_button("Download results", frame.to_csv(index=False).encode("utf-8-sig"),
                          file_name="charging_recommendations.csv", mime="text/csv")



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
                            help="Minimum battery percentage retained on arrival at every planned stop.")
        with st.expander("Cost assumptions"):
            battery_capacity = st.number_input("Battery Capacity (kWh)", 30, 150, 75)
            electricity_price = st.number_input("Estimated Price ($/kWh)", 0.00, 1.50, 0.45, 0.01,
                                                help="Use a station/app price when available; this value is an estimate.")
            st.caption("Recommendations use an 80% charge target. You may continue charging at the station if needed.")
        connector = st.selectbox("Connector Type", ["NACS", "CCS"])
        submitted = st.form_submit_button("Find Best Charging Stops")
    st.sidebar.caption("California charger data · Place search: © OpenStreetMap contributors")
    if submitted:
        st.session_state.pop("live_result", None)
        st.session_state["attempted_search"] = True
        inputs = {"origin_name": origin_name.strip(),
                  "destination_name": destination_name.strip(),
                  "battery_percent": battery, "full_range_miles": full_range,
                  "safety_reserve_pct": reserve, "vehicle_connector": connector,
                  "charge_target_pct": 80,
                  "battery_capacity_kwh": battery_capacity,
                  "electricity_price_per_kwh": electricity_price}
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
                route_lookup_many=fetch_routes_many,
                progress=lambda fraction, message: progress.progress(
                    min(1.0, max(0.0, fraction)), text=message),
                origin_coords=origin_coords,
                destination_coords=destination_coords,
                route_distance_miles=route["distance_miles"],
                charge_target_pct=80,
                battery_capacity_kwh=battery_capacity,
                electricity_price_per_kwh=electricity_price,
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
        st.info("Enter your trip details and select Find Best Charging Stops to calculate reachable charging options.")



if __name__ == "__main__":
    main()
