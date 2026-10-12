"""Notebook-compatible EV recommendations; all network access is injected.

Distances use miles, power uses kW, and battery/reserve inputs use 0–100 percent.
The model follows notebook 03_actual_route_and_map_backup.ipynb, including its
two-stage ranking. Unknown destination routes remain unknown, not reachable.
"""
import numpy as np
import pandas as pd

ORIGIN = (37.3382, -121.8863)
DESTINATION = (38.9399, -119.9772)
REQUIRED_COLUMNS = ["Latitude", "Longitude", "MaxPowerKW", "FastChargeCount",
                    "ConnectionTypes", "AddressTitle", "Town", "Operator"]
RESULT_COLUMNS = REQUIRED_COLUMNS + [
    "DistanceToActualRouteMiles", "ActualDrivingDistanceMiles",
    "EstimatedBatteryAtArrivalPct", "AmenityScore", "AmenityType",
    "PowerScore", "BatteryScore", "RouteScore", "CapacityScore",
    "RecommendationScore", "EnergyNeededKWh", "EffectiveChargingPowerKW",
    "RealisticChargeTimeMin", "DistanceToDestinationMiles",
    "DrivingTimeToDestinationHours", "EstimatedBatteryAtDestinationPct",
    "CanReachDestination", "NearbyAmenities", "NearbyAmenityCount",
    "AmenitySource", "AmenityQualityScore", "AmenityList", "PopupAmenities", "Rank",
    "ChargeTargetPct", "EstimatedChargingCost", "CanContinueSafely",
    "NextLegStatus", "RouteProgressMiles", "NextChargingStop",
]
AMENITY_KEYWORDS = ("mall", "shopping", "shops", "market", "marketplace", "plaza",
    "promenade", "costco", "target", "walmart", "raleys", "raley's", "hotel",
    "hilton", "burger", "restaurant", "cafe", "coffee", "grocery", "village", "center")


def distance_to_actual_route(lat, lon, route_points):
    """Notebook cell 5, vectorized across every segment (no route sampling)."""
    points = np.asarray(route_points, dtype=float)
    if len(points) < 2:
        return float("inf")
    starts, ends = points[:-1], points[1:]
    scale = 69.0 * np.cos(np.radians((starts[:, 0] + ends[:, 0] + lat) / 3))
    x, y = (lon - starts[:, 1]) * scale, (lat - starts[:, 0]) * 69.0
    dx, dy = (ends[:, 1] - starts[:, 1]) * scale, (ends[:, 0] - starts[:, 0]) * 69.0
    squared_length = dx * dx + dy * dy
    t = np.divide(x * dx + y * dy, squared_length,
                  out=np.zeros_like(squared_length), where=squared_length != 0)
    t = np.clip(t, 0, 1)
    return float(np.hypot(x - t * dx, y - t * dy).min())


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + (
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(3958.8 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def route_progress_miles(lat, lon, route_points):
    """Approximate a location's milepost along an ordered route polyline."""
    points = np.asarray(route_points, dtype=float)
    if len(points) < 2:
        return float("nan")
    radians = np.radians(points)
    delta = np.diff(radians, axis=0)
    h = (np.sin(delta[:, 0] / 2) ** 2
         + np.cos(radians[:-1, 0]) * np.cos(radians[1:, 0])
         * np.sin(delta[:, 1] / 2) ** 2)
    cumulative = np.concatenate(([0.0], np.cumsum(
        3958.8 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))))
    scale = np.cos(np.radians(lat))
    nearest = np.argmin((points[:, 0] - lat) ** 2 + ((points[:, 1] - lon) * scale) ** 2)
    return float(cumulative[nearest])


def charging_time_minutes(arrival_pct, target_pct, battery_kwh, rated_power_kw):
    """Planning estimate with slower charging above 80% state of charge."""
    arrival = float(np.clip(arrival_pct, 0, 100))
    target = float(np.clip(target_pct, arrival, 100))
    power = max(float(rated_power_kw), 1.0)
    fast_end = min(target, 80.0)
    fast_energy = max(fast_end - arrival, 0) / 100 * battery_kwh
    slow_energy = max(target - max(arrival, 80.0), 0) / 100 * battery_kwh
    # 65% of rated power through 80%; 30% above 80% approximates tapering.
    return (fast_energy / (power * .65) + slow_energy / (power * .30)) * 60


def classify_amenity(name):
    name = str(name).lower()
    groups = [
        ("Shopping", ("mall", "shopping", "shops", "promenade", "plaza", "marketplace", "center", "village")),
        ("Grocery / Shopping", ("raleys", "raley's", "market", "grocery", "costco", "target", "walmart")),
        ("Food", ("restaurant", "burger", "cafe", "coffee")),
        ("Hotel / Restroom", ("hotel", "hilton", "inn", "suites")),
    ]
    return next((label for label, words in groups if any(w in name for w in words)), "Unknown")


def fallback_amenities(name):
    name = str(name).lower()
    if "promenade" in name:
        return [{"Name": "The Promenade Shops at Orchard Valley", "Type": "shopping"}]
    if "raley" in name:
        return [{"Name": "Raley's", "Type": "supermarket"}]
    if any(word in name for word in ("mall", "shopping", "shops", "plaza", "marketplace")):
        return [{"Name": "Shopping Center", "Type": "shopping"}]
    return []


def amenity_quality_score(items):
    types = {str(item.get("Type", "")).lower() for item in items}
    return (0.30 * bool(types & {"toilets"})
            + 0.25 * bool(types & {"restaurant", "fast_food", "cafe"})
            + 0.25 * bool(types & {"supermarket", "convenience", "department_store"})
            + 0.20 * bool(types & {"mall", "shopping"}))


def score_candidates(df, final=False):
    """Return scored copy, using normalization within the supplied candidate set.

    Preliminary weights: power .30, battery .25, route .15, capacity .10,
    name amenity .20. Final top-three weights: .25, .25, .20, .20, .10.
    """
    result = df.copy()
    result["PowerScore"] = result["MaxPowerKW"] / result["MaxPowerKW"].max()
    result["BatteryScore"] = (result["EstimatedBatteryAtArrivalPct"] / 100).clip(0, 1)
    result["RouteScore"] = (1 - result["DistanceToActualRouteMiles"] / 5).clip(0, 1)
    result["CapacityScore"] = result["FastChargeCount"] / result["FastChargeCount"].max()
    weights = (0.25, 0.25, 0.20, 0.20, 0.10) if final else (0.30, 0.25, 0.15, 0.10, 0.20)
    columns = ["PowerScore", "BatteryScore", "RouteScore", "CapacityScore",
               "AmenityQualityScore" if final else "AmenityScore"]
    result["RecommendationScore"] = sum(result[c] * w for c, w in zip(columns, weights)) * 100
    return result


def _route_values(route_lookup, start, end):
    data = route_lookup(tuple(start), tuple(end))
    try:
        distance, hours = float(data["distance_miles"]), float(data["duration_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Routing result must contain distance_miles and duration_hours") from exc
    if not np.isfinite([distance, hours]).all() or min(distance, hours) < 0:
        raise ValueError("Routing result must contain finite, nonnegative distance and duration")
    return distance, hours


def calculate_recommendations(chargers, battery_percent, full_range_miles,
                              safety_reserve_pct, vehicle_connector, route_points,
                              route_lookup, amenity_lookup, progress=None,
                              origin_coords=ORIGIN, destination_coords=DESTINATION,
                              route_distance_miles=None, charge_target_pct=80,
                              battery_capacity_kwh=75, electricity_price_per_kwh=.45,
                              route_lookup_many=None):
    """Return (ranked DataFrame, JSON-compatible diagnostics).

    route_lookup(start, end) returns distance_miles and duration_hours.
    amenity_lookup(lat, lon) returns (list of {Name, Type}, source string).
    Optional progress(fraction, message) receives values between zero and one.
    Routing failures exclude origin candidates; destination failures use pd.NA.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(chargers.columns))
    if missing:
        raise ValueError("Missing charger columns: " + ", ".join(missing))
    if not np.isfinite([battery_percent, full_range_miles, safety_reserve_pct,
                        charge_target_pct, battery_capacity_kwh,
                        electricity_price_per_kwh]).all():
        raise ValueError("Battery, range, and reserve must be finite numbers")
    if not (0 <= battery_percent <= 100 and full_range_miles > 0 and 0 <= safety_reserve_pct <= 100):
        raise ValueError("Battery/reserve must be 0–100 percent; full range must be positive")
    if not (safety_reserve_pct < charge_target_pct <= 100):
        raise ValueError("Charge target must be above the safety reserve and at most 100%")
    if battery_capacity_kwh <= 0 or electricity_price_per_kwh < 0:
        raise ValueError("Battery capacity must be positive and electricity price cannot be negative")
    if route_distance_miles is not None and (
        not np.isfinite(route_distance_miles) or route_distance_miles < 0
    ):
        raise ValueError("Route distance must be a finite, nonnegative number")
    endpoints = np.asarray([origin_coords, destination_coords], dtype=float)
    if (endpoints.shape != (2, 2) or not np.isfinite(endpoints).all()
            or (np.abs(endpoints[:, 0]) > 90).any()
            or (np.abs(endpoints[:, 1]) > 180).any()):
        raise ValueError("Origin and destination must be valid [latitude, longitude] coordinates")
    origin_coords, destination_coords = tuple(endpoints[0]), tuple(endpoints[1])
    connector = str(vehicle_connector).strip().upper()
    if connector not in {"NACS", "CCS"}:
        raise ValueError("Vehicle connector must be NACS or CCS")
    diagnostic = {key: 0 for key in ("invalid_rows", "route_candidate_count", "compatible_count",
        "routing_failures", "window_count", "unique_count", "safe_count", "selected_count",
        "destination_routing_failures", "amenity_lookup_failures")}
    diagnostic.update(input_count=int(len(chargers)), warnings=[])
    # Reserve is a battery floor: with 45% battery and a 20% reserve, only 25%
    # of the full range is available before the next stop.
    safe_range = max(0.0, battery_percent - safety_reserve_pct) / 100 * full_range_miles
    charged_leg_range = max(0.0, charge_target_pct - safety_reserve_pct) / 100 * full_range_miles
    estimated_battery_at_destination = (
        None if route_distance_miles is None
        else battery_percent - route_distance_miles / full_range_miles * 100
    )
    charging_needed = (
        None if route_distance_miles is None
        else route_distance_miles > safe_range or estimated_battery_at_destination < 10
    )
    optional_trip = charging_needed is False
    short_trip = optional_trip and route_distance_miles < safe_range * .60
    min_stop = 0.0 if optional_trip else safe_range * .60
    max_stop = min(safe_range, route_distance_miles + 5) if optional_trip else safe_range
    diagnostic.update(
        safe_range_miles=float(safe_range),
        charged_leg_range_miles=float(charged_leg_range),
        reserve_battery_pct=float(safety_reserve_pct),
        charge_target_pct=float(charge_target_pct),
        min_stop_distance_miles=float(min_stop),
        max_stop_distance_miles=float(max_stop),
        optional_trip=optional_trip,
        optional_short_trip=bool(short_trip),
        charging_needed=charging_needed,
        estimated_battery_at_destination_without_charge_pct=estimated_battery_at_destination,
    )

    def empty(message, warn=True):
        if warn and not optional_trip:
            diagnostic["warnings"].append(message)
        return pd.DataFrame(columns=list(dict.fromkeys(list(chargers.columns) + RESULT_COLUMNS))), diagnostic

    def report(fraction, message):
        if progress is not None:
            progress(float(fraction), message)

    points = np.asarray(route_points, dtype=float)
    if len(points) < 2:
        return empty("No usable driving route was supplied.")
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or (np.abs(points[:, 0]) > 90).any() or (np.abs(points[:, 1]) > 180).any()):
        raise ValueError("Route points must be finite [latitude, longitude] coordinates")
    frame = chargers.copy().reset_index(drop=True)
    if "StateOrProvince" in frame:
        in_california = frame.StateOrProvince.astype("string").str.strip().str.casefold().isin(["ca", "california"])
        diagnostic["outside_california_count"] = int((~in_california).sum())
        frame = frame.loc[in_california].copy()
    numeric = ["Latitude", "Longitude", "MaxPowerKW", "FastChargeCount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(frame[numeric]).all(axis=1) & frame.Latitude.between(-90, 90)
             & frame.Longitude.between(-180, 180) & (frame.MaxPowerKW > 0) & (frame.FastChargeCount > 0))
    diagnostic["invalid_rows"] = int((~valid).sum())
    if diagnostic["invalid_rows"]:
        diagnostic["warnings"].append(f"Dropped {diagnostic['invalid_rows']} rows with invalid coordinates, power, or charger count.")
    frame = frame.loc[valid].copy()
    if frame.empty:
        return empty("No valid charger records remain.")
    report(.03, "Measuring distance to every route segment")
    frame["DistanceToActualRouteMiles"] = [distance_to_actual_route(lat, lon, points)
                                          for lat, lon in zip(frame.Latitude, frame.Longitude)]
    frame = frame.loc[frame.DistanceToActualRouteMiles <= 5].copy()
    frame["RouteProgressMiles"] = [route_progress_miles(lat, lon, points)
                                    for lat, lon in zip(frame.Latitude, frame.Longitude)]
    diagnostic["route_candidate_count"] = int(len(frame))
    pattern = {"NACS": "NACS|Tesla Supercharger", "CCS": "CCS"}[connector]
    frame = frame.loc[frame.ConnectionTypes.astype("string").str.contains(pattern, case=False, na=False)].copy()
    diagnostic["compatible_count"] = int(len(frame))
    if frame.empty:
        return empty("No compatible chargers within five miles of the route.")
    # Reject impossible first stops locally before requesting road directions.
    # Straight-line distance is only a rejection bound; it never proves safety.
    first = frame.loc[[haversine_miles(*origin_coords, lat, lon) <= max_stop
                       for lat, lon in zip(frame.Latitude, frame.Longitude)]].copy()
    diagnostic["first_stop_lookup_count"] = int(len(first))
    if first.empty:
        return empty("No compatible first stops within the available battery range.")
    route_cache = {}
    def checked_route(start, end):
        key = (tuple(start), tuple(end))
        if key not in route_cache:
            try:
                route_cache[key] = _route_values(route_lookup, start, end)
            except Exception:
                route_cache[key] = (np.nan, np.nan)
        return route_cache[key]
    if route_lookup_many is not None:
        report(.10, f"Loading road distances for {len(first)} reachable candidates…")
        ends = [(row.Latitude, row.Longitude) for row in first.itertuples()]
        try:
            batch = route_lookup_many(origin_coords, ends)
            if len(batch) != len(ends):
                raise ValueError("Incomplete route-distance batch")
            # Validate the whole response before committing any results.
            validated = []
            for value in batch:
                if value is None:
                    validated.append((np.nan, np.nan))
                else:
                    validated.append(_route_values(lambda *_: value, origin_coords, origin_coords))
            for end, value in zip(ends, validated):
                route_cache[(origin_coords, tuple(end))] = value
        except Exception:
            diagnostic["warnings"].append("Batch routing unavailable; checking road distances individually.")
    distances = []
    for index, row in enumerate(first.itertuples()):
        distance, _ = checked_route(origin_coords, (row.Latitude, row.Longitude))
        if not np.isfinite(distance):
            diagnostic["routing_failures"] += 1
        distances.append(distance)
        report(.10 + .40 * (index + 1) / len(first),
               f"Checking reachable next stops: {index + 1}/{len(first)}")
    first["ActualDrivingDistanceMiles"] = distances
    first = first.loc[first.ActualDrivingDistanceMiles.between(min_stop, max_stop)].copy()
    if diagnostic["routing_failures"]:
        diagnostic["warnings"].append(f"Could not verify driving routes to {diagnostic['routing_failures']} candidates; excluded them.")
    if first.empty:
        return empty("No safe charging candidates in the preferred driving window.")

    # Only explore continuation paths needed by reachable first stops, rather
    # than checking every corridor station back to the origin and destination.
    # Each accepted edge still requires an actual, successfully routed road leg.
    continuation = {}
    corridor = list(frame.sort_values("RouteProgressMiles", ascending=False).itertuples())
    def safe_continuation(row):
        if row.Index in continuation:
            return continuation[row.Index]
        start = (row.Latitude, row.Longitude)
        distance, _ = checked_route(start, destination_coords)
        if np.isfinite(distance) and distance <= charged_leg_range:
            continuation[row.Index] = "Destination"
            return "Destination"
        for later in corridor:
            if later.RouteProgressMiles <= row.RouteProgressMiles:
                continue
            if haversine_miles(*start, later.Latitude, later.Longitude) > charged_leg_range:
                continue
            distance, _ = checked_route(start, (later.Latitude, later.Longitude))
            if not np.isfinite(distance) or distance > charged_leg_range:
                continue
            if safe_continuation(later) is not None:
                continuation[row.Index] = later.AddressTitle
                return later.AddressTitle
        continuation[row.Index] = None
        return None
    for count, row in enumerate(first.itertuples(), start=1):
        safe_continuation(row)
        report(.50 + .25 * count / len(first),
               f"Verifying onward charging routes: {count}/{len(first)}")
    first["NextChargingStop"] = first.index.map(continuation)
    first["CanContinueSafely"] = first.NextChargingStop.notna()
    frame = first.loc[first.CanContinueSafely].copy()
    diagnostic["verified_route_lookup_count"] = len(route_cache)
    diagnostic["window_count"] = int(len(frame))
    frame["LatRounded"], frame["LonRounded"] = frame.Latitude.round(3), frame.Longitude.round(3)
    frame = frame.sort_values("MaxPowerKW", ascending=False).drop_duplicates(["LatRounded", "LonRounded"]).copy()
    diagnostic["unique_count"] = int(len(frame))
    frame["EstimatedBatteryAtArrivalPct"] = battery_percent - frame.ActualDrivingDistanceMiles / full_range_miles * 100
    frame = frame.loc[frame.EstimatedBatteryAtArrivalPct >= safety_reserve_pct].copy()
    diagnostic["safe_count"] = int(len(frame))
    if frame.empty:
        message = ("No optional chargers passed the route and battery filters. "
                   "This trip does not require charging."
                   if optional_trip else
                   "No safe charging candidates in the preferred driving window.")
        return empty(message, warn=not optional_trip)
    frame["AmenityScore"] = frame.AddressTitle.fillna("").map(lambda s: float(any(k in str(s).lower() for k in AMENITY_KEYWORDS)))
    frame["AmenityType"] = frame.AddressTitle.map(classify_amenity)
    ranked = score_candidates(frame).sort_values("RecommendationScore", ascending=False)
    selected = []
    for idx, row in ranked.iterrows():
        if all(haversine_miles(row.Latitude, row.Longitude, ranked.loc[i, "Latitude"], ranked.loc[i, "Longitude"]) >= 1 for i in selected):
            selected.append(idx)
        if len(selected) == 3:
            break
    top = ranked.loc[selected].copy()
    diagnostic["selected_count"] = int(len(top))
    top["ChargeTargetPct"] = float(charge_target_pct)
    top["EnergyNeededKWh"] = ((charge_target_pct - top.EstimatedBatteryAtArrivalPct)
                               / 100 * battery_capacity_kwh).clip(lower=0)
    top["EffectiveChargingPowerKW"] = top.MaxPowerKW * .65
    top["RealisticChargeTimeMin"] = [
        charging_time_minutes(arrival, charge_target_pct, battery_capacity_kwh, power)
        for arrival, power in zip(top.EstimatedBatteryAtArrivalPct, top.MaxPowerKW)
    ]
    top["EstimatedChargingCost"] = top.EnergyNeededKWh * electricity_price_per_kwh
    destination_distances, destination_hours, amenities, sources = [], [], [], []
    for count, row in enumerate(top.itertuples(), start=1):
        try:
            distance, hours = checked_route((row.Latitude, row.Longitude), destination_coords)
        except Exception:
            distance, hours = np.nan, np.nan
            diagnostic["destination_routing_failures"] += 1
        if not np.isfinite(distance):
            diagnostic["destination_routing_failures"] += 1
        destination_distances.append(distance)
        destination_hours.append(hours)
        try:
            items, source = amenity_lookup(float(row.Latitude), float(row.Longitude))
            items = [item for item in items if isinstance(item, dict) and item.get("Name")]
        except Exception as exc:
            items, source = [], f"Amenity lookup unavailable: {exc}"
        source = str(source)
        if source.startswith("Amenity lookup unavailable:"):
            diagnostic["amenity_lookup_failures"] += 1
        if not items:
            items = fallback_amenities(row.AddressTitle)
            reason = source if source.startswith("Amenity lookup unavailable:") else "Live lookup returned no amenities"
            source = ("Fallback from station location" if items else "No amenity data available") + f" ({reason})"
        amenities.append(items)
        sources.append(str(source))
        report(.75 + .25 * count / len(top), f"Checking destination and amenities: {count}/{len(top)}")
    top["DistanceToDestinationMiles"], top["DrivingTimeToDestinationHours"] = destination_distances, destination_hours
    top["EstimatedBatteryAtDestinationPct"] = charge_target_pct - top.DistanceToDestinationMiles / full_range_miles * 100
    top["CanReachDestination"] = pd.array([
        pd.NA if not np.isfinite(d) else d <= charged_leg_range
        for d in destination_distances], dtype="boolean")
    top["NextLegStatus"] = [
        "Can reach destination with reserve" if direct else f"Next charge: {next_stop}"
        for direct, next_stop in zip(top.CanReachDestination.fillna(False), top.NextChargingStop)
    ]
    top["NearbyAmenities"], top["AmenitySource"] = amenities, sources
    top["NearbyAmenityCount"] = top.NearbyAmenities.map(len)
    top["AmenityQualityScore"] = top.NearbyAmenities.map(amenity_quality_score)
    top["AmenityList"] = top.NearbyAmenities.map(lambda items: ", ".join(str(x["Name"]) for x in items) or "No amenity data available")
    top["PopupAmenities"] = top.NearbyAmenities.map(lambda items: ", ".join(str(x["Name"]) for x in items[:6]) or "No amenity data available")
    top = score_candidates(top, final=True).sort_values("RecommendationScore", ascending=False).reset_index(drop=True)
    top["Rank"] = np.arange(1, len(top) + 1)
    for key, message in [("destination_routing_failures", "Destination route unavailable; reachability is unknown."),
                         ("amenity_lookup_failures", "Live amenity lookup failed; station-name fallback was used where available.")]:
        if diagnostic[key]:
            diagnostic["warnings"].append(message)
    return top, diagnostic
