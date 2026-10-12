import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
import ev_logic as L
from local_cache import LocalCache

class SpeedTests(unittest.TestCase):
    def test_only_needed_routes_are_requested(self):
        origin = (36, -121)
        destination = (36, -117)
        rows = [dict(AddressTitle='First', Town='Test', Operator='Test',
                     Latitude=36, Longitude=-120, StateOrProvince='CA',
                     ConnectionTypes='CCS', MaxPowerKW=150, FastChargeCount=4)]
        for i in range(80):
            rows.append(dict(rows[0], AddressTitle=f'Later {i}',
                             Longitude=-118.8 + i * .02))
        calls = []
        def road(a, b):
            calls.append((a, b))
            if a == origin and b == (36, -120):
                return dict(distance_miles=60, duration_hours=1)
            if a == (36, -120) and b == destination:
                return dict(distance_miles=175, duration_hours=3)
            raise AssertionError('Unnecessary route request')
        result, diagnostics = L.calculate_recommendations(
            pd.DataFrame(rows), 45, 300, 20, 'CCS', [origin, destination],
            road, lambda *args: ([], 'Test'), origin_coords=origin,
            destination_coords=destination, route_distance_miles=235)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.EstimatedBatteryAtArrivalPct.iloc[0], 25)
        self.assertEqual(len(calls), 2)
        self.assertEqual(diagnostics['first_stop_lookup_count'], 1)

    def test_cache_survives_restart_and_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'routes.sqlite3'
            first = LocalCache(path)
            key = [[36, -121], [36, -120]]
            with patch('local_cache.time.time', return_value=100):
                first.put('route', key, {'distance_miles': 60})
            second = LocalCache(path)
            with patch('local_cache.time.time', return_value=105):
                self.assertEqual(second.get('route', key, 10), {'distance_miles': 60})
            with patch('local_cache.time.time', return_value=110):
                self.assertIsNone(second.get('route', key, 10))
            self.assertIsNone(second.get('route', list(reversed(key)), 10000))

    def test_unwritable_cache_does_not_crash(self):
        cache = LocalCache('/dev/null/invalid.sqlite3')
        cache.put('route', [], {'distance_miles': 60})
        self.assertIsNone(cache.get('route', [], 86400))

    def test_route_disk_hit_avoids_network_and_sleep(self):
        import app
        app.fetch_route.clear()
        with tempfile.TemporaryDirectory() as temp:
            cache = LocalCache(Path(temp) / 'routes.sqlite3')
            key = [[36, -121], [36, -120]]
            cache.put('osrm-distance-v1', key,
                      {'distance_miles': 60, 'duration_hours': 1})
            with patch.object(app, 'DISK_CACHE', cache), \
                 patch.object(app.requests, 'get', side_effect=AssertionError('Network called')), \
                 patch.object(app.time, 'sleep', side_effect=AssertionError('Sleep called')):
                self.assertEqual(app.fetch_route(tuple(key[0]), tuple(key[1]))['distance_miles'], 60)
        app.fetch_route.clear()

    def test_batch_returns_road_distances_and_preserves_unknown_routes(self):
        import app
        with tempfile.TemporaryDirectory() as temp:
            cache = LocalCache(Path(temp) / 'routes.sqlite3')
            class Reply:
                def raise_for_status(self): pass
                def json(self):
                    return {'code':'Ok','distances':[[1609.344,None]],
                            'durations':[[3600,None]]}
            with patch.object(app,'DISK_CACHE',cache), \
                 patch.object(app.requests,'get',return_value=Reply()) as get, \
                 patch.object(app.time,'sleep'):
                result = app.fetch_routes_many((36,-121),[(36,-120),(36,-119)])
                self.assertAlmostEqual(result[0]['distance_miles'],1)
                self.assertIsNone(result[1])
                self.assertNotIn('fallback_speed',get.call_args.kwargs['params'])
                self.assertIsNone(cache.get('osrm-distance-v1',[[36,-121],[36,-119]],86400))
                # A cached subset requires no request at all.
                get.reset_mock()
                self.assertEqual(len(app.fetch_routes_many((36,-121),[(36,-120)])),1)
                get.assert_not_called()

    def test_amenities_cache_uses_station_coordinates(self):
        import app
        app.fetch_amenities.clear()
        with tempfile.TemporaryDirectory() as temp:
            cache = LocalCache(Path(temp) / 'routes.sqlite3')
            class Reply:
                def raise_for_status(self): pass
                def json(self):
                    return {'elements':[{'tags':{'name':'Cafe','amenity':'cafe'}},
                                        {'tags':{'amenity':'toilets'}}]}
            with patch.object(app,'DISK_CACHE',cache), \
                 patch.object(app.requests,'get',return_value=Reply()) as get:
                self.assertEqual(len(app.fetch_amenities(36.,-120.)[0]),2)
                app.fetch_amenities.clear()
                result = app.fetch_amenities(36.,-120.)
                self.assertEqual(len(result[0]),2)
                self.assertIn('Cached',result[1])
                self.assertEqual(get.call_count,1)
        app.fetch_amenities.clear()

if __name__ == '__main__':
    unittest.main(verbosity=2)
