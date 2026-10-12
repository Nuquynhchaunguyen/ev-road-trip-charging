import sys,unittest,importlib.util
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('updated_logic',P/'ev_logic.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

def row(name,lon,state='CA'):
 return dict(AddressTitle=name,Town='Test',Operator='Test',Latitude=36,Longitude=lon,StateOrProvince=state,ConnectionTypes='CCS',MaxPowerKW=150,FastChargeCount=4)
class SafetyTests(unittest.TestCase):
 def run_case(self,stations,routes,start=45,reserve=20):
  origin=(36,-121);dest=(36,-116)
  def road(a,b):
   key=(tuple(a),tuple(b))
   if key not in routes:raise ValueError('Unverified route')
   return dict(distance_miles=routes[key],duration_hours=1)
  return L.calculate_recommendations(pd.DataFrame(stations),battery_percent=start,full_range_miles=300,safety_reserve_pct=reserve,vehicle_connector='CCS',route_points=[origin,dest],route_lookup=road,amenity_lookup=lambda *x:([],'Test'),origin_coords=origin,destination_coords=dest,route_distance_miles=300)
 def test_arrival_floor_and_exact_boundary(self):
  o=(36,-121);d=(36,-116);a=(36,-120);b=(36,-119.8)
  f,g=self.run_case([row('Boundary',a[1]),row('Too far',b[1])],{(o,a):75,(o,b):75.3,(a,d):180,(b,d):170})
  self.assertEqual(f.AddressTitle.tolist(),['Boundary']);self.assertEqual(f.EstimatedBatteryAtArrivalPct.tolist(),[20]);self.assertEqual(g['safe_range_miles'],75)
 def test_projected_chain_cannot_override_actual_road_distance(self):
  o=(36,-121);d=(36,-116);a=(36,-120);b=(36,-118)
  f,_=self.run_case([row('First',a[1]),row('Later',b[1])],{(o,a):60,(o,b):200,(a,d):230,(b,d):100,(a,b):181})
  self.assertTrue(f.empty)
 def test_verified_chain_works(self):
  o=(36,-121);d=(36,-116);a=(36,-120);b=(36,-118)
  f,_=self.run_case([row('First',a[1]),row('Later',b[1])],{(o,a):60,(o,b):200,(a,d):230,(b,d):100,(a,b):175})
  self.assertEqual(f.AddressTitle.tolist(),['First']);self.assertEqual(f.NextChargingStop.tolist(),['Later']);self.assertEqual(f.NextLegStatus.tolist(),['Next charge: Later'])
 def test_routing_failure_does_not_create_chain(self):
  o=(36,-121);d=(36,-116);a=(36,-120);b=(36,-118)
  f,_=self.run_case([row('First',a[1]),row('Later',b[1])],{(o,a):60,(o,b):200,(a,d):230,(b,d):100})
  self.assertTrue(f.empty)
 def test_exact_state_filter(self):
  o=(36,-121);d=(36,-116);a=(36,-120);b=(36,-119.9)
  f,g=self.run_case([row('CA',a[1]),row('Carolina',b[1],'North Carolina')],{(o,a):65,(o,b):70,(a,d):175,(b,d):170})
  self.assertEqual(f.AddressTitle.tolist(),['CA']);self.assertEqual(g['outside_california_count'],1)
 def test_no_usable_range_when_start_below_reserve(self):
  o=(36,-121);d=(36,-116);a=(36,-120)
  f,g=self.run_case([row('First',a[1])],{(o,a):5,(a,d):170},start=19)
  self.assertTrue(f.empty);self.assertEqual(g['safe_range_miles'],0)
if __name__=='__main__':unittest.main(verbosity=2)
