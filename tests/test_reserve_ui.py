from pathlib import Path
import sys
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
import app
st.cache_data.clear()
def live_view():
 import app
 app.main()
a=AppTest.from_function(live_view,default_timeout=15).run()
assert not a.exception,a.exception
assert len(a.dataframe)==0
assert any('Enter your trip' in x.value for x in a.info)
assert not any('saved notebook' in x.value.casefold() for x in a.caption)
# Unknown road routes clear previous output rather than expose stored examples.
with patch.object(app,'fetch_route',side_effect=ValueError('Route unavailable')):
 a.button[0].click().run()
assert not a.exception,a.exception
assert len(a.error)==1 and len(a.dataframe)==0
# A stale session result below the current reserve must never reach the table/map.
def stale_view():
 import app,pandas as pd
 app.show_results(pd.DataFrame({'EstimatedBatteryAtArrivalPct':[11.6]}),'stale-map',app.DEFAULT_INPUTS)
b=AppTest.from_function(stale_view,default_timeout=15).run()
assert not b.exception,b.exception
assert len(b.dataframe)==0 and len(b.warning)==1 and len(b.get('iframe'))==0
print('PASS: no historical defaults, failure clearing and stale-result reserve guard.')
