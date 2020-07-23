# Read the data from .csv
# Output: dataframe of 10 rows every 1 second

import pandas as pd
import time
import sys
import json
from utils import *

df = pd.read_csv(datapath)
step = 200
df_len = df.shape[0]

for i in range(0,df_len,step):
    columns = ['id','vendor_id','pickup_datetime','dropoff_datetime','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','trip_duration']
    df_o = pd.DataFrame(columns)
    df_o = df[i: i+step]
    dict = df_o.to_dict()
    # convert a dict to JSON
    jsonString = json.dumps(dict)
    sys.stdout.write(jsonString + '\n')
    sys.stdout.flush()
    time.sleep(0.01)


