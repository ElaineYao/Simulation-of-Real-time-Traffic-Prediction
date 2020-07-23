import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

datapath = './dataset/train.csv'

# Make the trip_duration time discrete in minutes,
# input: the dataframe of train or test.csv
# return: dataframe with 2 columns. The first is id, the second is labeled trip_duration time
# Tip: the one-hot encoding will be done after all the data is label
def output_preprocess(input):
    df_output = pd.DataFrame(columns = ['id', 'trip_duration'])
    df_output.loc[:, 'id'] = input['id']
    df_output.loc[:, 'trip_duration'] = np.round(input['trip_duration']/60)
    df_output.loc[:, 'trip_duration'] = df_output['trip_duration'].map(output_label)
    return df_output

# Label the output.
# <5 min -> label 0; 5~10min -> label 1; 10~15min -> label 2; 15~20min -> label 3;
#  20~25min -> label 4;  25~30min -> label 5;  >30min -> label 6;
def output_label(trip_duration):
    if trip_duration<5:
        return 0
    elif (trip_duration>=5)&(trip_duration<10):
        return 1
    elif (trip_duration>=10)&(trip_duration<15):
        return 2
    elif (trip_duration>=15)&(trip_duration<20):
        return 3
    elif (trip_duration>=20)&(trip_duration<25):
        return 4
    elif (trip_duration>=25)&(trip_duration<30):
        return 5
    elif trip_duration>=30:
        return 6


# Define some distance features
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b



def get_zone_features(input_temp):
    # Define zone features
    # First，delete some outliers
    input_df = input_temp.loc[(input_temp.pickup_latitude > 40.6) & (input_temp.pickup_latitude < 40.9)]
    input_df = input_df.loc[(input_df.dropoff_latitude > 40.6) & (input_df.dropoff_latitude < 40.9)]
    input_df = input_df.loc[(input_df.dropoff_longitude > -74.05) & (input_df.dropoff_longitude < -73.7)]
    input_df = input_df.loc[(input_df.pickup_longitude > -74.05) & (input_df.pickup_longitude < -73.7)]

    # Then create the lat_long grid
    pick_lat_max = input_df.loc[:, 'pickup_latitude'].max()
    drop_lat_max = input_df.loc[:, 'dropoff_latitude'].max()
    lat_max = max(pick_lat_max, drop_lat_max)

    pick_lat_min = input_df.loc[:, 'pickup_latitude'].min()
    drop_lat_min = input_df.loc[:, 'dropoff_latitude'].min()
    lat_min = min(pick_lat_min, drop_lat_min)

    pick_long_max = input_df.loc[:, 'pickup_longitude'].max()
    drop_long_max = input_df.loc[:, 'dropoff_longitude'].max()
    long_max = max(pick_long_max, drop_long_max)

    pick_long_min = input_df.loc[:, 'pickup_longitude'].min()
    drop_long_min = input_df.loc[:, 'dropoff_longitude'].min()
    long_min = min(pick_long_min, drop_long_min)

    #print(lat_max, lat_min, long_max, long_min)
    return input_df, lat_min, long_min, lat_max, long_max

#
def get_standard_onehot_x(x_data, input_df):
    x_data.loc[:, 'id'] = input_df.id
    x_data.loc[:, 'vendor_id'] = input_df.vendor_id
    x_data.loc[:, 'distance_haversine'] = input_df.distance_haversine
    x_data.loc[:, 'distance_dummy_manhattan'] = input_df.distance_dummy_manhattan
    x_data.loc[:, 'avg_speed_h'] = input_df.avg_speed_h
    x_data.loc[:, 'avg_speed_m'] = input_df.avg_speed_m
    x_data.loc[:, 'pickup_weekday'] = input_df.pickup_weekday
    x_data.loc[:, 'pickup_hour'] = input_df.pickup_hour
    x_data.loc[:, 'pickup_minute'] = input_df.pickup_minute
    x_data.loc[:, 'pickup_lat_label'] = input_df.pickup_lat_label
    x_data.loc[:, 'pickup_long_label'] = input_df.pickup_long_label
    x_data.loc[:, 'dropoff_lat_label'] = input_df.dropoff_lat_label
    x_data.loc[:, 'dropoff_long_label'] = input_df.dropoff_long_label

    # Drop id
    x_data = x_data.drop('id', axis=1)

    # Standardize some features -- 'distance_haversine', 'distance_dummy_manhattan', 'avg_speed_h', 'avg_speed_m'
    x_data[['distance_haversine', 'distance_dummy_manhattan', 'avg_speed_h',
            'avg_speed_m']] = StandardScaler().fit_transform(x_data[['distance_havers'
                                                                     'ine', 'distance_dummy_manhattan', 'avg_speed_h',
                                                                     'avg_speed_m']])

    (lat_max, lat_min, long_max, long_min) = (40.89995575, 40.60009384, -73.70030212, -74.04998779)
    weekday_dict = p_weekday_dict()
    x_data_array_5 = [weekday_dict[x] if x in weekday_dict else x for x in x_data['pickup_weekday'].values]

    hour_dict = p_hour_dict()
    x_data_array_6 = [hour_dict[x] if x in hour_dict else x for x in x_data['pickup_hour'].values]

    minute_dict = p_minute_dict()
    x_data_array_7 = [minute_dict[x] if x in minute_dict else x for x in x_data['pickup_minute'].values]

    lat_dict = p_lat_dict(lat_min, lat_max)
    x_data_array_8 = [lat_dict[x] if x in lat_dict else x for x in x_data['pickup_lat_label'].values]

    long_dict = p_long_dict(long_min, long_max)
    x_data_array_9 = [long_dict[x] if x in long_dict else x for x in x_data['pickup_long_label'].values]

    lat_dict = p_lat_dict(lat_min, lat_max)
    x_data_array_10 = [lat_dict[x] if x in lat_dict else x for x in x_data['dropoff_lat_label'].values]

    long_dict = p_long_dict(long_min, long_max)
    x_data_array_11 = [long_dict[x] if x in long_dict else x for x in x_data['dropoff_long_label'].values]

    x_data_value = np.vstack((x_data['vendor_id'].values, x_data['distance_haversine'].values,
                              x_data['distance_dummy_manhattan'].values,
                              x_data['avg_speed_h'].values, x_data['avg_speed_m'].values))
    x_data_value = x_data_value.transpose()
    x_data_onehot = np.hstack((x_data_array_5, x_data_array_6, x_data_array_7, x_data_array_8, x_data_array_9,
                               x_data_array_10, x_data_array_11))
    # x_data_onehot = np.hstack((x_data_array_5, x_data_array_6))
    # print(x_data_array_7)

    x_data_array = np.hstack((x_data_value, x_data_onehot))

    x_row = np.size(x_data_array, 0)
    x_col = np.size(x_data_array, 1)

    # print('For x -- x_row: ' + str(x_row) + ', x_col: ' + str(x_col))

    return x_data_array

def p_weekday_dict():
    key = list(range(7))
    value = tf.keras.utils.to_categorical(key)
    dictt = dict(zip(key, value))
    return dictt

def p_hour_dict():
    key = list(range(24))
    value = tf.keras.utils.to_categorical(key)
    dictt = dict(zip(key, value))
    return dictt

def p_minute_dict():
    key = list(range(60))
    value = tf.keras.utils.to_categorical(key)
    dictt = dict(zip(key, value))
    return dictt

def p_lat_dict(lat_min, lat_max):
    size = (lat_max-lat_min)//0.01 + 10
    key = list(range(int(size)))
    value = tf.keras.utils.to_categorical(key)
    dictt = dict(zip(key, value))
    return dictt

def p_long_dict(long_min, long_max):
    size = (long_max-long_min)//0.01 + 10
    key = list(range(int(size)))
    value = tf.keras.utils.to_categorical(key)
    dictt = dict(zip(key, value))
    return dictt

def y_dict():
    key = list(range(7))
    value = tf.keras.utils.to_categorical(key)
    dictt = dict(zip(key, value))
    return dictt