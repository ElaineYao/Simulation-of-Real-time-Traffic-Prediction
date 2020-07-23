from utils import *
import pandas as pd
import sys
import json

while True:
    jsonString = sys.stdin.readline()
    dict = json.loads(jsonString)
    input_temp = pd.DataFrame.from_dict(dict)
    # Define input features
    # Distance features: 'distance_haversine', 'distance_dummy_manhattan'
    # Speed features：'avg_speed_h'，'avg_speed_m', unit:m/s
    # Time features: 'pick_up_h', 'pick_up_m', 'weekday', (0 represents Sunday)
    # Zone features: 'pickup_lat_label', 'pickup_long_label', 'dropoff_lat_label', 'dropoff_long_label'
    # Define distance features
    input_temp.loc[:, 'distance_haversine'] = haversine_array(input_temp['pickup_latitude'].values,
                                                              input_temp['pickup_longitude'].values,
                                                              input_temp['dropoff_latitude'].values,
                                                              input_temp['dropoff_longitude'].values)
    input_temp.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(input_temp['pickup_latitude'].values,
                                                                             input_temp['pickup_longitude'].values,
                                                                             input_temp['dropoff_latitude'].values,
                                                                             input_temp['dropoff_longitude'].values)
    # Define speed features
    input_temp.loc[:, 'avg_speed_h'] = 1000 * input_temp['distance_haversine'] / input_temp['trip_duration']
    input_temp.loc[:, 'avg_speed_m'] = 1000 * input_temp['distance_dummy_manhattan'] / input_temp['trip_duration']

    # Define time features
    input_temp['pickup_datetime'] = pd.to_datetime(input_temp.pickup_datetime)
    input_temp.loc[:, 'pickup_weekday'] = input_temp['pickup_datetime'].dt.weekday
    input_temp.loc[:, 'pickup_hour'] = input_temp['pickup_datetime'].dt.hour
    input_temp.loc[:, 'pickup_minute'] = input_temp['pickup_datetime'].dt.minute

    # Define zone features
    input_df, lat_min, long_min, lat_max, long_max = get_zone_features(input_temp)
    input_df.loc[:, 'pickup_lat_label'] = np.round((input_df.pickup_latitude - lat_min) / 0.01)
    input_df.loc[:, 'pickup_long_label'] = np.round((input_df.pickup_longitude - long_min) / 0.01)
    input_df.loc[:, 'dropoff_lat_label'] = np.round((input_df.dropoff_latitude - lat_min) / 0.01)
    input_df.loc[:, 'dropoff_long_label'] = np.round((input_df.dropoff_longitude - long_min) / 0.01)

    # Standardize and one-hot encode the input data
    x_data = pd.DataFrame(
        columns=['id', 'vendor_id', 'distance_haversine', 'distance_dummy_manhattan', 'avg_speed_h', 'avg_speed_m',
                 'pickup_weekday', 'pickup_hour', 'pickup_minute', 'pickup_lat_label', 'pickup_long_label',
                 'dropoff_lat_label', 'dropoff_long_label'])
    x_data_onehot = get_standard_onehot_x(x_data, input_df)

    # One-hot encode the output data
    y_data = output_preprocess(input_df)
    y_data = y_data.drop('id', axis=1)
    yyy_dict = y_dict()
    y_data_list = [yyy_dict[int(x)] if x in yyy_dict else x for x in y_data['trip_duration'].values]
    y_data_onehot = np.array(y_data_list)

    y_row = np.size(y_data_onehot, 0)
    y_col = np.size(y_data_onehot, 1)

    # print('For y -- y_row: ' + str(y_row) + ', y_col: ' + str(y_col))

    # Split the data into training set, test set and validation set
    x_train_tmp, x_test, y_train_tmp, y_test = train_test_split(x_data_onehot, y_data_onehot, test_size=0.2,
                                                                random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_tmp, y_train_tmp, test_size  = 0.2, random_state=55)

    dict = {'x_train': x_train.tolist(),
            'x_val': x_val.tolist(),
            'x_test': x_test.tolist(),
            'y_train': y_train.tolist(),
            'y_val': y_val.tolist(),
            'y_test': y_test.tolist()}

    jsonString = json.dumps(dict)

    sys.stdout.write(jsonString + "\n")
    sys.stdout.flush()

