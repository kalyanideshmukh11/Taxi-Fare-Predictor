import pickle
import numpy as np
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.ensemble  import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"
train_types = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

test_types =  {'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
train_cols = list(train_types.keys())
test_cols = list(test_types.keys())

def add_datetime_features(data_chunk):
    data_chunk['hour'] = data_chunk.pickup_datetime.dt.hour
    data_chunk['day'] = data_chunk.pickup_datetime.dt.day
    data_chunk['month'] = data_chunk.pickup_datetime.dt.month
    data_chunk['weekday'] = data_chunk.pickup_datetime.dt.weekday
    data_chunk['year'] = data_chunk.pickup_datetime.dt.year
    
    return data_chunk
    
def add_distance(lat1, lon1, lat2, lon2):
    p = np.pi/180.0
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...    

def calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = pickup_lon - dropoff_lon
    
    #Compute bearing distance
    a = np.arctan2(np.sin(dlon * np.cos(dropoff_lat)),np.cos(pickup_lat) * np.sin(dropoff_lat) - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon))
    return a

def add_dist_from_jfk(data_chunk):
    JFK_coord = (40.6413, -73.7781)
    pickup_JFK = add_distance(data_chunk['pickup_latitude'], data_chunk['pickup_longitude'], JFK_coord[0], JFK_coord[1]) 
    dropoff_JFK = add_distance(JFK_coord[0], JFK_coord[1], data_chunk['dropoff_latitude'], data_chunk['dropoff_longitude'])
    data_chunk['JFK_distance'] = pd.concat([pickup_JFK, dropoff_JFK], axis=1).min(axis=1)

    return pickup_JFK, dropoff_JFK


def clean_df(PATH, cols, datatypes, chunksize):
    if PATH == TEST_PATH:
        df_list = []
        for chunk in pd.read_csv(PATH, usecols = cols, dtype=datatypes, chunksize=chunksize):
            chunk['pickup_datetime'] = chunk['pickup_datetime'].str.slice(0, 16)
            chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'], utc=True, 
                                                      format='%Y-%m-%d %H:%M')
            
            #add datetime features to the data 
            #pickup_datetime -> hour, month, year, isWeekday, day
            add_datetime_features(chunk)
            
            #add distance travelled by the cab to dataframe
            chunk['haversine_distnace'] = add_distance(chunk.pickup_latitude, chunk.pickup_longitude,
                        chunk.dropoff_latitude, chunk.dropoff_longitude)

            #add direction of the distance
            chunk['direction'] = calculate_direction(chunk.pickup_latitude, chunk.pickup_longitude,
                        chunk.dropoff_latitude, chunk.dropoff_longitude)
            
            #add a distance_from_JFK_airport to the features as the fare would be more for places closer to airport
            pickup_JFK, dropoff_JFK = add_dist_from_jfk(chunk)
            chunk['JFK_distance'] = pd.concat([pickup_JFK, dropoff_JFK], axis=1).min(axis=1)

            df_list.append(chunk)
        return pd.concat(df_list)
    
    elif(PATH == TRAIN_PATH):
        df_list = []
        for chunk in pd.read_csv(PATH, usecols = cols, dtype=datatypes, chunksize=chunksize):  
            #converting pickup datetime to proper format
            chunk['pickup_datetime'] = chunk['pickup_datetime'].str.slice(0, 16)
            chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'], utc=True, 
                                                      format='%Y-%m-%d %H:%M')
            #outlier removal
            #remove rows with passenger count 0 and more than 6
            chunk = chunk[(chunk['passenger_count'] > 0) & (chunk['passenger_count'] < 7) ]
            
            #remove rows with fare amount less than $1 and greater than $400 
            chunk = chunk[(chunk['fare_amount'] > 1.0) & (chunk['fare_amount'] < 400.0)]     

            # Remove data with invalid coordinates
            # NY Coordinates are around 40 N and 74 W
            # 1 degree is equal to around 69 miles
            # So this range should be fine for our purpose
            chunk = chunk[(chunk['pickup_latitude'] < 42.0) & (chunk['pickup_latitude'] > 39.0)]
            chunk = chunk[(chunk['dropoff_latitude'] < 42.0) & (chunk['dropoff_latitude'] > 39.0)]
            chunk = chunk[(chunk['pickup_longitude'] < -70.0) & (chunk['pickup_longitude'] > -75.0)]
            chunk = chunk[(chunk['dropoff_longitude'] < -70.0) & (chunk['dropoff_longitude'] > -75.0)]       
            
            #add datetime features to the data 
            #pickup_datetime -> hour, month, year, isWeekday, day
            add_datetime_features(chunk)
            
            #add distance travelled by the cab to dataframe
            chunk['haversine_distnace'] = add_distance(chunk.pickup_latitude, chunk.pickup_longitude,
                        chunk.dropoff_latitude, chunk.dropoff_longitude)

            #add direction of the distance
            chunk['direction'] = calculate_direction(chunk.pickup_latitude, chunk.pickup_longitude,
                        chunk.dropoff_latitude, chunk.dropoff_longitude)            
            
            chunk = chunk[(chunk['haversine_distnace'] > 0.1) & (chunk['haversine_distnace'] < 100)] 
            
            #add a distance_from_JFK_airport to the features as the fare would be more for places closer to airport
            pickup_JFK, dropoff_JFK = add_dist_from_jfk(chunk)
            chunk['JFK_distance'] = pd.concat([pickup_JFK, dropoff_JFK], axis=1).min(axis=1)

            #appending chunks to the dataframe list
            df_list.append(chunk)
            
        df = pd.concat(df_list)
        return  df

train_target = clean_df(TRAIN_PATH, train_cols, train_types, chunksize = 500000)
train_target.describe()
train_target.head()
features_to_drop = ['pickup_datetime']
train_target.drop(labels = features_to_drop, axis=1, inplace=True)
train_target.month = train_target.month.astype(dtype = 'uint8')
train_target.year = train_target.year.astype(dtype = 'uint16')
train_target.hour = train_target.hour.astype(dtype = 'uint8')
train_target.columns.values

y = train_target['fare_amount']
x = train_target.drop(columns=['fare_amount'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

Regression =RandomForestRegressor(random_state=0, warm_start=True, n_estimators=10, n_jobs=-1)
Regression.fit(X_train,y_train)
Regression.set_params(n_estimators=18)
Regression.fit(X_train, y_train)

Regression.set_params(n_estimators=25)
Regression.fit(X_train, y_train)

y_pred = Regression.predict(X_test)
print(mean_squared_error(y_test, y_pred) ** 0.5)

original_test_data = pd.read_csv(TEST_PATH)
test_dataset = clean_df(TEST_PATH, test_cols, test_types, chunksize = 2000)
features_to_drop = ['pickup_datetime']
test_dataset.drop(labels = features_to_drop, axis=1, inplace=True)

test_pred = Regression.predict(test_dataset)
test_pred = np.around(test_pred, 2)

submission = pd.DataFrame(
    {'key': original_test_data.key, 'fare_amount': test_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)