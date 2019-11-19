import numpy as np
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sbn

print(os.listdir("/home/014529112/cmpe257/NY_Taxi_Fare_Prediction"))

#read train and test dataset 
TRAIN_PATH = "/home/sam2220/Desktop/ML_projects/NY_taxi_fare_prediction/kaggle_challenge_dataset/train.csv"
TEST_PATH = "/home/sam2220/Desktop/ML_projects/NY_taxi_fare_prediction/kaggle_challenge_dataset/test.csv"
with open(TRAIN_PATH) as file:
    len_train = len(file.readlines())

print(len_train)

#train_temp = pd.read_csv(TRAIN_PATH, nrows=5)
#print("training dataset head:")
#print(train_temp)
#test_temp = pd.read_csv(TEST_PATH, nrows=5)
#print("test dataset head:")
#print(test_temp)


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

def clean_df(PATH, cols, datatypes, chunksize):
    if PATH == TEST_PATH:
        df_list = []
        for chunk in pd.read_csv(PATH, usecols = cols, dtype=datatypes, chunksize=chunksize):
            chunk['pickup_datetime'] = chunk['pickup_datetime'].str.slice(0, 16)
            chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'], utc=True, 
                                                      format='%Y-%m-%d %H:%M')

            df_list.append(chunk)
            #print(len(df_list))
        return pd.concat(df_list)
    
    elif(PATH == TRAIN_PATH):
        X_df_list = []
        y_df_list = []
        #df_list = []
        for chunk in pd.read_csv(PATH, usecols = cols, dtype=datatypes, chunksize=chunksize):  
            #converting pickup datetime to proper format
            chunk['pickup_datetime'] = chunk['pickup_datetime'].str.slice(0, 16)
            chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'], utc=True, 
                                                      format='%Y-%m-%d %H:%M')
            #outlier removal
            #remove rows with passenger count 0 and more than 6
            chunk = chunk[(chunk['passenger_count'] > 0) & (chunk['passenger_count'] < 7) ]
            
            #remove rows with fare amount less than $2 and greater than $400 
            chunk = chunk[(chunk['fare_amount'] > 2.0) & (chunk['fare_amount'] < 400.0)]            
            
            #add datetime features to the data 
            #pickup_datetime -> hour, month, year, isWeekday, day
            add_datetime_features(chunk)
            
            #removing data with missing points -> (does not affect the number of sample points)
            #chunk = chunk.dropna(how = 'any', axis = 'rows')
            
            #add distance travelled by the cab to dataframe
            chunk['haversine_distnace'] = add_distance(chunk.pickup_latitude, chunk.pickup_longitude,
                        chunk.dropoff_latitude, chunk.dropoff_longitude)
            
            chunk = chunk[(chunk['haversine_distnace'] > 0.1) & (chunk['haversine_distnace'] < 100)] 
            
            #appending chunks to the dataframe list
            #df_list.append(chunk)
            y_df_list.append(chunk['fare_amount'])
            chunk.drop(labels = 'fare_amount', axis=1, inplace=True)
            X_df_list.append(chunk)
            
        #df = pd.concat(df_list)
        X_df = pd.concat(X_df_list)
        y_df = pd.concat(y_df_list)
        return  X_df, y_df

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

#insample dataset
train_target, train_label = clean_df(TRAIN_PATH, train_cols, train_types, chunksize = 500000)
#n_train = len(train_label)
print(train_target.shape)

#out of samples dataset
test_dataset = clean_df(TEST_PATH, chunk.drop(labels = 'fare_amount', axis=1, inplace=True)test_cols, test_types, chunksize = 2000)
n_test = len(test_dataset)
print(n_test)

train_target.describe()

#drop location features as the distance has already been calculated and added to the dataset
features_to_drop = ['pickup_longitude',	'pickup_latitude', 'dropoff_longitude',	'dropoff_latitude']
train_target.drop(labels = features_to_drop, axis=1, inplace=True)
print(train_target.shape)

#histogram of haversine distance 
train_target[train_target.haversine_distnace<100].haversine_distnace.hist(bins=40, figsize=(12,4))
plt.xlabel('distance miles')
plt.title('Histogram ride distances in miles')

train_df = pd.concat([train_target, train_label], axis = 1)
print(train_df.shape)

corrs = train_df.corr()
plt.figure(figsize = (12, 12))
sbn.heatmap(corrs, annot = True, vmin = -1, vmax = 1, fmt = '.3f', cmap=plt.cm.PiYG_r);


# the correlation between fare_amount and haversine_distance is 0.791. It is a significant feature for learning





