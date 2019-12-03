import numpy as np
import pandas as pd
import datetime as dt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#Load data 
train_dataset = pd.read_csv("./train.csv",nrows= 4000000)
test_dataset = pd.read_csv("./test.csv")

# The Haversine formula lets us calculate the distance on a sphere using
# Latitude and Longitude
def haversine(lon1, lat1, lon2, lat2):
    lat1 = np.radians(lat1)
    lat2= np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat=(lat2-lat1).abs()
    dlon=(lon2-lon1).abs()
    R = 6371 #radius of Earth
    a = (np.sin(dlat/2.0))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def distance_travelled(df):
    df['Distance_Travelled'] = haversine(df.dropoff_longitude,df.dropoff_latitude,df.pickup_longitude,df.pickup_latitude)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df.pickup_datetime.dt.hour
    df['day'] = df.pickup_datetime.dt.day
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['year'] = df.pickup_datetime.dt.year

distance_travelled(train_dataset)
distance_travelled(test_dataset)

train_dataset.drop(['key','passenger_count','dropoff_longitude','dropoff_latitude','pickup_longitude','pickup_latitude','pickup_datetime'],axis=1,inplace=True)
test_dataset.drop(['passenger_count','dropoff_longitude','dropoff_latitude','pickup_longitude','pickup_latitude'],axis=1,inplace=True)

train_dataset.dropna(how = 'any', axis = 'rows', inplace=True)
train_dataset.isnull().sum()

x_predict = test_dataset.drop(['key','pickup_datetime'], axis=1)
y = train_dataset.pop('fare_amount')
x = train_dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(x_train, label=y_train)

param = {
    'max_depth':7,
    'nthread':8,
    'eval_metric': 'rmse',
    'min_child_weight': 1,
    'eta':0.2,
    'colsample_bytree': 0.8
}
model = xgb.train(param, dtrain)
pred = model.predict(xgb.DMatrix(x_predict), ntree_limit=model.best_ntree_limit)
result = pd.DataFrame({"key":test_dataset["key"], "fare_amount": pred},
                         columns = ['key', 'fare_amount'])
print(result)
result.to_csv('fare_pred.csv', index=False)