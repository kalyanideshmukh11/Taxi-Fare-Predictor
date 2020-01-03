# New York City Taxi Fare Prediction
With  the  emergence  of  taxi  companies  such  as  Uber  andLyft,  traditional  taxi  drivers  such  as  New  York  Yellow  taxidrivers 
are  concerned  of  the  unfair  competition.Those  com-panies have mobile apps, analytics and other data which give them  considerable advantage  in  the competition. Providing similar service to traditional taxi drivers would give them a fairchance to compete with Uber, Lyft and other such companies.

Predicting fares can also help passengers as they can use thisto  find  the  expected  fare  amount  and  based  on  that  they can decide when to start the ride. 
Furthermore, this visibility can also attract more customers.Our approach is to analyze the trip data of New York yellow taxi rides 
and using that data build a machine learning modelwhich  can  predict  the  expected  fare  amount.  We  have  only used  the  data such  as  pickup,  drop  off  locations,  time  of  theride, number of passengers etc.
All of that data are availableat the start of the ride.
## Dataset
The  dataset  used  in  this  project  is  approximately  5.7GB comma  separated  value  (train.csv)  file  with  55  Million  rowsand   8   
columns   available   on   Kaggle   [6].   
Out of whichseven columns represent features and one  column  represents’fareamount’ i.e. a label. The dataset consists of information about  yellow  taxi  rides  between  year  2009  to  2015.  Thefeatures are divided into 4 parts, as follows:
* Key: an object containing unique identifier for each rowin the dataset and pickup date and time.
* Passenger  count:  number  of  passengers  travelled  in  theride of datatype int64.
* Location:  float  data  of  latitude  and  longitude  of  pickupand drop-off locations
* Pickup  date-time:  a  timestamp  object  containing  pickupdate and timeThe  data  was  first  cleaned  using  data  pre-processing tech-niques followed by feature engineering, training using modelslike  Linear  Regression,  XGBoost,  Light  Gradient  BoostingMachine, Random  Forest  Regression and Keras Regression and comparison of the results obtained from every model.
## Future Scope
The  models  have  trade-off  between  accuracy,  memoryrequired to store the models for validation and training time.The accuracy  of prediction  can  be  further  improved  byconsidering other features like traffic, driving speed, etc. The distance  can  be  accurately estimated  by  using  Google  mapAPI’s distanceMatrix() method for journey details and can beadded as a feature to the dataset, as the haversine distance isnot the real distance covered by the taxi.The  histogram  of  fare  amount  is  concentrated  between  $5 and $20. 
Thus the training would be biased to give predictions around  these  values.  Better  training  can  be  performed  using’SMOTE’ which stands for Synthetic Minority OversamplingTechnique.
