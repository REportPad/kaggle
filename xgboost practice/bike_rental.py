import pandas as pd
df_bikes = pd.read_csv('C:/ML/xgboost_book/handson-gb-main/Chapter01/bike_rentals.csv')

df_bikes[df_bikes.isna().any(axis=1)]
df_bikes['atemp'] = df_bikes['atemp'].interpolate()
df_bikes['temp'] = df_bikes['temp'].interpolate()
df_bikes['hum'] = df_bikes['hum'].interpolate()
df_bikes['windspeed'] = df_bikes['windspeed'].interpolate()
df_bikes.iloc[730,'mnth'] = 12
df_bikes.loc[730,'yr'] = 1

df_bikes = df_bikes.drop('dteday', axis=1)
df_bikes = df_bikes.drop(['casual', 'registered'], axis=1)
#df_bikes.to_csv('C:/ML/xgboost_book/bike_rentals_after.csv', index=False)

X=df_bikes.iloc[:,:-1]
y=df_bikes.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
xg_reg = XGBRegressor()
xg_reg.fit(X_train, y_train)
y_pred = xg_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)
