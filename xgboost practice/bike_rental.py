import pandas as pd
df_bikes = pd.read_csv('C:/ML/xgboost_book/handson-gb-main/Chapter01/bike_rentals.csv')

#data preprocessing
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

#X:input, y:output
X=df_bikes.iloc[:,:-1]
y=df_bikes.iloc[:,-1]

#data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

#ignore warning
import warnings
warnings.filterwarnings('ignore')

#training model
from xgboost import XGBRegressor
xg_reg = XGBRegressor()
xg_reg.fit(X_train, y_train)
y_pred = xg_reg.predict(X_test)

#print rmse
from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

#using cross validation
from sklearn.model_selection import cross_val_score
model = XGBRegressor()
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
rmse = np.sqrt(-scores)
print('rmse: ', rmse.mean())

#searching best parameter
from sklearn.model_selection import GridSearchCV
def grid_search(params, reg=XGBRegressor()):
    grid_reg = GridSearchCV(reg, params, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1)
    grid_reg.fit(X_train, y_train)
    best_params = grid_reg.best_params_
    print('best_params: ', best_params)

    best_model = grid_reg.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred)**0.5
    print('test score: ', rmse_test)
    
params = {'max_depth':[None,2,3,4,5,6,7,8,9,10,20],'min_samples_leaf':[1,2,4,8,16,32,64]}
grid_search(params)


#searching random parameter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

def randomized_search_clf(params, runs=20, clf=XGBRegressor()):
    rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, n_jobs=-1, random_state=2)
    rand_clf.fit(X_train, y_train)
    best_model = rand_clf.best_estimator_
    best_score = rand_clf.best_score_
    print('train score: ', best_score)
    
    y_pred = best_model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print('test score: ', accuracy)
    return best_model

randomized_search_clf(
    params={
        'criterion':['entropy', 'gini'],
        'splitter':['random', 'best'],
        'min_samples_split':[2, 3, 4, 5, 6, 8, 10],
        'min_samples_leaf':[1, 0.01, 0.02, 0.03, 0.04],
        'min_impurity_decrease':[0.0, 0.0005, 0.005, 0.05, 0.10, 0.15, 0.2],
        'max_leaf_nodes':[10, 15, 20, 25, 30, 35, 40, 45, 50, None],
        'max_features':['sqrt', 0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
        'max_depth':[None, 2,4,6,8],
        'min_weight_fraction_leaf':[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.05]
    })
