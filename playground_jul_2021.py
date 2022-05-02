#read csv file
import pandas as pd
df_data = pd.read_csv('C:/kaggle/tabular-playground-series-jul-2021/train.csv')
df_test = pd.read_csv('C:/kaggle/tabular-playground-series-jul-2021/test.csv')
#df_data.head()
#df_test.head()

#data split
from sklearn.model_selection import train_test_split
X=df_data.iloc[:,1:-3]
y1=df_data.iloc[:,-3]
y2=df_data.iloc[:,-2]
y3=df_data.iloc[:,-1]
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=2247)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=2247)
X3_train, X3_test, y3_train, y3_test = train_test_split(X, y3, test_size=2247)

#creating xgboost model
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
xg_reg = XGBRegressor()
model = MultiOutputRegressor(estimator=xg_reg).fit(X_train, y_train)
y_pred = model.predict(X_test)

#write submission file
submission = pd.read_csv('C:/kaggle/tabular-playground-series-jul-2021/sample_submission.csv')
submission['target_carbon_monoxide'] = y_pred[:,0]
submission['target_benzene'] = y_pred[:,1]
submission['target_nitrogen_oxides'] = y_pred[:,2]
submission.to_csv('C:/kaggle/tabular-playground-series-jul-2021/submission_20220502.csv', index=False)
