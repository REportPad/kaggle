#read csv file
import pandas as pd
df_data = pd.read_csv('C:/kaggle/tabular-playground-series-aug-2021/train.csv')
df_test = pd.read_csv('C:/kaggle/tabular-playground-series-aug-2021/test.csv')
df_data.head()

#data split
from sklearn.model_selection import train_test_split
X=df_data.iloc[:,1:-1]
y=df_data.iloc[:,-1]
X_train=X
y_train=y
X_test = df_test.iloc[:,1:len(df_test)-1]

#creating xgboost model
from xgboost import XGBRegressor
xg_reg = XGBRegressor(tree_method='gpu_hist', gpu_id=0, max_depth=8,n_estimators=1000, eta=0.01)
xg_reg.fit(X_train, y_train)
y_pred = xg_reg.predict(X_test)

#write submission file
submission = pd.read_csv('C:/kaggle/tabular-playground-series-aug-2021/sample_submission.csv')
submission['loss'] = y_pred
submission.to_csv('C:/kaggle/tabular-playground-series-aug-2021/submission_20220724.csv', index=False)
