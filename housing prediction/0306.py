# import package
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

#import data
df = pd.read_csv("C:\\Users\\rick\\AI code\\housing prediction\\Housing_Dataset_Sample.csv")

#observing dataset
print(df.head)
# 蠻好用的 學起來
df.describe().T
sns.distplot(df['Price'])
sns.jointplot(x = df['Avg. Area Income'], y = df['Price'])


#preparetotrainmodel
# #X是所有可能的影響變因
# #取得所有的列的0,1,2,3,4欄位

dx = df.iloc[:, :5]
print(dx.head)
dy = df['Price']
print(dy.head)

#分訓練及測試
from sklearn.model_selection import train_test_split
dx_train,dx_test,dy_train,dy_test=train_test_split(dx,dy,test_size=0.3,random_state=42)
#using linear regression model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(dx_train, dy_train) #這裡模型就訓練好了
predictions = reg.predict(dx_test) #利用訓練好的模型來愈測測試資料
print(predictions)
print(predictions.shape)  #看陣列長度
from sklearn.metrics import r2_score

r2_score(predictions, dy_test)

#plt.scatter(predictions, dy_test, color = 'blue')
