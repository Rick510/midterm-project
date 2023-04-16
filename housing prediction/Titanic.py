import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\rick\\AI code\\housing prediction\\train_data_titanic.csv")
print(df.head)
print(df.info)
print(df.columns.values)

df.drop(['Name', 'Ticket'], axis = 1, inplace = True)
df.head()
df.info()
sns.pairplot(df[['Survived', 'Fare']], dropna = True)
df.groupby('Survived').mean()

df.isnull().sum()
df.drop(['Cabin'], axis = 1, inplace = True) #Cabin欄位空職處理 
df.head()

df['Age'].isnull().value_counts()
df.groupby('Sex')['Age'].median().plot(kind = 'bar')

df['Age']=df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))


                                         


#將檔案壓縮成一個檔

import joblib 
joblib.dump(lr, '檔名', )