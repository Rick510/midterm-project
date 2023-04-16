import joblib

model_pretrained = joblib.load('剛剛建立的檔名')
import pandas as pd
#要刪掉甚麼欄位取決於當初模型訓練用那些欄位
df_test = pd.read_csv("C:\\Users\\rick\\AI code\\housing prediction\\test.csv")
df_test.drop(['Name', 'Ticket'], axis = 1, inplace = True)
df_test.drop('cabin', axis = 1, inplace = True)
df_test['Age'] = df_test.groupby('Sex')['Age'].apply()