import pandas as pd
from sklearn import linear_model

# df = pd.read_csv(r'C:/Users/lucky/OneDrive/Desktop/programs/project/flask/Deployment_flask/joblib/house_price.csv')
df = pd.read_csv('house_price.csv')
# print (df)

model =linear_model.LinearRegression()
X = df[['Area']]
y = df[['Price']]
model.fit(X,y)
print(model.predict([[5]]))

import pickle 
with open ('model_pickle','wb') as file:
    pickle.dump(model,file)

with open ('model_pickle','rb')as file:
    mp = pickle.load(file)

print(f"From pickle {mp.predict([[5]])}")

import joblib
joblib.dump(model,'model_joblib')
mj = joblib.load('model_joblib')
print(f"Price from joblib{mj.predict([[5]])}")