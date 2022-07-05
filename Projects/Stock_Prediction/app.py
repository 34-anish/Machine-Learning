import streamlit as st
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


stocks =('AKPL','BARUN' , 'GBIME' , 'HBL','IDBL','MSLBS','SBL' , 'SLBS')
selected_stock = st.selectbox("Select ticker ", stocks)

def load_data (ticker):
    path = f'E:\Programs\Python\Machine-Learning\Stock_Prediction\sp_data\\new_folder\\{ticker}.csv '
    return pd.read_csv(path)

data_load_state = st.text("Load data...")
df = load_data(selected_stock)
data_load_state.text("Loading Data.... Done!!!")

#description of data
st.subheader('Description of Data')
st.write(df.describe())

#Visualising the data
st.subheader('Price vs time Chart')

def chart_maker(data):
  plt.style.use('fivethirtyeight')
  plt.figure(figsize=(16,8))
  plt.plot(data['date'],data['close'])
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Closing Price (Nrs.)',fontsize=18)
  plt.title('Close Price History of '+ data.company[0])
#   plt.show()
  return plt



def change_date(df):
  df.date = pd.to_datetime(df['date'])

change_date(df)



st.pyplot(chart_maker(df))



#100 days moving average
ma100 = df['close'].rolling(100).mean()
# fig = plt.figure(figsize=(12,6))
# plt.plot(ma100)
# plt.plot(df.close)
# st.pyplot(fig)


def double_chart_maker(data, another_data):
  plt.style.use('fivethirtyeight')
  plt.figure(figsize=(16,8))
  plt.plot(data['date'],data['close'])
  plt.plot(data['date'], another_data)
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Closing Price (Nrs.)',fontsize=18)
  plt.title('Close Price History of '+ data.company[0])
  plt.legend(['Stock Price' , '100 days moving average'],loc='upper right')
#   plt.show()
  return plt

def triple_chart_maker(data, another_data1,another_data2):
  plt.style.use('fivethirtyeight')
  plt.figure(figsize=(16,8))
  plt.plot(data['date'],data['close'])
  plt.plot(data['date'], another_data1)
  plt.plot(data['date'], another_data2)

  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Closing Price (Nrs.)',fontsize=18)
  plt.title('Close Price History of '+ data.company[0])
  plt.legend(['Stock Price' , '100 days moving average','200 days moving average'],loc='upper right')
#   plt.show()
  return plt

st.pyplot(double_chart_maker(df, ma100))

#200 days average
ma200 = df['close'].rolling(200).mean()
st.pyplot(triple_chart_maker(df,ma100 ,ma200))

#Scaling the data
def filtering(df, parameter):
  return df.filter([parameter])

def filtering_multiple(df, parametera, parameterb,parameterc):
  return df.filter([parametera , parameterb,parameterc])

from sklearn.preprocessing import MinMaxScaler

def scaling(df):
  global scaler
  scaler = MinMaxScaler()
  df_to_scale= filtering (df,'close')
  df_scaled = scaler.fit_transform(np.array(df_to_scale))
  return pd.DataFrame(df_scaled)

df_scaled = scaling(df)


#Datasets creation
import math
def separating_data(df):
  data_size = len (df)
  # return data_size
  training_size = math.floor(data_size * 0.8)
  test_size = data_size - training_size

  train_data = df[0:training_size ]
  test_data = df[training_size : ]
  return np.array(train_data), np.array(test_data)

def create_dataset(dataset , time_stamp = 1):
  dataX , dataY = [] , []
  for i in range(len(dataset) - time_stamp):
    a = dataset[i:(time_stamp+i) , 0]
    dataX.append(a)
    dataY.append(dataset[i+time_stamp , 0])
  return np.array(dataX) , np.array(dataY)

time_step=100
train_data, test_data = separating_data(df_scaled)

X_train , y_train = create_dataset(train_data,time_step)
X_test , y_test = create_dataset(test_data,time_step)
st.write(X_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
st.write(X_train.shape , X_test.shape)

#Model Deployment

def filtering_multiple(df, parametera, parameterb):
  return df.filter([parametera , parameterb])





def magnified_chart_maker(data, symbol):
  plt.style.use('fivethirtyeight')
  plt.figure(figsize=(16,8))
  plt.plot(data['date'],data[['close','Predictions']])
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Closing Price (Nrs.)',fontsize=18)
  plt.title('Close Price History of '+ str(symbol))
  return plt

def output_prediction(test_data,model,symbol):
  x_input=test_data[len(test_data)-100:].reshape(1,-1)
  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()
  # demonstrate prediction for next 30 days
  from numpy import array

  lst_output=[]
  n_steps=100
  i=0
  while(i<30):
      
      if(len(temp_input)>100):
          
          x_input=np.array(temp_input[1:])
          # print("{} day input {}".format(i,x_input))
          x_input=x_input.reshape(1,-1)
          x_input = x_input.reshape((1, n_steps, 1))
          yhat = model.predict(x_input, verbose=0)
          # print("{} day output {}".format(i,yhat))
          temp_input.extend(yhat[0].tolist())
          temp_input=temp_input[1:]
          lst_output.extend(yhat.tolist())
          i=i+1
      else:
          x_input = x_input.reshape((1, n_steps,1))
          yhat = model.predict(x_input, verbose=0)
          # print(yhat[0])
          temp_input.extend(yhat[0].tolist())
          # print(len(temp_input))
          lst_output.extend(yhat.tolist())
          i=i+1
    

  predicted = scaler.inverse_transform(lst_output)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM
# model = Sequential()
def model_deploy(X_train, y_train, X_test, y_test,test_data,symbol):

  model=Sequential()
  model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
  model.add(LSTM(50,return_sequences=True))
  model.add(LSTM(50))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
  model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64,verbose=1)
  output_prediction(test_data,model,symbol) 
  # print(f"Model for {count} times")
  
  #Inverting to the original form
  train_predict=model.predict(X_train)
  print(train_predict.shape,X_train.shape)
  pd.DataFrame(train_predict)
  test_predict=model.predict(X_test)
  print(test_predict.shape,X_test.shape)

  train_predict=scaler.inverse_transform(train_predict)
  train_predict.shape
  test_predict=scaler.inverse_transform(test_predict)
  test_predict.shape
  model.save('keras_model.h5')
  return train_predict, test_predict


train_predict , test_predict = model_deploy(X_train, y_train, X_test, y_test, test_data,df.company[0])
#Plotting the data 
data = filtering_multiple (df,'close','date')
train=data[:-X_test.shape[0]]

valid=data[-X_test.shape[0]:]
valid['Predictions'] = test_predict
i =0
st.write(train,valid)


def chart_for_prediction(data,valid,symbol):
  plt.figure(figsize=(16,8))
  plt.title('Model')

  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Close Price($)',fontsize=18)
  plt.title('Close Price Train and Prediction of '+ str(symbol))
  pd.DataFrame(train['date'])
  plt.plot(train['date'], train['close'])
  plt.plot(valid['date'], valid[['close','Predictions']])
  plt.legend(['Train','Val','Predictions'],loc='upper right')
  return plt  


st.pyplot(chart_for_prediction(data,valid,df.company[0]))
st.pyplot(magnified_chart_maker(valid,i))
