import streamlit as st
import pandas_datareader as data
import pandas as pd
import matplotlib.pyplot as plt

start = '2010-01-01'
end = '2019-12-31'

filename = r'E:\Programs\Python\Machine-Learning\Stock_Prediction\sp_data\NABIL.csv'
df = pd.read_csv(filename)

#description of data
st.subheader('From 2010  - 2019')
st.write(df.describe())

#Visualising the data
st.subheader('Price vs time Chart')

def chart_maker(data):
  plt.style.use('fivethirtyeight')
  plt.figure(figsize=(16,8))
  plt.plot(data['date'],data['close'])
  plt.xlabel('Date',fontsize=18)
  plt.ylabel('Closing Price (Nrs.)',fontsize=18)
  plt.title('Close Price History of '+ data.symbol[0])
#   plt.show()
  return plt


def change_date(df):
  df.date = pd.to_datetime(df['date'])

change_date(df)



st.pyplot(chart_maker(df))