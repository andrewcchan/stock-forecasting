import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from gsheetsdb import connect
import base_functions

# APP

keras = tf.keras

st.title('Stock Forecasting')
tic_option = st.selectbox(
    'Select Ticker Symbol',
    ('GOOG','AAPL'))

option = st.selectbox(
    'Select Forcasting Algorithm',
    ('Naive', 'Difference+Moving Average', 'RNN'))

st.write('You selected:', option)

st.write('## Tune Hyperparameters')

st.write('## Graph')


# LOAD DATA
# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    return rows
if tic_option == 'GOOG':
    sheet_url = st.secrets["gsheets_url_goog"]
else:
    sheet_url = st.secrets["gsheets_url_aapl"]

rows = run_query(f'SELECT * FROM "{sheet_url}"')

dates = []
prices = []
for row in rows:
  dates.append(row.Date)
  prices.append(row.Close)


split_time = 365*3-367
time_train = dates[:split_time]
x_train = prices[:split_time]
time_valid = dates[split_time:]
x_valid = prices[split_time:]


naive_forecast = prices[split_time - 1:-1]

fig = plt.figure(figsize=(10, 6))
plt.plot(time_valid[0:365], x_valid[0:365], label='Series')
plt.plot(time_valid[1:366], naive_forecast[1:366], label='Forecast')
plt.legend(fontsize=14)
plt.xlabel("Date")
plt.ylabel("Closing Price")
title = tic_option + ' 1 Year Closing Price'
plt.title(title)
plt.grid(True)

st.pyplot(fig)
