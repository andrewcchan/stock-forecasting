import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from gsheetsdb import connect



# APP

keras = tf.keras

st.title('Stock Forecasting')

st.write('## Select Ticker Symbol')

option = st.selectbox(
    'Select Forcasting Algorithm',
    ('Naive', 'Difference+Moving Average', 'RNN'))

st.write('You selected:', option)

st.write('## Tune Hyperparameters')

st.write('## Graph')

def plot_series(time, series, tic_symbol, format="-", start=0, end=None, label=None):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(tic_symbol,'1 Year Closing Price')
    plt.grid(True)
    st.pyplot(fig)
    
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level




# LOAD DATA
# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    return rows

sheet_url = st.secrets["gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

dates = []
prices = []
for row in rows:
  dates.append(row.Date)
  prices.append(row.Close)

plot_series(dates, prices, 'GOOG')
