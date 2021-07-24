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

st.write('## Select Ticker Symbol')

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

sheet_url = st.secrets["gsheets_url_goog"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')

dates = []
prices = []
for row in rows:
  dates.append(row.Date)
  prices.append(row.Close)

base_functions.plot_series(dates, prices, 'GOOG')
