import streamlit as st
from datetime import date
import time
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import itertools
from bs4 import BeautifulSoup
import requests
import pandas as pd

# Set start date and today's date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit app title with styling
st.title('Stock Forecast App')
st.subheader('Top Stocks to Watch')

# List of items for welcome rolling text
welcome_items = ['Suzlon', 'Tesla', 'Amazon', 'Microsoft', 'Google', 'Apple', 'Facebook']

# Placeholder for welcome rolling text
welcome_placeholder = st.empty()

# Rolling text display for welcome with styling
welcome_text_html = """
<style>
  .welcome-text {{
    white-space: nowrap;
    overflow: hidden;
    border-right: 2px solid #2F4F4F;
    font-size: 24px;
    font-family: 'Courier New', monospace;
    animation: typing {duration}s steps(40) infinite;
    margin-top: 20px;
    color: #008080;
  }}

  @keyframes typing {{
    from {{
      width: 0;
    }}
    to {{
      width: 100%;
    }}
  }}
</style>
<div class="welcome-text">
  <p>{}</p>
</div>
"""

# Display each item one after the other
for item in welcome_items:
    welcome_placeholder.markdown(welcome_text_html.format(item, duration=len(item)*0.1), unsafe_allow_html=True)
    time.sleep(0.5)  # Optional: Adjust the sleep duration between items

# Web scraping for top-performing stocks
url = "https://www.ndtv.com/business/marketdata/stocks-gainers/nifty_weekly"
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html')
soup1 = soup.find_all('th')
table_titles = [i.text for i in soup1]

df = pd.DataFrame(columns=table_titles)
soup2 = soup.find_all('tr')
for row in soup2[1:]:
    row_data = row.find_all('td')
    individual_row_data = [data.text.strip() for data in row_data]

    length = len(df)
    df.loc[length] = individual_row_data

# Display top-performing stocks table with styling
st.subheader("Top Performing Stocks of the Week")
st.dataframe(df.style.set_properties(**{'background-color': '#008080', 'color': 'white'}))

# Stocks for prediction
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME',)
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Indices for real-time stats
indices = ['^NSEI', '^NSEBANK', '^BSESN']
selected_index = st.selectbox('Select index for real-time stats', indices)

# Years of prediction slider
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Placeholder for loading data state
data_load_state = st.empty()

# Function to load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to load real-time index data
@st.cache_data
def load_index_data(index_ticker):
    index_data = yf.download(index_ticker, START, TODAY)
    index_data.reset_index(inplace=True)
    return index_data

# Loading data state
data_load_state.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Real-time stats for selected index
index_data = load_index_data(selected_index)

# Display real-time stats for selected index
st.subheader('Real-Time Stats for Selected Index')
st.write(index_data.tail())

# Display raw data for selected stock
st.subheader('Raw Data for Selected Stock')
st.write(data.tail())

# Function to plot raw data for selected stock
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Plot raw data for selected stock
plot_raw_data()

# Real-time charts for selected index
st.subheader('Real-Time Charts for Selected Index')
fig_index = go.Figure()
fig_index.add_trace(go.Scatter(x=index_data['Date'], y=index_data['Open'], name=f"{selected_index}_open"))
fig_index.add_trace(go.Scatter(x=index_data['Date'], y=index_data['Close'], name=f"{selected_index}_close"))
fig_index.layout.update(title_text=f'Time Series Data for {selected_index} with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig_index)

# Predict forecast with Prophet for selected stock
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display and plot forecast for selected stock
st.subheader('Forecast Data for Selected Stock')
st.write(forecast.tail())

st.subheader(f'Forecast Plot for {n_years} Years for Selected Stock')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components for Selected Stock")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Infinite rolling text
for _ in range(1000):
    for item in welcome_items:
        welcome_placeholder.markdown(welcome_text_html.format(item, duration=len(item)*0.1), unsafe_allow_html=True)
        time.sleep(1.5)  # Optional: Adjust the sleep duration between items
