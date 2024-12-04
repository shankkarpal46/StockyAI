import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Prediction')

stocks = ('HINDUNILVR.NS','TATAELXSI.NS','HINDALCO.NS','JSWSTEEL.NS','ONGC.NS','UPL.NS','EICHERMOT.NS','TATASTEEL.NS','TECHM.NS','SBIN.NS','NTPC.NS','INDUSINDBK.NS','DIVISLAB','DRREDDY.NS','POWERGRID.NS','HCLTECH','LT.NS','TCS.NS','HDFCBANK.NS','HDFC.NS','TATACONSUM','BPCL.NS','TITAN.NS','BAJAJFINSV.NS','SUNPHARMA.NS','WIPRO.NS','GRASIM.NS','BAJAJAUTO.NS','BAJFINANCE.NS','MARUTI.NS','ADANIPORTS.NS','KOTAKBANK.NS','M&M.NS','COALINDIA.NS','INFY.NS','CIPLA.NS','TATAMOTORS.NS','ITC.NS','AXISBANK.NS','ADANIENT.NS','BRITANNIA.NS','APOLLOHOSP.NS','HEROMOTOCO.NS','RELIANCE.NS','SBILIFE.NS','ULTRACEMCO.NS','ASIANPAINT.NS','BHARTIARTL.NS','NESTLEIND.NS','ICICIBANK.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
option = st.selectbox('period',('day', 'month', 'year'))
st.write('You selected:', option)
days = st.slider('select number of prediction you want:', 1, 10)
period=0
if(option=='day'):
	period = days
elif(option=='month'):
	period=days*30
else:
	period=days*365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Dataset or Historical Data')
st.write(data)

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
fore=forecast 
forecast=forecast[['ds', 'yhat']]
forecast=forecast.rename(columns={"ds": "Date", "yhat": "Closing"})

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast)

    
st.write(f'Forecast plotted')
fig1 = plot_plotly(m, fore)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(fore)
st.write(fig2)