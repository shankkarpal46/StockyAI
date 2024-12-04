import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

#import data
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Trend Prediction')

stocks = ('HINDUNILVR.NS','TATAELXSI.NS','HINDALCO.NS','JSWSTEEL.NS','ONGC.NS','UPL.NS','EICHERMOT.NS','TATASTEEL.NS','TECHM.NS','SBIN.NS','NTPC.NS','INDUSINDBK.NS','DIVISLAB','DRREDDY.NS','POWERGRID.NS','HCLTECH','LT.NS','TCS.NS','HDFCBANK.NS','HDFC.NS','TATACONSUM','BPCL.NS','TITAN.NS','BAJAJFINSV.NS','SUNPHARMA.NS','WIPRO.NS','GRASIM.NS','BAJAJAUTO.NS','BAJFINANCE.NS','MARUTI.NS','ADANIPORTS.NS','KOTAKBANK.NS','M&M.NS','COALINDIA.NS','INFY.NS','CIPLA.NS','TATAMOTORS.NS','ITC.NS','AXISBANK.NS','ADANIENT.NS','BRITANNIA.NS','APOLLOHOSP.NS','HEROMOTOCO.NS','RELIANCE.NS','SBILIFE.NS','ULTRACEMCO.NS','ASIANPAINT.NS','BHARTIARTL.NS','NESTLEIND.NS','ICICIBANK.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
option = st.selectbox('period',('day', 'month', 'year'))
st.write('You selected:', option)
period = 0

if(option=='day'):
	time = st.slider('select number of days prediction you want:', 1, 10)
	period = time
elif(option=='month'):	
	time = st.slider('select number of months prediction you want:', 1, 10)
	period=time*30
else:
	time = st.slider('select number of years prediction you want:', 1, 10)
	period=time*365


#Loading Data
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

current_value=data.Close[len(data.Close)-1]

#Plotting the raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


#Forecasting Data
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
fore=forecast 
forecast=forecast[['ds', 'yhat']]
forecast=forecast.rename(columns={"ds": "Date", "yhat": "Closing"})


#Complete Predicted Data 
st.subheader('Complete Forecast data')

predicted_value=round(fore.trend[len(fore.trend)-1],2)
predicted_date=fore.ds[len(fore.ds)-1]
value_change=predicted_value-current_value
per_change=(value_change/current_value)*100
st.write("The predicted value as of ",predicted_date," is Rs.",predicted_value)
if per_change > 0:
	st.write("Percentage change ",round(per_change,2), " % ")
else:
	d=round(per_change,2)
	d=str(d)
	new_title = "<p style='color:red;'>Percentage change " + d + " %</p>"
	st.markdown(new_title, unsafe_allow_html=True)

st.write(fore)
st.download_button(label='Download CSV',data=fore.to_csv(),mime='text/csv')


#Required Predicted Data
st.subheader('Required Forecast data')

st.write(forecast)
st.download_button(label='Download CSV',data=forecast.to_csv(),mime='text/csv')
    
st.write(f'Forecast plotted')
fig1 = plot_plotly(m, fore)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(fore)
st.write(fig2)

#cross validation
st.subheader('Cross Validation')
df_cv = cross_validation(m, initial='1095 days', period='180 days', horizon = '365 days')
st.write(df_cv.head())

#performance metrics
st.subheader('Performance Metrics')
df_p = performance_metrics(df_cv)
st.code(df_p)


