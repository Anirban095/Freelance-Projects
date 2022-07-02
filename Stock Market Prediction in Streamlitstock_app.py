from multiprocessing.connection import wait
from turtle import onclick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


st.title('Live Stock Market Prediction')

start = st.sidebar.date_input('Start', value=pd.to_datetime('2000-01-01'))
end = st.sidebar.date_input('End', value= dt.datetime.now())

prediction_days = st.sidebar.slider('Days of prediction', 1,7, value=5)
pastDays = st.sidebar.text_input('Enter no. of days to be considered in the model (in Days)', value=60)
pastDays = int(pastDays)

stocks = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NFLX', 'TWTR', 'SPOT', 'NVDA', 'ABNB']
stocks = tuple(sorted(stocks))

st.sidebar.text('Not working now. Just for demo.')
st.sidebar.multiselect('Enter a stock', stocks)
choice_options = st.sidebar.radio('Choose an option: ', ('Popular Stocks', 'Custom Stock'))
if choice_options == 'Popular Stocks':
    user_input = st.sidebar.selectbox('Enter a stock ticker', stocks)
else:
    user_input = st.sidebar.text_input('Enter a Stock')
    user_input = user_input.strip()
data_load_state = st.text('Please enter a stock...')


#@st.cache
def load_data(stock):
    #data = pdr.DataReader(stock, 'yahoo', start, end)
    data = yf.download(stock, start, end)
    data.reset_index(inplace = True)
    return data
if user_input is not '':
    df = load_data(stock=user_input)
    data_load_state.text('Loading data...done!')

    st.subheader(f'{user_input} Raw Data')
    st.write(df.tail())

    #decribe
    st.subheader(f'{user_input} historical data since 2000')
    st.write(df.describe())

    #visualisation
    def plot_Stock_data(df):
        figure = go.Figure()
        figure.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'Stocks_close'))
        figure.add_trace(go.Scatter(x = df['Date'], y = df['Open'], name = 'Stocks_open'))
        figure.add_trace(go.Scatter(x = df['Date'], y = df['Adj Close'], name = 'Stocks_Adj_close'))
        figure.layout.update(width=1000, height=700, title_text = "Time Series Data", xaxis_rangeslider_visible = True)
        st.plotly_chart(figure)

    def plot_Stock_MA(df):
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        #st.subheader(f'{user_input} Closing Price vs time chart with 100 MA and 200 MA')
        figure = go.Figure()
        figure.add_trace(go.Scatter(x = df['Date'], y = df['Close'], name = 'Stocks_close'))
        figure.add_trace(go.Scatter(x = df['Date'], y = ma100, name = 'Stocks_MA100'))
        figure.add_trace(go.Scatter(x = df['Date'], y = ma200, name = 'Stocks_MA200'))
        figure.layout.update(width=1000, height=700, title_text = "Time Series Data with moving average indicator", xaxis_rangeslider_visible = True)
        st.plotly_chart(figure)

    st.subheader(f'{user_input} Closing Price vs time chart')
    plot_Stock_data(df)
    st.subheader(f'{user_input} Closing Price vs time chart with 100 MA and 200 MA')
    plot_Stock_MA(df)

    def dataProcessing(data):
        new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close'])
        for i in range(0,len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Close'][i] = data['Close'][i]

        #setting index
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        #creating train and test sets
        dataset = new_data.values

        train = dataset[0:train_range,:]
        valid = dataset[train_range:,:]
        return dataset, train, valid

    def scaleData(scaled_data, train):
        x_train, y_train = [], []
        for i in range(pastDays,len(train)):
            x_train.append(scaled_data[i-pastDays:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
        return x_train, y_train

    def CreateModel(input_shape):
        model = Sequential()
        # model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
        # model.add(Dropout(0.2))

        # model.add(LSTM(units=60, activation='relu', return_sequences=True))
        # model.add(Dropout(0.3))

        # model.add(LSTM(units=80, activation='relu', return_sequences=True))
        # model.add(Dropout(0.4))

        # model.add(LSTM(units=120, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(LSTM(units=120, activation='tanh', return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=150, activation='tanh'))

        model.add(Dense(units = 1))

        return model

    def fit_model(model, x_train, y_train):
        model.compile(loss='mse', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

    def predict(model, dataset, valid):
        #predicting next values, using past days from the train data
        inputs = dataset[len(dataset) - len(valid) - pastDays:]
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)

        X_test = []
        for i in range(pastDays,inputs.shape[0]):
            X_test.append(inputs[i-pastDays:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        closing_price_scaled = model.predict(X_test)
        return closing_price_scaled

    def PlotPredictedData():
        st.subheader(f'{user_input} Prediction vs Original')
        figure = go.Figure()
        figure.add_trace(go.Scatter(x = compare['Date'], y = compare['Close'], name = 'Actual'))
        figure.add_trace(go.Scatter(x = compare['Date'], y = compare['Prediction'], name = 'Predicted'))
        figure.layout.update(width=1000, height=700, title_text = "Time Series Data with moving average indicator", xaxis_rangeslider_visible = True)
        st.plotly_chart(figure)

    train_range = len(df) - 500

    if st.sidebar.button('Train Model and Predict'):
        train_text = st.sidebar.text('Model is training...Please wait.')
        dataset, train, valid = dataProcessing(df)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        X_train, y_train = scaleData(scaled_data, train)
        lstm_model = CreateModel(input_shape=(X_train.shape[1], 1))
        fit_model(lstm_model, X_train, y_train)
        train_text.text('Model is trained.')
        y_preds = predict(lstm_model, dataset, valid)
        y_preds = scaler.inverse_transform(y_preds)
        compare = df[train_range:][['Date', 'Close']]
        compare['Prediction'] = y_preds.reshape(len(y_preds))
        st.write(compare.tail(prediction_days))
        PlotPredictedData()


#prediction



def fb():
    #Forecasting using fbProphet
    # st.header(f'{user_input} Stock prediction using fbProphet')
    # df_train = df[['Date', 'Close']]
    # df_train = df_train.rename(columns = {'Date': 'ds', 'Close': 'y'})
    # model = Prophet()
    # model.fit(df_train)

    # future_dates = model.make_future_dataframe(periods = n_days)
    # f_cast = model.predict(future_dates)

    # st.subheader('Forecasted Data')
    # st.write(f_cast.tail())

    # st.write('Forecasted Data graph')
    # fig = plot_plotly(model, f_cast)
    # st.plotly_chart(fig)

    # #train test split
    # train = pd.DataFrame(df.Close[:int(len(df) * 0.70)])
    # test = pd.DataFrame(df.Close[int(len(df) * 0.70) : ])

    # scaler = MinMaxScaler(feature_range = (0,1))
    # scaled_train = scaler.fit_transform(train)
    # x_train, y_train = [], []

    # for i in range(100, scaled_train.shape[0]):
    #     x_train.append(scaled_train[i-100:i])
    #     y_train.append(scaled_train[i, 0])
    # x_train, y_train = np.array(x_train), np.array(y_train)

    #load my model
    # model_tf = load_model('keras_model.h5')
    return 1
