import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf





# DEFINING FUNCTIONS

def fetch_stock_data(stocks, start_date, end_date):
    data = yf.download(stocks, start=start_date, end=end_date)
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

def monthly_returns(returns):
    monthly = returns.asfreq('M').ffill()
    return monthly

def yearly_returns(returns):
    yearly = returns.asfreq('Y').ffill()
    return yearly

def calculate_mean_and_variance(returns):
    mean = returns.mean()
    variance = returns.var()
    return mean, variance

def monthly_mean_and_variance(monthly):
    m_mean = monthly.mean()
    m_variance = monthly.var()
    return m_mean, m_variance

def yearly_mean_and_variance(yearly):
    y_mean = yearly.mean()
    y_variance = yearly.var()
    return y_mean, y_variance


def calculate_volatility(returns):
    volatility = returns.std()
    return volatility

def monthly_volatility(monthly):
    m_volatility = monthly.std()
    return m_volatility

def yearly_volatility(yearly):
    y_volatility = yearly.std()
    return y_volatility

def calculate_probability_profit_loss(returns):
    mean_return = returns.mean()
    std_return = returns.std()
    prob_profit = norm.cdf(0, mean_return, std_return)
    prob_loss = 1 - prob_profit
    return prob_profit, prob_loss

def monthly_probability_profit_loss(monthly):
    m_mean_return = monthly.mean()
    m_std_return = monthly.std()
    m_prob_profit = norm.cdf(0, m_mean_return, m_std_return)
    m_prob_loss = 1 - m_prob_profit
    return m_prob_profit, m_prob_loss

def calculate_probability_profit_loss(yearly):
    y_mean_return = yearly.mean()
    y_std_return = yearly.std()
    y_prob_profit = norm.cdf(0, y_mean_return, y_std_return)
    y_prob_loss = 1 - y_prob_profit
    return y_prob_profit, y_prob_loss

def create_sequences(data, time_step):
    x, y = [], []
    for i in range(len(data) - time_step):
        x.append(data[i : (i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)




def main():
    st.title("Stock Prediction App")
    st.subheader("Understanding Stock Data")
    st.markdown("""
    Stocks represent ownership in a company. When you invest in stocks, you become a shareholder and have a stake in the company's performance.
    Returns are the percentage change in a stock's price over time. Positive returns indicate a gain, while negative returns indicate a loss.
    Gains and losses are measured based on the difference between the selling price and the buying price of a stock. Gains occur when the selling price is higher than the buying price, while losses occur when the selling price is lower.
    Moving averages are used to smooth out fluctuations in stock prices over a specific period. They help identify trends and potential entry or exit points for trading.
    """)
    
    
# SIDEBAR
    # heading
    query = st.sidebar.subheader('Query Parameters')

    # Calendar
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    
    # number of stocks
    num_stocks = st.sidebar.number_input("Number of Stocks", min_value=1, value=1, step=1)
    
    # enter stock names
    stocks = []
    for i in range(num_stocks):
        stock_name = st.sidebar.text_input(f"Stock Name {i+1}")
        stocks.append(stock_name)
        
    

# Main Page

    st.subheader("Stock Graph")
    st.markdown("""Choose an option where the graph will be plotted comparing the preferred stocks with the chosen columns
                """)
# Select the column for graph
    column = []
    if st.checkbox("Open"):
        column.append("Open")
    if st.checkbox("Close"):
        column.append("Close")
    if st.checkbox("High"):
        column.append("High")
    if st.checkbox("Low"):
        column.append("Low")

# Generate Graph
    if st.button("Generate Graph"):
        for i in column:
            data = fetch_stock_data(stocks, start_date, end_date)[i]
        st.line_chart(data)
    
   
      

# CALCULATE RETURNS
    
    st.subheader('STOCK RETURNS')
    st.markdown("""Returns are important for investors as they help assess the performance of their investments.
                Positive returns indicate profit, while negative returns indicate a loss. 
                Tracking returns over time allows investors to evaluate the success of their investment strategies and make informed decisions about buying, selling, or holding assets.
                We can calculate returns in daily basis, monthly aswell as yearly""")
    if st.button("Daily"):
        for stock in stocks:
            data = fetch_stock_data(stock, start_date, end_date)
            returns = calculate_returns(data)
            mean, variance = calculate_mean_and_variance(returns)
            prob_profit,prob_loss = calculate_probability_profit_loss(returns)
            
            st.subheader(f"Returns for {stock}")
            st.write(returns)
            st.write("Mean Returns: ", round(mean.mean(),5))
            st.write("Variance Returns: ", round(variance.mean(),5))
            st.write("Probability of Profit: ",round(prob_profit.mean(),5))
            st.write("Probability of Loss: ",round(prob_loss.mean(),5))
        
            data = fetch_stock_data(stocks, start_date, end_date)['Close']
            returns = calculate_returns(data)
            volatility = calculate_volatility(returns)
            most_volatile_stock = volatility.idxmax()
            max_volatility = volatility[most_volatile_stock]
            least_volatile_stock = volatility.idxmin()
            min_volatility = volatility[least_volatile_stock]
            st.write(f"Most Volatile Stock: {most_volatile_stock}")
            st.write(f"Volatility: {max_volatility}")
            st.write(f"Least Volatile Stock: {least_volatile_stock}")
            st.write(f"Volatility: {min_volatility}")
   
            
 
    if st.button("Monthly"):
        for stock in stocks:
            data = fetch_stock_data(stock, start_date, end_date)
            returns = calculate_returns(data)
            monthly = monthly_returns(returns)
            m_mean, m_variance = monthly_mean_and_variance(monthly)
            m_prob_profit,m_prob_loss = monthly_probability_profit_loss(monthly)
            
            st.subheader(f"Returns for {stock}")
            st.write(monthly)
            st.write("Mean Returns:", round(m_mean.mean(),5))
            st.write("Variance Returns:", round(m_variance.mean(),5))
            st.write("Probability of Profit: ",round(m_prob_profit.mean(),5))
            st.write("Probability of Loss: ",round(m_prob_loss.mean(),5))
            
            data = fetch_stock_data(stocks, start_date, end_date)['Close']
            monthly = monthly_returns(data)
            m_volatility = monthly_volatility(monthly)
            m_most_volatile_stock = m_volatility.idxmax()
            m_max_volatility = m_volatility[m_most_volatile_stock]
            m_least_volatile_stock = m_volatility.idxmin()
            m_min_volatility = m_volatility[m_least_volatile_stock]
            st.write(f"Most Volatile Stock: {m_most_volatile_stock}")
            st.write(f"Volatility: {m_max_volatility}")
            st.write(f"Least Volatile Stock: {m_least_volatile_stock}")
            st.write(f"Volatility: {m_min_volatility}")
   
  
    if st.button("Annually"):
        for stock in stocks:
            data = fetch_stock_data(stock, start_date, end_date)
            returns = calculate_returns(data)
            yearly = yearly_returns(returns)
            y_mean, y_variance = yearly_mean_and_variance(yearly)
            y_prob_profit,y_prob_loss = monthly_probability_profit_loss(yearly)
            # ys_mean = y_mean.mean()
            
            st.subheader(f"Returns for {stock}")
            st.write(yearly)
            st.write("Mean Returns:", round(y_mean.mean(),5))
            st.write("Variance Returns:", round(y_variance.mean(),5))
            st.write("Probability of Profit: ",round(y_prob_profit.mean(),5))
            st.write("Probability of Loss: ",round(y_prob_loss.mean(),5))
            
            data = fetch_stock_data(stocks, start_date, end_date)['Close']
            yearly = yearly_returns(data)
            y_volatility = yearly_volatility(yearly)
            y_most_volatile_stock = y_volatility.idxmax()
            y_max_volatility = y_volatility[y_most_volatile_stock]
            y_least_volatile_stock = y_volatility.idxmin()
            y_min_volatility = y_volatility[y_least_volatile_stock]
            st.write(f"Most Volatile Stock: {y_most_volatile_stock}")
            st.write(f"Volatility: {y_max_volatility}")
            st.write(f"Least Volatile Stock: {y_least_volatile_stock}")
            st.write(f"Volatility: {y_min_volatility}")
            
    # Choose stock for prediction
    st.subheader('Prediction')
    st.markdown('Here you can select a stock from the stocks you have already listed. It will predict the High price, Low price and Closing price with the help of opening price.')
    selected_stock = st.selectbox("Select Stock for Prediction", stocks)
    
    # Generate Graph for selected stock
    if selected_stock:
        data = fetch_stock_data(selected_stock, start_date, end_date)['Open']
        if not data.empty:
            st.subheader(f"Opening Price for {selected_stock}")
            st.line_chart(data)
        else:
            st.warning("No data available for the selected stock. Please check the stock name or the date range.")
    else:
        st.warning("Please enter at least one stock name to generate the graph.")
     
     
    if selected_stock:
        data = fetch_stock_data(selected_stock, start_date, end_date)['Close']
        st.subheader('Close Price vs Moving Average 50 days')
        ma_50_days = data.rolling(50).mean()
        dic = {'Close price':data,'MA50':ma_50_days}
        df= pd.DataFrame(dic)
        fig50 = plt.figure(figsize=(8,6))
        st.line_chart(df)
    



# MACHINE LEARNING
    if selected_stock:
        datas = fetch_stock_data(selected_stock, start_date, end_date)
        datas.reset_index(inplace=True)
        
        open=datas.iloc[:,1:-5].values
        high=datas.iloc[:,-5].values
        low=datas.iloc[:,-4].values
        close=datas.iloc[:,-3].values
    
    # train test data
        open_train,open_test,close_train,close_test=train_test_split(open,close,test_size=0.30,random_state=42)
        open_train,open_test,high_train,high_test=train_test_split(open,high,test_size=0.30,random_state=42)
        open_train,open_test,low_train,low_test=train_test_split(open,low,test_size=0.30,random_state=42)

    # model creation
        close_model =LinearRegression()
        high_model =LinearRegression()
        low_model =LinearRegression()
        close_model.fit(open_train,close_train)
        low_model.fit(open_train,low_train)
        high_model.fit(open_train,high_train)
        highpred=high_model.predict(open_test)
        lowpred=low_model.predict(open_test)
        closepred=close_model.predict(open_test)   
        
        open_price = st.number_input("Enter Open Price", value=0.00, step=0.001, format="%.6f")

    # Predict close price based on the entered open price
        predicted_close = close_model.predict([[open_price]])
        st.write(f"The predicted close price is: ",round(predicted_close[0],6))
        predicted_high = high_model.predict([[open_price]])
        st.write(f"The predicted high price is: ",round(predicted_high[0],6))
        predicted_low = low_model.predict([[open_price]])
        st.write(f"The predicted low price is: ",round(predicted_low[0],6))
     
        if predicted_close>open_price:
            st.write('Buying the stock with the open price and selling it with the close price is recommended. Profit : ', predicted_close-open_price)
        else:
            st.write('High risk is seen in buying the stock with the open price and selling it with the close price. Loss: ',predicted_close-open_price)
# DEEP LEARNING
    if selected_stock:
        datass = fetch_stock_data(selected_stock, start_date, end_date) 
        datass.reset_index(inplace=True)
        z = datass.iloc[:,4:-2].values
        
    # Standardization
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(z)
    
        train_size = int(len(scaled) * 0.80)                                              
        test_size = len(scaled)-train_size                                              
        train_data, test_data = scaled[0:train_size,:], scaled[train_size-100:len(scaled),:]
        
        time_step = 100

        x_train, y_train = create_sequences(train_data, time_step)
        x_test, y_test = create_sequences(test_data, time_step)
        
    # sample,time_steps,features
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
    # model creation
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1),activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.LSTM(units=50,return_sequences = True,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.LSTM(units=25,activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Dense(units=1))
        
        model.compile(optimizer="adam", loss="mean_squared_error")
        
        model.fit(x_train, y_train, epochs=5, batch_size=32)
    
    # predicted value    
        pred = model.predict(x_test)
        pred = scaler.inverse_transform(pred)
        
    # plot graph    
        train = datass[:train_size].copy()
        test = datass[train_size:].copy()
        test.insert(loc=7,
                    column = 'pred',
                    value=pred)

        plt.plot(train.index,train['Close'])
        plt.plot(test.index,test[['Close','pred']])
        plt.legend(['Train','Test','Predictions'],loc='upper left')
        st.pyplot(plt)
        st.warning(' You may see slightly different numerical results due to floating-point round-off errors from different computation orders.')
        

if __name__ == "__main__":
    main()
