import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# URL to fetch S&P 500 stock list
url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
s_and_p_500 = pd.read_csv(url)

# Function to fetch and calculate required metrics
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        # Calculate 12-month price change (Momentum)
        price_change = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100
        
        # Calculate standard deviation of daily returns (Volatility)
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Get other fundamental data (Value and Quality)
        pe_ratio = stock.info.get('trailingPE', None)
        pb_ratio = stock.info.get('priceToBook', None)
        ps_ratio = stock.info.get('priceToSalesTrailing12Months', None)
        roe = stock.info.get('returnOnEquity', None)
        roa = stock.info.get('returnOnAssets', None)
        debt_to_equity = stock.info.get('debtToEquity', None)
        
        return price_change, volatility, pe_ratio, pb_ratio, ps_ratio, roe, roa, debt_to_equity
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None, None, None, None, None, None

# Fetch data for all S&P 500 stocks and calculate scores
def fetch_and_calculate():
    tickers = s_and_p_500['Symbol'].tolist()
    
    metrics = {'Stock': [], 'Price_Change': [], 'Volatility': [], 'PE_Ratio': [], 'PB_Ratio': [], 'PS_Ratio': [], 'ROE': [], 'ROA': [], 'Debt_to_Equity': []}
    
    for ticker in tickers:
        price_change, volatility, pe_ratio, pb_ratio, ps_ratio, roe, roa, debt_to_equity = get_stock_data(ticker)
        if all(v is not None for v in [price_change, volatility, pe_ratio, pb_ratio, ps_ratio, roe, roa, debt_to_equity]):
            metrics['Stock'].append(ticker)
            metrics['Price_Change'].append(price_change)
            metrics['Volatility'].append(volatility)
            metrics['PE_Ratio'].append(pe_ratio)
            metrics['PB_Ratio'].append(pb_ratio)
            metrics['PS_Ratio'].append(ps_ratio)
            metrics['ROE'].append(roe)
            metrics['ROA'].append(roa)
            metrics['Debt_to_Equity'].append(debt_to_equity)
    
    df = pd.DataFrame(metrics)
    
    for metric in ['Price_Change', 'Volatility', 'PE_Ratio', 'PB_Ratio', 'PS_Ratio', 'ROE', 'ROA', 'Debt_to_Equity']:
        df[f'{metric}_Norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    df['Value_Score'] = df[['PE_Ratio_Norm', 'PB_Ratio_Norm', 'PS_Ratio_Norm']].mean(axis=1)
    df['Quality_Score'] = df[['ROE_Norm', 'ROA_Norm', 'Debt_to_Equity_Norm']].mean(axis=1)
    df['Momentum_Score'] = df['Price_Change_Norm']
    df['Volatility_Score'] = df['Volatility_Norm']
    
    df['Composite_Score'] = (0.3 * df['Value_Score'] + 0.3 * df['Quality_Score'] + 0.3 * df['Momentum_Score'] + 0.1 * df['Volatility_Score'])
    
    df['Rank'] = df['Composite_Score'].rank(ascending=False)
    
    return df.sort_values(by='Rank')

# Streamlit app
st.title('S&P 500 Stock Ranking')
st.write("This app ranks S&P 500 stocks based on their value, quality, momentum, and volatility metrics. Data is updated daily.")

# Fetch and calculate the stock rankings
df_sorted = fetch_and_calculate()

# Cache the data for 24 hours to prevent multiple fetches within a day
@st.cache_data(ttl=86400)
def get_cached_data():
    return fetch_and_calculate()

df_sorted = get_cached_data()

# Display the DataFrame
st.dataframe(df_sorted)

# Add some interactivity: Select a stock to see detailed information
selected_stock = st.selectbox('Select a stock to see details:', df_sorted['Stock'])

if selected_stock:
    stock = yf.Ticker(selected_stock)
    info = stock.info
    st.write(f"**{info['longName']} ({selected_stock})**")
    st.write(f"**Sector:** {info['sector']}")
    st.write(f"**Industry:** {info['industry']}")
    st.write(f"**Market Cap:** {info['marketCap']}")
    st.write(f"**Trailing P/E:** {info['trailingPE']}")
    st.write(f"**Forward P/E:** {info['forwardPE']}")
    st.write(f"**Return on Equity:** {info['returnOnEquity'] * 100:.2f}%")
    st.write(f"**12-month Price Change:** {df_sorted.loc[df_sorted['Stock'] == selected_stock, 'Price_Change'].values[0]:.2f}%")
    st.write(f"**Volatility:** {df_sorted.loc[df_sorted['Stock'] == selected_stock, 'Volatility'].values[0]:.2f}%")
    st.write(f"**Composite Score:** {df_sorted.loc[df_sorted['Stock'] == selected_stock, 'Composite_Score'].values[0]:.2f}")
