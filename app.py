import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yfinance as yf
import datetime
import os
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# --- 1. PAGE SETUP & STYLING ---
st.set_page_config(page_title="AI FinTech Analytics", layout="wide", page_icon="💹")

# --- CUSTOM UI STYLING (FIXED FOR LIGHT & DARK MODE) ---
# --- UNIVERSAL UI STYLING (SMART LIGHT/DARK MODE) ---
st.markdown("""
    <style>
    /* 1. Use CSS Variables for automatic Theme Switching */
    :root {
        --header-color: var(--text-color);
    }

    /* 2. Main Title Styling */
    .main-title { 
        font-size: 42px; 
        font-weight: bold; 
        color: var(--text-color); /* Automatically switches white/black */
    }

    /* 3. Metric Container - Glassmorphism look */
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,0.2);
        background-color: rgba(128,128,128,0.1); /* Subtle tint on both modes */
        padding: 20px; 
        border-radius: 12px;
    }

    /* 4. Metric Label (e.g., 'Current Price') - Force Visibility */
    [data-testid="stMetricLabel"] p {
        color: var(--text-color) !important; 
        opacity: 0.8; /* Slightly softer than the value */
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* 5. Metric Value (The numbers) */
    [data-testid="stMetricValue"] { 
        color: #00d4ff !important; /* A bright cyan that pops on both white and black */
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    /* 6. Fix for Headings */
    h1, h2, h3 {
        color: var(--text-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)
   
# --- 2. SIDEBAR (CONTROL PANEL) ---
st.sidebar.header("🕹️ Project Control Panel")
user_input = st.sidebar.text_input('Stock Ticker Symbol', 'AAPL')
start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Project Details")
st.sidebar.info("Developed by Tripti 🖤")
st.sidebar.caption("CSE Final Year Project 2026")

LOOKBACK = 100 

# --- 3. DATA INGESTION (CRITICAL: Must come before displaying data) ---
@st.cache_data
def get_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty: return None
        
        # --- FIX STARTS HERE ---
        # If columns are Multi-Index (e.g., ('Close', 'AAPL')), flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # --- FIX ENDS HERE ---
        
        # Technical Indicators
        data['MA100'] = data['Close'].rolling(100).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

df = get_data(user_input, start_date, end_date)

# Check if data exists before proceeding
if df is None or len(df) < 300:
    st.warning("Waiting for valid ticker or insufficient data (need at least 300 days).")
    st.stop()
# --- 1. LIVE MARKET INDICATORS ---
st.header("Live Market Indicators")

# Check if df has enough rows to calculate indicators
if df is not None and len(df) >= 200:
    # Calculate Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Calculate Moving Averages safely
    ma50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
    ma200 = float(df['Close'].rolling(window=200).mean().iloc[-1])

    if ma50 > ma200:
        st.success("🚀 **Golden Cross Detected:** Long-term Bullish Trend!")
    else:
        st.warning("⚠️ **Death Cross Detected:** Long-term Bearish Trend!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Price", f"${float(df['Close'].iloc[-1]):.2f}")

    with col2:
        vol = float(df['Volatility'].iloc[-1])
        st.metric("Market Volatility (20-day)", f"${vol:.2f}")

    with col3:
        st.metric("RSI (14-day)", f"{df['RSI'].iloc[-1]:.2f}" if 'RSI' in df else "N/A")
else:
    st.info("📥 Please wait... Fetching enough historical data to calculate trends.")

# --- 5. DATASET INSIGHTS ---
st.divider()
st.header(" Dataset Insights & EDA")
tab_data, tab_stats, tab_corr = st.tabs(["📊 Raw Data", "📉 Stats", "🔥 Correlation"])

with tab_data:
    st.dataframe(df.tail(100), use_container_width=True)
    csv = df.to_csv().encode('utf-8')
    st.download_button("📥 Download Dataset", data=csv, file_name=f'{user_input}_dataset.csv')

with tab_stats:
    st.write("### OHLCV Statistical Distribution")
    st.table(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

with tab_corr:
    st.write("### Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    corr_data = df[['Open', 'High', 'Low', 'Close', 'MA100', 'RSI']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# --- 6. MODEL PERFORMANCE ---
st.divider()
st.header(" AI Prediction Performance")

model_df = df[['Close']].dropna()
dataset = model_df.values
training_len = int(np.ceil(len(dataset) * .85))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Training logic
train_data = scaled_data[0:training_len, :]
x_train, y_train = [], []
for i in range(LOOKBACK, len(train_data)):
    x_train.append(train_data[i-LOOKBACK:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

@st.cache_resource
def get_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    #you have changed
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
    return model

model = get_lstm_model()

# Testing logic
test_data = scaled_data[training_len - LOOKBACK: , :]
x_test, y_test = [], dataset[training_len:, :]
for i in range(LOOKBACK, len(test_data)):
    x_test.append(test_data[i-LOOKBACK:i, 0])
x_test = np.reshape(np.array(x_test), (len(x_test), LOOKBACK, 1))

lstm_preds = scaler.inverse_transform(model.predict(x_test))

tab_l, tab_e = st.tabs(["🧠 LSTM Forecast", "📊 Error Analysis"])

with tab_l:
    st.metric("Model Confidence (R²)", f"{r2_score(y_test, lstm_preds):.4f}")
    fig_lstm = plt.figure(figsize=(12,5))
    plt.plot(y_test, 'g', label="Market Reality")
    plt.plot(lstm_preds, 'r', label="LSTM Prediction")
    plt.legend()
    st.pyplot(fig_lstm)

with tab_e:
    st.write("### Prediction Error (Residuals)")
    residuals = y_test - lstm_preds
    fig_res, ax_res = plt.subplots(figsize=(12, 3))
    ax_res.plot(residuals, color='gray', alpha=0.5)
    ax_res.axhline(0, color='red', linestyle='--')
    st.pyplot(fig_res)


# --- 6.5 MULTI-MODEL COMPARATIVE ANALYSIS ---
st.divider()
st.header("📊 Model Comparison: ML vs. Deep Learning")
st.info("Comparing our LSTM against traditional Machine Learning models to validate performance.")

# Prepare 'flattened' data for traditional ML models
x_train_ml = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test_ml = x_test.reshape(x_test.shape[0], x_test.shape[1])

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Create Tabs for Comparison
tab_lr, tab_rf, tab_comp = st.tabs(["📉 Linear Regression", "🌲 Random Forest", "🏆 Final Comparison"])

with tab_lr:
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(x_train_ml, y_train)
    lr_preds = lr_model.predict(x_test_ml)
    lr_preds_real = scaler.inverse_transform(lr_preds.reshape(-1, 1))
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Linear Regression R²", f"{r2_score(y_test, lr_preds_real):.4f}")
        st.caption("Standard statistical approach.")
    with col2:
        fig_lr = plt.figure(figsize=(10, 4))
        plt.plot(y_test, label="Actual", color='green')
        plt.plot(lr_preds_real, label="LR Prediction", color='orange', linestyle='--')
        plt.legend(); st.pyplot(fig_lr)

with tab_rf:
    # 2. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train_ml, y_train)
    rf_preds = rf_model.predict(x_test_ml)
    rf_preds_real = scaler.inverse_transform(rf_preds.reshape(-1, 1))
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Random Forest R²", f"{r2_score(y_test, rf_preds_real):.4f}")
        st.caption("Ensemble learning using decision trees.")
    with col2:
        fig_rf = plt.figure(figsize=(10, 4))
        plt.plot(y_test, label="Actual", color='green')
        plt.plot(rf_preds_real, label="RF Prediction", color='blue', linestyle='--')
        plt.legend(); st.pyplot(fig_rf)

with tab_comp:
    # 3. Summary Table for Report
    st.write("### Model Performance Summary")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# For LSTM
    lstm_mse = mean_squared_error(y_test, lstm_preds)
    lstm_mape = mean_absolute_percentage_error(y_test, lstm_preds)

# Add these to your Comparison Table dictionary:
    comparison_data = {
        "Model": ["Linear Regression", "Random Forest", "LSTM (Proposed)"],
        "R² Score": [
            f"{r2_score(y_test, lr_preds_real):.4f}",
            f"{r2_score(y_test, rf_preds_real):.4f}",
            f"{r2_score(y_test, lstm_preds):.4f}"
        ],
        "MSE": [
            f"{mean_squared_error(y_test, lr_preds_real):.2f}",
            f"{mean_squared_error(y_test, rf_preds_real):.2f}",
            f"{lstm_mse:.2f}"
        ],
        "MAPE (%)": [
            f"{mean_absolute_percentage_error(y_test, lr_preds_real)*100:.2f}%",
            f"{mean_absolute_percentage_error(y_test, rf_preds_real)*100:.2f}%",
            f"{lstm_mape*100:.2f}%"
        ]
    }
    st.table(pd.DataFrame(comparison_data))

# --- 7. FUTURE FORECAST ---
st.divider()
st.header(" AI Forecast Results")
last_val = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
tomorrow = scaler.inverse_transform(model.predict(last_val))
st.success(f"### Predicted Price for {user_input} (Next Session): **${tomorrow[0][0]:.2f}**")

#if anything goes wrong
# --- 8. MULTI-STEP LONG-TERM FORECAST ---
st.divider()
st.header(" Long-Term AI Projections")

# 1. Prepare the starting window (Last 100 days)
# We make sure this is a 2D array of the last 100 scaled points
current_batch = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, 1)

month_preds = []
year_preds = []

# Total trading days to predict (approx 252 days in a market year)
total_steps = 252 

with st.spinner('Generating Long-Term Projections...'):
    # We use a temporary variable so we don't overwrite our batch
    temp_batch = current_batch.copy()
    
    for i in range(total_steps):
        # Predict 1 step ahead
        current_pred = model.predict(temp_batch, verbose=0)
        
        # Store the prediction
        if i < 22: # Approx 1 trading month
            month_preds.append(current_pred[0])
        year_preds.append(current_pred[0])
        
        # RECURSIVE STEP: 
        # Shift the window: Remove the first day, add the prediction at the end
        new_input = np.append(temp_batch[:, 1:, :], [current_pred], axis=1)
        temp_batch = new_input

# 2. Convert to Dollar amounts ONLY if we have data
if len(month_preds) > 0:
    final_month = scaler.inverse_transform(month_preds)
    final_year = scaler.inverse_transform(year_preds)

    c_m, c_y = st.columns(2)
    with c_m:
        st.subheader("📅 Next 1-Month Trend")
        avg_month = np.mean(final_month)
        st.metric("Avg. Predicted Price", f"${avg_month:.2f}")
        
        fig_m, ax_m = plt.subplots(figsize=(10, 4))
        ax_m.plot(final_month, color='#00ffcc', linewidth=2)
        ax_m.set_ylabel("Price ($)")
        st.pyplot(fig_m)

    with c_y:
        st.subheader("📅 Next 1-Year Projection")
        avg_year = np.mean(final_year)
        st.metric("Avg. Predicted Price", f"${avg_year:.2f}")
        
        fig_y, ax_y = plt.subplots(figsize=(10, 4))
        ax_y.plot(final_year, color='#ff3366', linewidth=2)
        ax_y.set_ylabel("Price ($)")
        st.pyplot(fig_y)
    
    st.warning("⚠️ **Note:** Long-term projections use recursive logic where errors compound over time. Use these for directional trend analysis only.")
else:
    st.error("Model failed to generate long-term steps. Check data scaling.")

    # --- FINAL SECTION: MARKET CONTEXT & NEWS ---
st.divider()
st.header("📰 Latest Market Headlines")
st.markdown(f"Recent news coverage for **{user_input}** to provide qualitative context to the AI forecasts.")

try:
    ticker_data = yf.Ticker(user_input)
    news = ticker_data.news

    if news:
        # Display headlines in a clean, vertical list
        for article in news[:8]: # Showing more headlines since the score is gone
            title = article.get('title') or article.get('content', {}).get('title') or "News Update"
            link = article.get('link', 'https://finance.yahoo.com')
            publisher = article.get('publisher', 'Financial News')
            
            # Using a cleaner Markdown format
            st.markdown(f"**🔹 [{title}]({link})**")
            st.caption(f"Source: {publisher}")
            st.write("---") # Small separator between articles
    else:
        st.info("No recent headlines found for this ticker.")
        
except Exception as e:
    st.error("Market news feed temporarily unavailable.")
