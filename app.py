import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")

st.title("Trader Performance vs Market Sentiment Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_data.csv")
    df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce')
    df['Size USD'] = pd.to_numeric(df['Size USD'], errors='coerce')
    df = df.dropna(subset=['Closed PnL'])
    df['win'] = (df['Closed PnL'] > 0).astype(int)
    return df

merged = load_data()

section = st.sidebar.radio(
    "Navigation",
    ["Part A - Data Overview",
     "Part B - Sentiment Analysis",
     "Part C - Strategy Rules",
     "Bonus - Predictive Model",
     "Bonus - Trader Clustering"]
)

# ---------------- PART A ----------------
if section == "Part A - Data Overview":
    st.subheader("Dataset Overview")
    st.write("Shape:", merged.shape)
    st.write("Missing Values:")
    st.write(merged.isnull().sum())
    st.write("Duplicate Rows:", merged.duplicated().sum())
    st.dataframe(merged.head())

# ---------------- PART B ----------------
elif section == "Part B - Sentiment Analysis":
    st.subheader("PnL Distribution by Sentiment")

    fig, ax = plt.subplots()
    sns.boxplot(data=merged, x='classification', y='Closed PnL', ax=ax)
    st.pyplot(fig)

    st.subheader("Win Rate by Sentiment")
    win_rate = merged.groupby('classification')['win'].mean().reset_index()
    st.dataframe(win_rate)

    st.subheader("Average Trade Size by Sentiment")
    size_analysis = merged.groupby('classification')['Size USD'].mean().reset_index()
    st.dataframe(size_analysis)

# ---------------- PART C ----------------
elif section == "Part C - Strategy Rules":
    st.subheader("Actionable Strategy Recommendations")

    st.markdown("""
    **Strategy Idea 1: Sentiment-Based Dynamic Risk Adjustment**

The analysis shows that Fear regimes produce higher volatility and amplify losses for inconsistent traders. This indicates that risk exposure should not remain constant across sentiment states.

Rule of Thumb:

During Fear days:

Reduce position size by 25–40%.

Limit trading activity for traders with win rate < 60%.

Tighten stop-loss thresholds.

During Greed days:

Maintain controlled exposure instead of increasing leverage.

This ensures capital preservation during unstable sentiment regimes.
   
**Strategy Idea 2: Performance-Based Capital Allocation**

Segment analysis shows that High PnL and Consistent traders remain more stable across sentiment regimes, while Low PnL and Inconsistent traders experience larger drawdowns.

Rule of Thumb:

Allocate higher capital weight to historically High PnL traders during volatile regimes.

Reduce exposure to Low PnL traders during Fear periods.

Allow increased trade frequency only for traders with proven consistent win rates.

This creates a sentiment-aware portfolio allocation framework rather than uniform exposure. """)

# ---------------- BONUS: Predictive Model ----------------
elif section == "Bonus - Predictive Model":
    st.subheader("Predict Daily Profitability")

    daily_pnl = (
        merged
        .groupby(['Account','date','classification'])['Closed PnL']
        .sum()
        .reset_index()
    )

    daily_pnl['profitable'] = (daily_pnl['Closed PnL'] > 0).astype(int)

    daily_trades = (
        merged
        .groupby(['Account','date'])
        .size()
        .reset_index(name='trade_count')
    )

    daily_pnl = daily_pnl.merge(daily_trades, on=['Account','date'])
    le = LabelEncoder()
    daily_pnl['sentiment_encoded'] = le.fit_transform(daily_pnl['classification'])

    X = daily_pnl[['sentiment_encoded','trade_count']]
    y = daily_pnl['profitable']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", round(accuracy,3))

# ---------------- BONUS: Clustering ----------------
elif section == "Bonus - Trader Clustering":
    st.subheader("Behavioral Archetype Clustering")

    trader_features = merged.groupby('Account').agg({
        'Closed PnL': 'sum',
        'win': 'mean',
        'Size USD': 'mean'
    }).reset_index()

    trade_counts = merged.groupby('Account').size().reset_index(name='trade_count')
    trader_features = trader_features.merge(trade_counts, on='Account')

    X = trader_features[['Closed PnL','win','Size USD','trade_count']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    trader_features['cluster'] = kmeans.fit_predict(X_scaled)

    st.dataframe(trader_features.groupby('cluster').mean(numeric_only=True))

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        trader_features['Closed PnL'],
        trader_features['win'],
        c=trader_features['cluster']
    )
    ax2.set_xlabel("Total PnL")
    ax2.set_ylabel("Win Rate")
    st.pyplot(fig2)
