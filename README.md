Trader Performance vs Market Sentiment Analysis

Project Overview

This project analyzes how market sentiment (Fear & Greed Index) influences trader behavior and profitability.

The objective is to:

- Understand performance differences across sentiment regimes
- Identify behavioral patterns among traders
- Segment traders into meaningful archetypes
- Propose actionable trading rules
- Build predictive and clustering models (Bonus)
- Create an interactive dashboard using Streamlit

This project combines exploratory data analysis, behavioral segmentation, risk-based reasoning, and lightweight machine learning.

Part A — Data Preparation

Steps Performed

- Loaded trader and sentiment datasets
- Cleaned column names
- Handled missing values and duplicates
- Converted timestamps (milliseconds → datetime)
- Aligned trader activity with daily sentiment classification
- Created derived metrics:
  - Daily PnL
  - Win flag
  - Win rate
  - Trade frequency
  - Average trade size

Part B — Sentiment & Behavioral Analysis

1. Performance by Sentiment

- Compared PnL distribution across:
  - Fear
  - Greed
  - Extreme Greed
  - Neutral
- Calculated mean, median, and volatility
- Visualized distributions using boxplots

2. Behavioral Changes

- Analyzed trade frequency by sentiment
- Compared average trade size across regimes
- Evaluated win rate differences

3. Trader Segmentation

Traders were segmented into:

- High vs Low PnL
- Frequent vs Infrequent
- Consistent vs Inconsistent

These segments were analyzed across sentiment regimes to identify behavioral differences.

Part C — Actionable Strategy Recommendations

Based on the findings:

Strategy Idea 1: Sentiment-Based Dynamic Risk Adjustment

The analysis shows that Fear regimes produce higher volatility and amplify losses for inconsistent traders. This indicates that risk exposure should not remain constant across sentiment states.

Rule of Thumb:

During Fear days:

Reduce position size by 25–40%.

Limit trading activity for traders with win rate < 60%.

Tighten stop-loss thresholds.

During Greed days:

Maintain controlled exposure instead of increasing leverage.

This ensures capital preservation during unstable sentiment regimes.

Strategy Idea 2: Performance-Based Capital Allocation

Segment analysis shows that High PnL and Consistent traders remain more stable across sentiment regimes, while Low PnL and Inconsistent traders experience larger drawdowns.

Rule of Thumb:

Allocate higher capital weight to historically High PnL traders during volatile regimes.

Reduce exposure to Low PnL traders during Fear periods.

Allow increased trade frequency only for traders with proven consistent win rates.

This creates a sentiment-aware portfolio allocation framework rather than uniform exposure.

Predictive Model

A Logistic Regression model was built to predict daily trader profitability using:

- Sentiment classification
- Trade frequency

The model demonstrates that behavioral and sentiment features contain predictive signal.


Trader Clustering

KMeans clustering was applied to group traders based on:

- Total PnL
- Win rate
- Average trade size
- Trade count

This identified behavioral archetypes such as:

- Consistent performers
- Aggressive traders
- Low participation traders


Interactive Dashboard

A lightweight Streamlit dashboard was developed to explore:

- Sentiment impact on performance
- Win rate by regime
- PnL distribution
- Segment performance
- Clustering results
- Predictive model accuracy

Installation and setup:

To run this project locally, follow the steps below:

First, ensure that Python (version 3.9 or above) is installed on your system. You can verify this by running:

python --version

Once Python is available, open a terminal inside the project folder.

It is recommended to create a virtual environment to keep dependencies isolated:

python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

Then install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn streamlit

Running the Notebook

If you want to explore the full analysis:

jupyter notebook

Open analysis.ipynb and run all cells from top to bottom.
This will generate the cleaned dataset and output files.

Running the Dashboard

To launch the interactive Streamlit dashboard:

python -m streamlit run app.py

The dashboard will automatically open in your browser at:

http://localhost:8501

The dashboard includes:

Data overview (Part A)

Sentiment analysis (Part B)

Strategy recommendations (Part C)

Predictive model (Bonus)

Trader clustering visualization (Bonus)

Stopping the App

To stop the dashboard, press:

CTRL + C

in the terminal.



