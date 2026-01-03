#!/usr/bin/env python
# coding: utf-8

# #### 1: Imports and Setup

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt  # Streamlit supports st.pyplot

st.set_page_config(page_title="Quant Research Log Analyzer", layout="wide")
st.title("🛠 Quantitative Research Log Analyzer")
st.markdown("""
Interactive dashboard for monitoring quant trading/research system logs.  
Supports file upload, exploratory filtering, metrics, and automated anomaly alerts.
""")

DATA_DIR = Path("data")


# #### 2: File Upload and Parsing

# In[2]:


@st.cache_data(show_spinner="Parsing logs...")
def load_and_parse(uploaded_file):
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = DATA_DIR / "uploaded_logs.jsonl"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = temp_path
    else:
        file_path = DATA_DIR / "research_logs.jsonl"
    
    # Simple JSONL parser (reuse logic from Phase 2)
    logs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    df = pd.DataFrame(logs)
    
    # Type conversions
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

uploaded_file = st.file_uploader("Upload logs (JSONL)", type="jsonl")
df = load_and_parse(uploaded_file)

st.success(f"Loaded {len(df):,} log entries")


# #### 3: Key Metrics Sidebar

# In[3]:


st.sidebar.header("Key Metrics")

level_counts = df['level'].value_counts()
st.sidebar.metric("Total Entries", len(df))
st.sidebar.metric("ERROR Count", level_counts.get('ERROR', 0))
st.sidebar.metric("WARNING Count", level_counts.get('WARNING', 0))

latency = df['latency_ms'].dropna()
if len(latency) > 0:
    st.sidebar.metric("Avg Latency (ms)", f"{latency.mean():.1f}")
    st.sidebar.metric("P95 Latency (ms)", f"{np.percentile(latency, 95):.1f}")
    st.sidebar.metric("P99 Latency (ms)", f"{np.percentile(latency, 99):.1f}")


# #### 4: Interactive Filters and Preview

# In[4]:


from datetime import datetime, timedelta  # ← Add timedelta here

st.header("Data Preview & Filters")

col1, col2 = st.columns(2)
with col1:
    level_filter = st.multiselect(
        "Log Level",
        options=sorted(df['level'].unique()),
        default=sorted(df['level'].unique())
    )
with col2:
    component_filter = st.multiselect(
        "Component",
        options=sorted(df['component'].unique()),
        default=sorted(df['component'].unique())
    )

# Convert Pandas Timestamp → Python datetime for Streamlit compatibility
min_ts = pd.Timestamp(df['timestamp'].min()).to_pydatetime()
max_ts = pd.Timestamp(df['timestamp'].max()).to_pydatetime()

# Time range slider
date_range = st.slider(
    "Time Range",
    min_value=min_ts,
    max_value=max_ts,
    value=(min_ts, max_ts),
    step=timedelta(minutes=15),      # Now works!
    format="YYYY-MM-DD HH:mm:ss"
)

# Apply filters
filtered_df = df[
    df['level'].isin(level_filter) &
    df['component'].isin(component_filter) &
    (df['timestamp'] >= pd.Timestamp(date_range[0])) &
    (df['timestamp'] <= pd.Timestamp(date_range[1]))
].copy()

st.dataframe(filtered_df.head(200), use_container_width=True)
st.caption(f"Showing {len(filtered_df):,} of {len(df):,} entries after filtering")


# #### 5: Overall Log Level Breakdown

# In[5]:


st.markdown("---")  # Optional: visual separator
st.subheader("Log Level Distribution in Current View")

level_dist = filtered_df['level'].value_counts()

if not level_dist.empty:
    # Define nice colors for each level (works on light/dark theme)
    colors = {
        'INFO': '#2ca02c',    # Green
        'WARNING': '#ff7f0e', # Orange
        'ERROR': '#d62728',   # Red
        'DEBUG': '#7f7f7f'    # Gray
    }
    color_list = [colors.get(level, '#9467bd') for level in level_dist.index]

    fig0, ax0 = plt.subplots(figsize=(10, 5))
    bars = ax0.bar(level_dist.index, level_dist.values, color=color_list)
    ax0.set_title("Log Entries by Level")
    ax0.set_ylabel("Count")
    ax0.set_xlabel("Log Level")

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width()/2., height + max(level_dist.values)*0.01,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig0)
else:
    st.info("ℹ️ No logs match the current filters")

st.markdown("---")  # Separator before next section


# #### 6: Visualizations

# In[6]:


st.header("Visualizations")

# Ensure 'hour' column exists for time-based charts
if 'timestamp' in filtered_df.columns and len(filtered_df) > 0:
    filtered_df['hour'] = filtered_df['timestamp'].dt.floor('H')
else:
    filtered_df['hour'] = pd.NaT

# 1. Error Count by Hour – only show if there are any ERROR logs
error_df = filtered_df[filtered_df['level'] == 'ERROR']
if len(error_df) > 0:
    hourly_errors = error_df.groupby('hour').size()
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    hourly_errors.plot(kind='bar', ax=ax1, color='red', alpha=0.8)
    ax1.set_title("ERROR Count by Hour")
    ax1.set_ylabel("Number of Errors")
    ax1.set_xlabel("Hour")
    ax1.tick_params(axis='x', rotation=45)
    # Add count labels on bars
    for i, v in enumerate(hourly_errors):
        ax1.text(i, v + max(hourly_errors)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig1)
else:
    st.info("ℹ️ No ERROR logs in current filter → Error-by-hour chart hidden")

# 2. Latency Distribution
col3, col4 = st.columns(2)

with col3:
    latency_data = filtered_df['latency_ms'].dropna()
    if len(latency_data) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bins = min(50, max(10, len(latency_data) // 20))  # Adaptive bins
        latency_data.hist(bins=bins, color='skyblue', alpha=0.8, ax=ax2)
        ax2.set_title("Latency Distribution")
        ax2.set_xlabel("Latency (ms)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
    else:
        st.warning("⚠️ No latency data in current view")

# 3. Top Components (now shows ALL logs, not just errors)
with col4:
    component_counts = filtered_df['component'].value_counts().head(10)
    if len(component_counts) > 0:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        component_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax3, startangle=90)
        ax3.set_title("Top 10 Components (All Logs)")
        ax3.set_ylabel("")  # Clean pie chart
        st.pyplot(fig3)
    else:
        st.info("ℹ️ No data for component breakdown")


# #### 7: Anomaly Alerts and Report Generation

# In[7]:


st.header("Anomaly Alerts")

# Reuse simple detection logic
rolling_z = (filtered_df['latency_ms'] - filtered_df['latency_ms'].rolling(100).mean()) / filtered_df['latency_ms'].rolling(100).std()
latency_anomalies = filtered_df[rolling_z.abs() > 3]

error_rate = len(filtered_df[filtered_df['level'] == 'ERROR']) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0

alerts = []
if len(latency_anomalies) > 5:
    alerts.append(f"🚨 **High Latency Spikes** detected ({len(latency_anomalies)} events >3σ)")
if error_rate > 10:
    alerts.append(f"⚠️ **Elevated Error Rate**: {error_rate:.1f}%")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("✅ No major anomalies detected in current view")

# Report button
if st.button("Generate Full Report"):
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "total_entries": len(df),
        "filtered_entries": len(filtered_df),
        "error_rate_pct": error_rate,
        "alerts": alerts
    }
    report_path = DATA_DIR / f"dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    st.success(f"Report saved to {report_path.name}")


# In[ ]:




