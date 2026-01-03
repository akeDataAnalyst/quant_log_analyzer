# Imports for analysis
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
CLEANED_FILE = Path("data/parsed_logs.parquet")

# Load the cleaned DataFrame from Phase 2
if not CLEANED_FILE.exists():
    raise FileNotFoundError(f"Parsed file not found at {CLEANED_FILE}. Complete Phase 2 first!")

df = pd.read_parquet(CLEANED_FILE)

print(f"Loaded DataFrame with {len(df):,} rows and {df.shape[1]} columns")
print(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
display(df.head())

def calculate_key_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive monitoring metrics for quant research logs.
    Returns a nested dictionary for easy display and reporting.
    """
    metrics = {}
    
    # Basic counts
    metrics['total_entries'] = len(df)
    metrics['unique_assets'] = df['asset'].nunique()
    metrics['unique_components'] = df['component'].nunique()
    
    # Log level distribution
    level_counts = df['level'].value_counts().to_dict()
    metrics['level_counts'] = level_counts
    metrics['error_rate_pct'] = (level_counts.get('ERROR', 0) / len(df)) * 100
    metrics['warning_rate_pct'] = (level_counts.get('WARNING', 0) / len(df)) * 100
    
    # Latency statistics (using NumPy for efficiency)
    latency = df['latency_ms'].dropna()
    if len(latency) > 0:
        metrics['latency'] = {
            'mean_ms': float(np.mean(latency)),
            'median_ms': float(np.median(latency)),
            'p95_ms': float(np.percentile(latency, 95)),
            'p99_ms': float(np.percentile(latency, 99)),
            'max_ms': float(np.max(latency)),
            'std_ms': float(np.std(latency)),
            'spike_count_gt_1000ms': int((latency > 1000).sum()),
        }
    else:
        metrics['latency'] = "No latency data"
    
    # Errors by component
    error_by_component = df[df['level'] == 'ERROR']['component'].value_counts().head(10).to_dict()
    metrics['top_error_components'] = error_by_component
    
    # Errors by asset
    error_by_asset = df[df['level'] == 'ERROR']['asset'].value_counts().head(10).to_dict()
    metrics['top_error_assets'] = error_by_asset
    
    # Optional: Signal failure rate (if signal_id present on errors)
    if 'signal_id' in df.columns and 'level' in df.columns:
        failed_signals = df[(df['level'] == 'ERROR') & (df['signal_id'].notna())]
        total_signals = df[df['signal_id'].notna()]
        metrics['signal_failure_rate_pct'] = (
            len(failed_signals) / len(total_signals) * 100 if len(total_signals) > 0 else 0
        )
    
    return metrics

# Calculate and display
metrics = calculate_key_metrics(df)

print("Key Monitoring Metrics Calculated:")
for key, value in metrics.items():
    if isinstance(value, dict):
        print(f"\n{key.replace('_', ' ').title()}:")
        for subkey, subval in value.items():
            print(f"   {subkey.replace('_', ' ').title()}: {subval:.2f}" if isinstance(subval, float) else f"   {subkey.replace('_', ' ').title()}: {subval}")
    else:
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

# Add hour column for grouping
df['hour'] = df['timestamp'].dt.floor('H')

# Hourly error counts
hourly_errors = df[df['level'] == 'ERROR'].groupby('hour').size()
hourly_errors.name = 'error_count'

# Hourly average latency
hourly_latency = df.groupby('hour')['latency_ms'].mean()
hourly_latency.name = 'avg_latency_ms'

# Combine
hourly_summary = pd.concat([hourly_errors, hourly_latency], axis=1).fillna(0)
hourly_summary['error_count'] = hourly_summary['error_count'].astype(int)

print("Hourly Summary (Errors and Avg Latency):")
display(hourly_summary.sort_index())

# For Streamlit later – save this
hourly_summary.to_parquet("data/hourly_summary.parquet")


# Sort just in case
df_sorted = df.sort_values('timestamp').reset_index(drop=True)

# Rolling p99 latency over 5-minute windows
df_sorted['latency_p99_5min'] = df_sorted['latency_ms'].rolling(window=50, min_periods=1).quantile(0.99)

# Flag spikes: current latency > 3x rolling median
df_sorted['latency_spike_flag'] = df_sorted['latency_ms'] > 3 * df_sorted['latency_ms'].rolling(window=50, center=True, min_periods=1).median()

spike_events = df_sorted[df_sorted['latency_spike_flag']]

print(f"Detected {len(spike_events)} significant latency spike events (>3x rolling median)")
display(spike_events[['timestamp', 'latency_ms', 'component', 'asset', 'message']].head(10))

# Save flagged events for dashboard
spike_events.to_parquet("data/latency_spikes.parquet")


report = {
    "summary_metrics": metrics,
    # Convert hourly_summary to dict with string-index (timesteps as strings)
    "hourly_summary_sample": hourly_summary.head(10).reset_index().rename(columns={'hour': 'hour'}).to_dict(orient='records'),
    # Ensure timestamps in spikes are strings
    "latency_spike_sample": spike_events[['timestamp', 'latency_ms', 'message']].head(5).assign(timestamp=lambda x: x['timestamp'].astype(str)).to_dict(orient='records')
}

# Custom serializer for any remaining complex types (NumPy, residual Timestamps, etc.)
def json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return str(obj)  # Fallback for Timestamps, etc.

import json
from pathlib import Path

Path("data").mkdir(exist_ok=True)

with open("data/analysis_report.json", "w") as f:
    json.dump(report, f, indent=2, default=json_serializable)

print("Full analysis report successfully saved to data/analysis_report.json")


