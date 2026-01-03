# Quantitative Research Log Analyzer

A professional, production-ready internal tool built in Python to support quantitative researchers in monitoring system correctness, validating data quality, and performing exploratory log analysis.

This Streamlit dashboard parses, analyzes, and visualizes large volumes of unstructured research/trading system logs — demonstrating the exact skills required for a Python Analyst role in quantitative trading firms

**Live Demo**: [https://quantloganalyzer-epiqhjt8h3z2wvqs6wjmvx.streamlit.app/] 


## Project Overview

This tool simulates real-world log monitoring workflows in high-performance quant environments:
- Handles noisy, high-volume logs with timestamps, latency, components, assets, and optional fields
- Supports file upload for ad-hoc analysis
- Provides interactive filtering, metrics, visualizations, and anomaly alerts
- Exports automated reports

Built with clean, readable, production-quality code using **Python, Pandas, NumPy, and Streamlit**.

## Phase-by-Phase Summary

- **Generating Realistic Synthetic Dataset**  
  Created ~5,000 realistic synthetic log entries in JSONL format mimicking quant trading/research systems, including natural distributions, latency spikes, error clustering, and quant-specific fields.

- **Log Parsing and Structuring**  
  Implemented robust line-by-line JSONL parser with graceful handling of malformed lines, converted data to a typed Pandas DataFrame, and added validation checks for data quality issues.

- **Analysis and Metric Calculation**  
  Computed key monitoring metrics using Pandas and NumPy (error/warning rates, latency percentiles, hourly breakdowns, top failing components/assets, signal failure rates) and saved intermediate results.

- **Anomaly Detection and Alerting**  
  Built multi-layer anomaly detection (latency z-score spikes, error bursts, component failures) with severity levels and generated timestamped JSON/CSV alert reports.

- **Interactive Streamlit Dashboard**  
  Developed a full interactive web application with file upload, sidebar metrics, multi-filtering, log level distribution overview, robust visualizations, live anomaly alerts, and one-click report generation.

## Tech Stack
- **Python** – Core language with clean, PEP8-compliant code
- **Pandas** – Data structuring, filtering, and grouping
- **NumPy** – Efficient statistical calculations
- **Streamlit** – Interactive dashboard with live updates
- **Matplotlib** – Custom charts with labels and theme compatibility
