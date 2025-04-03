# Options Analysis Web Application

This web application provides a user-friendly interface to run options analysis and view visualizations for different tickers.

## Features

- **Ticker Analysis**: Enter any ticker symbol to run a complete options analysis
- **Visualizations Gallery**: View all generated charts organized by category
- **Interactive 3D Plots**: Open interactive volatility surface visualizations
- **Real-time Status**: See which analyses are running and automatically refresh when complete
- **Result Downloads**: Access CSV files with detailed analysis results

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Flask
- All dependencies from the main options analysis module

### Installation

1. Ensure you have all required packages:

```bash
pip install flask pandas numpy matplotlib plotly yfinance scipy
```

2. Run the web application:

```bash
python app.py
```

3. Open your browser and navigate to:

```
http://localhost:5000
```

## Using the Application

### Analyzing a New Ticker

1. Enter a valid ticker symbol (e.g., AAPL, MSFT, GOOGL) in the form at the top of the home page
2. Specify the maximum days to expiry (default: 90 days)
3. Click "Run Analysis"
4. The analysis will run in the background; you can watch its progress or analyze other tickers

### Viewing Visualizations

1. Click on any analyzed ticker from the home page
2. Visualizations are organized into categories:
   - Volatility Surface
   - Gamma Exposure
   - Delta Exposure
   - Put/Call Ratio
   - Support & Resistance
3. Click on any image to view it in full size
4. For 3D interactive plots, click the "Open Interactive 3D Plot" button

### Refreshing Analysis

To update the analysis for a ticker:
1. Navigate to the ticker's visualization page
2. Click the "Refresh Analysis" button at the top of the page

## Directory Structure

The web application manages the following directories:

- `charts/`: Stores all generated visualizations organized by ticker
- `data/`: Contains raw options data in CSV format
- `results/`: Contains analysis results in CSV format
- `templates/`: Contains HTML templates for the web interface
- `static/`: Serves static files for the web application

## Technical Notes

- The application uses Flask as the web framework
- Analyses run in background threads to prevent blocking the web interface
- The visualization page automatically refreshes when an analysis is complete
- Interactive 3D plots are served as standalone HTML files with Plotly.js

## Troubleshooting

- If images don't appear, check that the charts directory exists and has proper permissions
- If analysis fails, check the console output for error messages
- For performance issues, consider limiting the days to expiry to reduce the amount of data processed