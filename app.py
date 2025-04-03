#!/usr/bin/env python3
"""
Options Analysis Web Application

A Flask-based web app that allows users to generate and view options analysis visualizations.
"""

import os
import glob
import datetime
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from options_analysis import OptionsAnalyzer

app = Flask(__name__)
app.secret_key = 'options_analysis_secret_key'

# Directory configuration
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Create directories if they don't exist
for directory in [CHARTS_DIR, DATA_DIR, RESULTS_DIR, STATIC_DIR]:
    os.makedirs(directory, exist_ok=True)

# Create static image directory symlink if it doesn't exist
CHARTS_STATIC_DIR = os.path.join(STATIC_DIR, 'charts')
if not os.path.exists(CHARTS_STATIC_DIR):
    try:
        # Create symlink on Unix/Linux/Mac
        os.symlink(CHARTS_DIR, CHARTS_STATIC_DIR)
    except:
        # If symlink fails (e.g., on Windows), create a directory and copy files when needed
        os.makedirs(CHARTS_STATIC_DIR, exist_ok=True)

# Dictionary to track running analyses
running_analyses = {}

def analyze_ticker(ticker, days_to_expiry):
    """Run analysis for a ticker in a separate thread."""
    try:
        analyzer = OptionsAnalyzer(ticker, days_to_expiry=days_to_expiry)
        analyzer.run_full_analysis()
        running_analyses[ticker] = False  # Mark as completed
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        running_analyses[ticker] = False  # Mark as failed

@app.route('/')
def index():
    """Home page with form to enter ticker."""
    tickers_with_data = []
    
    # Get list of tickers with data
    if os.path.exists(CHARTS_DIR):
        tickers_with_data = [os.path.basename(d) for d in glob.glob(os.path.join(CHARTS_DIR, '*')) 
                           if os.path.isdir(d)]
    
    # Get list of running analyses
    current_analyses = {ticker: status for ticker, status in running_analyses.items()}
    
    return render_template('index.html', 
                         tickers_with_data=sorted(tickers_with_data),
                         running_analyses=current_analyses)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle form submission to analyze a new ticker."""
    ticker = request.form.get('ticker', '').strip().upper()
    days_to_expiry = int(request.form.get('days_to_expiry', 90))
    
    if not ticker:
        flash('Please enter a valid ticker symbol', 'error')
        return redirect(url_for('index'))
    
    # Check if analysis is already running for this ticker
    if running_analyses.get(ticker, False):
        flash(f'Analysis for {ticker} is already running', 'info')
        return redirect(url_for('index'))
    
    # Start analysis in a separate thread
    running_analyses[ticker] = True
    thread = threading.Thread(target=analyze_ticker, args=(ticker, days_to_expiry))
    thread.daemon = True
    thread.start()
    
    flash(f'Analysis for {ticker} started', 'success')
    return redirect(url_for('index'))

@app.route('/visualizations/<ticker>')
def visualizations(ticker):
    """Display visualizations for a ticker."""
    ticker = ticker.upper()
    
    # Check if ticker directory exists
    ticker_dir = os.path.join(CHARTS_DIR, ticker)
    if not os.path.exists(ticker_dir):
        flash(f'No data found for {ticker}', 'error')
        return redirect(url_for('index'))
    
    # Find all PNG files for this ticker
    image_files = sorted(glob.glob(os.path.join(ticker_dir, f"{ticker}_*.png")))
    
    # Get image filenames without path
    image_filenames = [os.path.basename(f) for f in image_files]
    
    # Get HTML files (for 3D plots)
    html_files = sorted(glob.glob(os.path.join(ticker_dir, f"{ticker}_*.html")))
    html_filenames = [os.path.basename(f) for f in html_files]
    
    # Get results CSV files
    results_dir = os.path.join(RESULTS_DIR, ticker)
    csv_files = []
    if os.path.exists(results_dir):
        csv_files = sorted(glob.glob(os.path.join(results_dir, f"{ticker}_*.csv")))
    csv_filenames = [os.path.basename(f) for f in csv_files]
    
    # Organize images by type
    image_categories = {
        'Volatility Surface': [f for f in image_filenames if 'vol_surface' in f or 'vol_heatmap' in f],
        'Gamma Exposure': [f for f in image_filenames if 'gamma_profile' in f or 'multi_expiry_gex' in f],
        'Delta Exposure': [f for f in image_filenames if 'delta_profile' in f],
        'Put/Call Ratio': [f for f in image_filenames if 'pcr_chart' in f],
        'Support & Resistance': [f for f in image_filenames if 'support_resistance' in f],
        'Other': [f for f in image_filenames if not any(x in f for x in ['vol_', 'gamma_', 'delta_', 'pcr_', 'support_'])]
    }
    
    # Check if analysis is still running
    is_running = running_analyses.get(ticker, False)
    
    return render_template('visualizations.html', 
                         ticker=ticker,
                         image_categories=image_categories,
                         html_files=html_filenames,
                         csv_files=csv_filenames,
                         is_running=is_running)

@app.route('/charts/<path:filename>')
def charts(filename):
    """Serve chart images."""
    # Extract ticker from filename (assumed format: TICKER_something.png)
    parts = filename.split('_')
    if len(parts) > 1:
        ticker = parts[0].upper()
        return send_from_directory(os.path.join(CHARTS_DIR, ticker), filename)
    return "File not found", 404

@app.route('/html/<path:filename>')
def html_file(filename):
    """Serve HTML files (3D plots)."""
    parts = filename.split('_')
    if len(parts) > 1:
        ticker = parts[0].upper()
        return send_from_directory(os.path.join(CHARTS_DIR, ticker), filename)
    return "File not found", 404

@app.route('/results/<path:filename>')
def results(filename):
    """Serve results CSV files."""
    parts = filename.split('_')
    if len(parts) > 1:
        ticker = parts[0].upper()
        return send_from_directory(os.path.join(RESULTS_DIR, ticker), filename)
    return "File not found", 404

@app.route('/refresh/<ticker>')
def refresh(ticker):
    """Refresh status for a specific ticker."""
    ticker = ticker.upper()
    is_running = running_analyses.get(ticker, False)
    return {'is_running': is_running}

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Options Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            background-color: #f5f5f5;
        }
        .ticker-card {
            transition: transform 0.2s;
            margin-bottom: 15px;
        }
        .ticker-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .badge-running {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4 text-center">Options Analysis Dashboard</h1>
        
        <!-- Flash Messages -->
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category if category != 'error' else 'danger' }} alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <!-- Analysis Form -->
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="card-title mb-0">Analyze New Ticker</h5>
            </div>
            <div class="card-body">
                <form action="{{ url_for('analyze') }}" method="post">
                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label for="ticker" class="form-label">Ticker Symbol</label>
                            <input type="text" class="form-control" id="ticker" name="ticker" placeholder="e.g., AAPL" required>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label for="days_to_expiry" class="form-label">Days to Expiry</label>
                            <input type="number" class="form-control" id="days_to_expiry" name="days_to_expiry" value="90" min="1" max="365">
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary">Run Analysis</button>
                </form>
            </div>
        </div>
        
        <!-- Available Tickers -->
        <h2 class="mb-3">Available Tickers</h2>
        {% if tickers_with_data %}
            <div class="row">
                {% for ticker in tickers_with_data %}
                    <div class="col-md-3">
                        <div class="card ticker-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">{{ ticker }}</h5>
                                {% if running_analyses.get(ticker, False) %}
                                    <span class="badge bg-warning badge-running">Analysis Running</span>
                                {% endif %}
                                <a href="{{ url_for('visualizations', ticker=ticker) }}" class="btn btn-outline-primary btn-sm mt-2">View Visualizations</a>
                            </div>
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="alert alert-info">
                No ticker data available yet. Use the form above to analyze a ticker.
            </div>
        {% endif %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
""")
    
    # Create visualizations.html template if it doesn't exist
    if not os.path.exists('templates/visualizations.html'):
        with open('templates/visualizations.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ ticker }} Options Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            background-color: #f5f5f5;
        }
        .viz-container {
            margin-bottom: 30px;
        }
        .viz-image {
            max-width: 100%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        .viz-image:hover {
            transform: scale(1.02);
        }
        .badge-running {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .btn-category {
            margin-right: 5px;
            margin-bottom: 5px;
        }
        #loading-indicator {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background-color: #007bff;
            animation: loading 2s infinite;
        }
        @keyframes loading {
            0% { width: 0%; }
            50% { width: 65%; }
            100% { width: 100%; }
        }
    </style>
</head>
<body>
    {% if is_running %}
    <div id="loading-indicator"></div>
    {% endif %}
    
    <div class="container">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1>{{ ticker }} Options Analysis</h1>
            <div>
                <a href="{{ url_for('index') }}" class="btn btn-outline-secondary">Back to Home</a>
                <a href="{{ url_for('analyze') }}" class="btn btn-primary" onclick="event.preventDefault(); document.getElementById('refresh-form').submit();">Refresh Analysis</a>
                <form id="refresh-form" action="{{ url_for('analyze') }}" method="post" style="display:none;">
                    <input type="hidden" name="ticker" value="{{ ticker }}">
                    <input type="hidden" name="days_to_expiry" value="90">
                </form>
            </div>
        </div>
        
        {% if is_running %}
            <div class="alert alert-info">
                <strong>Analysis in progress...</strong> This page will automatically update when new visualizations are available.
            </div>
        {% endif %}
        
        <!-- Category Navigation -->
        <div class="mb-4 category-nav">
            <h5>Jump to:</h5>
            {% for category, images in image_categories.items() %}
                {% if images %}
                    <a href="#{{ category|replace(' ', '-')|lower }}" class="btn btn-outline-primary btn-category">{{ category }}</a>
                {% endif %}
            {% endfor %}
            {% if html_files %}
                <a href="#interactive-plots" class="btn btn-outline-success btn-category">Interactive 3D Plots</a>
            {% endif %}
            {% if csv_files %}
                <a href="#data-tables" class="btn btn-outline-info btn-category">Data Tables</a>
            {% endif %}
        </div>
        
        <!-- Static Images -->
        {% for category, images in image_categories.items() %}
            {% if images %}
                <div class="viz-container" id="{{ category|replace(' ', '-')|lower }}">
                    <h2 class="mb-3">{{ category }}</h2>
                    <div class="row">
                        {% for image in images %}
                            <div class="col-md-6 mb-4">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">{{ image|replace(ticker ~ '_', '')|replace('.png', '')|replace('_', ' ')|title }}</h5>
                                    </div>
                                    <div class="card-body text-center">
                                        <a href="{{ url_for('charts', filename=image) }}" target="_blank">
                                            <img src="{{ url_for('charts', filename=image) }}" class="viz-image" alt="{{ image }}">
                                        </a>
                                    </div>
                                </div>
                            </div>
                        {% endfor %}
                    </div>
                </div>
            {% endif %}
        {% endfor %}
        
        <!-- Interactive 3D Plots -->
        {% if html_files %}
            <div class="viz-container" id="interactive-plots">
                <h2 class="mb-3">Interactive 3D Plots</h2>
                <div class="row">
                    {% for html_file in html_files %}
                        <div class="col-md-6 mb-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">{{ html_file|replace(ticker ~ '_', '')|replace('.html', '')|replace('_', ' ')|title }}</h5>
                                </div>
                                <div class="card-body text-center">
                                    <a href="{{ url_for('html_file', filename=html_file) }}" class="btn btn-primary" target="_blank">
                                        Open Interactive 3D Plot
                                    </a>
                                    <p class="mt-2 text-muted">
                                        <small>Opens in a new window. Allows rotation, zooming, and interaction.</small>
                                    </p>
                                </div>
                            </div>
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% endif %}
        
        <!-- Results Data Tables -->
        {% if csv_files %}
            <div class="viz-container" id="data-tables">
                <h2 class="mb-3">Analysis Results</h2>
                <div class="row">
                    {% for csv_file in csv_files %}
                        <div class="col-md-6 mb-4">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">{{ csv_file|replace(ticker ~ '_', '')|replace('.csv', '')|replace('_', ' ')|title }}</h5>
                                </div>
                                <div class="card-body text-center">
                                    <a href="{{ url_for('results', filename=csv_file) }}" class="btn btn-info" target="_blank">
                                        Download CSV
                                    </a>
                                </div>
                            </div>
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% endif %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Auto-refresh when analysis is running -->
    {% if is_running %}
    <script>
        // Check status every 5 seconds
        const checkStatus = () => {
            fetch('/refresh/{{ ticker }}')
                .then(response => response.json())
                .then(data => {
                    if (!data.is_running) {
                        window.location.reload();
                    } else {
                        setTimeout(checkStatus, 5000);
                    }
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                    setTimeout(checkStatus, 10000);
                });
        };
        
        setTimeout(checkStatus, 5000);
    </script>
    {% endif %}
</body>
</html>
""")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8000)