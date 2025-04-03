#!/usr/bin/env python3
"""
Options Analysis Runner

This script runs the options analysis module for a list of tickers.
It can be configured via a JSON file or command line arguments.
"""

import os
import json
import argparse
import datetime
from options_analysis import OptionsAnalyzer

def main():
    """Main function to run options analysis"""
    parser = argparse.ArgumentParser(description='Run options analysis for selected tickers')
    
    parser.add_argument('--config', type=str, default='options_analysis_config.json',
                        help='Path to configuration file (default: options_analysis_config.json)')
    parser.add_argument('--ticker', type=str, help='Run analysis for a specific ticker')
    parser.add_argument('--days', type=int, help='Days to expiry (overrides config)')
    parser.add_argument('--all', action='store_true', help='Run analysis for all tickers in config')
    
    args = parser.parse_args()
    
    # Load configuration if it exists
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    else:
        print(f"Configuration file {args.config} not found, using defaults")
        config = {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "days_to_expiry": 90,
            "output_directory": None
        }
    
    # Override config with command-line arguments
    days_to_expiry = args.days if args.days is not None else config.get("days_to_expiry", 90)
    output_dir = config.get("output_directory", None)
    
    # Determine which tickers to analyze
    tickers_to_analyze = []
    
    if args.ticker:
        tickers_to_analyze = [args.ticker]
    elif args.all:
        tickers_to_analyze = config.get("tickers", [])
    else:
        # Interactive selection if no specific ticker or --all flag
        tickers = config.get("tickers", [])
        if tickers:
            print("\nAvailable tickers:")
            for i, ticker in enumerate(tickers):
                print(f"{i+1}. {ticker}")
            
            selection = input("\nSelect ticker(s) to analyze (comma-separated numbers, 'a' for all, or enter new ticker): ")
            
            if selection.lower() == 'a':
                tickers_to_analyze = tickers
            elif ',' in selection:
                # Process comma-separated list of numbers
                try:
                    indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                    tickers_to_analyze = [tickers[idx] for idx in indices if 0 <= idx < len(tickers)]
                except (ValueError, IndexError):
                    print("Invalid selection, using first ticker")
                    tickers_to_analyze = [tickers[0]] if tickers else []
            elif selection.strip().isdigit():
                # Process single number
                try:
                    idx = int(selection.strip()) - 1
                    if 0 <= idx < len(tickers):
                        tickers_to_analyze = [tickers[idx]]
                    else:
                        print("Invalid selection, using first ticker")
                        tickers_to_analyze = [tickers[0]] if tickers else []
                except (ValueError, IndexError):
                    print("Invalid selection, using first ticker")
                    tickers_to_analyze = [tickers[0]] if tickers else []
            else:
                # Treat input as a new ticker symbol
                new_ticker = selection.strip().upper()
                if new_ticker:
                    tickers_to_analyze = [new_ticker]
        else:
            ticker = input("Enter ticker symbol to analyze: ").strip().upper()
            if ticker:
                tickers_to_analyze = [ticker]
    
    # Run analysis for each ticker
    if not tickers_to_analyze:
        print("No tickers selected for analysis. Exiting.")
        return
    
    start_time = datetime.datetime.now()
    print(f"\nStarting analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will analyze {len(tickers_to_analyze)} ticker(s) with {days_to_expiry} days to expiry")
    
    for i, ticker in enumerate(tickers_to_analyze):
        print(f"\n[{i+1}/{len(tickers_to_analyze)}] Analyzing {ticker}...")
        try:
            analyzer = OptionsAnalyzer(ticker, days_to_expiry=days_to_expiry, output_dir=output_dir)
            analyzer.run_full_analysis()
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nAnalysis completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main()