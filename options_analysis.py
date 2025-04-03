#!/usr/bin/env python3
"""
Options Analysis and Visualization Module

This module implements options analysis concepts including:
- Volatility surface visualization
- Put/Call ratio analysis
- Gamma and delta exposure calculations
- Gamma flip detection
- Options volume analysis

Based on financial theory documents in pqf/resources/
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import yfinance as yf
from scipy.interpolate import griddata
import seaborn as sns
from scipy.stats import norm

# Create directories for data storage if they don't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')

for directory in [DATA_DIR, RESULTS_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class OptionsAnalyzer:
    """Class for analyzing options data and visualizing results."""
    
    def __init__(self, ticker, days_to_expiry=None, output_dir=None):
        """
        Initialize the OptionsAnalyzer with a ticker symbol.
        
        Args:
            ticker (str): Stock ticker symbol
            days_to_expiry (int, optional): Filter expirations to this number of days. Default is None (all expirations)
            output_dir (str, optional): Directory for output files. Default is None (uses module directories)
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.current_price = self.stock.history(period="1d")['Close'].iloc[-1]
        self.today = datetime.datetime.now().date()
        self.days_to_expiry = days_to_expiry
        
        # Set output directories
        self.data_dir = output_dir or DATA_DIR
        self.results_dir = output_dir or RESULTS_DIR
        self.charts_dir = output_dir or CHARTS_DIR
        
        # Get options data
        self.expirations = self.stock.options
        
        # Filter expirations if days_to_expiry is specified
        if days_to_expiry:
            filtered_exp = []
            for exp in self.expirations:
                exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
                days = (exp_date - self.today).days
                if days <= days_to_expiry:
                    filtered_exp.append(exp)
            self.expirations = filtered_exp
        
        # Load all options data
        self.calls = []
        self.puts = []
        self.load_options_data()
    
    def load_options_data(self):
        """Load options data for all expirations."""
        all_calls = []
        all_puts = []
        
        for expiry in self.expirations:
            try:
                chain = self.stock.option_chain(expiry)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                
                # Add expiration date
                calls["expirationDate"] = expiry
                puts["expirationDate"] = expiry
                
                # Convert expiration to datetime
                exp_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
                
                # Add days to expiry
                calls["daysToExpiry"] = (exp_date - self.today).days
                puts["daysToExpiry"] = (exp_date - self.today).days
                
                # Add expiration ordinal for plotting
                calls["expirationOrdinal"] = exp_date.toordinal()
                puts["expirationOrdinal"] = exp_date.toordinal()
                
                # Calculate moneyness (strike/spot - 1)
                calls["moneyness"] = (calls["strike"] / self.current_price) - 1
                puts["moneyness"] = (puts["strike"] / self.current_price) - 1
                
                all_calls.append(calls)
                all_puts.append(puts)
                
            except Exception as e:
                print(f"Error loading options for {expiry}: {e}")
        
        if all_calls and all_puts:
            self.calls = pd.concat(all_calls, ignore_index=True)
            self.puts = pd.concat(all_puts, ignore_index=True)
            
            # Save data to CSV files
            os.makedirs(os.path.join(self.data_dir, self.ticker), exist_ok=True)
            self.calls.to_csv(os.path.join(self.data_dir, self.ticker, f"{self.ticker}_calls.csv"), index=False)
            self.puts.to_csv(os.path.join(self.data_dir, self.ticker, f"{self.ticker}_puts.csv"), index=False)
            
            print(f"Loaded {len(self.calls)} calls and {len(self.puts)} puts for {self.ticker}")
        else:
            print(f"No options data found for {self.ticker}")
    
    def analyze_put_call_ratio(self):
        """
        Calculate Put/Call ratio based on volume and open interest.
        
        Returns:
            dict: Dictionary with PCR values and interpretation
        """
        # Calculate PCR by volume
        total_call_volume = self.calls['volume'].sum()
        total_put_volume = self.puts['volume'].sum()
        
        if total_call_volume > 0:
            pcr_volume = total_put_volume / total_call_volume
        else:
            pcr_volume = float('inf')
        
        # Calculate PCR by open interest
        total_call_oi = self.calls['openInterest'].sum()
        total_put_oi = self.puts['openInterest'].sum()
        
        if total_call_oi > 0:
            pcr_oi = total_put_oi / total_call_oi
        else:
            pcr_oi = float('inf')
        
        # Interpret PCR values
        if pcr_volume > 1.2:
            volume_interpretation = "Bearish (high put activity)"
        elif pcr_volume < 0.7:
            volume_interpretation = "Bullish (high call activity)"
        else:
            volume_interpretation = "Neutral"
            
        if pcr_oi > 1.2:
            oi_interpretation = "Bearish positioning (high put open interest)"
        elif pcr_oi < 0.7:
            oi_interpretation = "Bullish positioning (high call open interest)"
        else:
            oi_interpretation = "Neutral positioning"
        
        results = {
            "ticker": self.ticker,
            "spot_price": self.current_price,
            "date": self.today.strftime("%Y-%m-%d"),
            "put_volume": total_put_volume,
            "call_volume": total_call_volume,
            "pcr_volume": pcr_volume,
            "volume_interpretation": volume_interpretation,
            "put_open_interest": total_put_oi,
            "call_open_interest": total_call_oi,
            "pcr_open_interest": pcr_oi,
            "oi_interpretation": oi_interpretation
        }
        
        # Save results to CSV
        os.makedirs(os.path.join(self.results_dir, self.ticker), exist_ok=True)
        pd.DataFrame([results]).to_csv(
            os.path.join(self.results_dir, self.ticker, f"{self.ticker}_pcr_analysis_{self.today.strftime('%Y%m%d')}.csv"),
            index=False
        )
        
        return results
    
    def plot_put_call_ratio(self):
        """Create a plot of put/call ratio by expiration date."""
        # Group by expiration date and calculate PCR for each
        pcr_by_exp = []
        
        for exp in self.expirations:
            calls_exp = self.calls[self.calls['expirationDate'] == exp]
            puts_exp = self.puts[self.puts['expirationDate'] == exp]
            
            call_vol = calls_exp['volume'].sum()
            put_vol = puts_exp['volume'].sum()
            
            call_oi = calls_exp['openInterest'].sum()
            put_oi = puts_exp['openInterest'].sum()
            
            # Calculate PCR or set to NaN if division by zero
            pcr_vol = put_vol / call_vol if call_vol > 0 else np.nan
            pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan
            
            days_to_exp = (datetime.datetime.strptime(exp, '%Y-%m-%d').date() - self.today).days
            
            pcr_by_exp.append({
                'expiration': exp,
                'days_to_expiry': days_to_exp,
                'pcr_volume': pcr_vol,
                'pcr_open_interest': pcr_oi,
                'total_volume': call_vol + put_vol,
                'total_oi': call_oi + put_oi
            })
        
        if not pcr_by_exp:
            print("No data available for PCR plot")
            return
            
        pcr_df = pd.DataFrame(pcr_by_exp)
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Days to Expiry')
        ax1.set_ylabel('Put/Call Ratio', color=color)
        ax1.plot(pcr_df['days_to_expiry'], pcr_df['pcr_volume'], 'o-', color=color, label='PCR (Volume)')
        ax1.plot(pcr_df['days_to_expiry'], pcr_df['pcr_open_interest'], 's-', color='tab:purple', label='PCR (Open Interest)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add second y-axis for volume
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Total Volume', color=color)
        ax2.bar(pcr_df['days_to_expiry'], pcr_df['total_volume'], alpha=0.3, color=color, label='Total Volume')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'{self.ticker} Put/Call Ratio by Expiration')
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
        plt.savefig(os.path.join(self.charts_dir, self.ticker, f"{self.ticker}_pcr_chart_{self.today.strftime('%Y%m%d')}.png"), dpi=300)
        plt.close()

    def calculate_greeks(self, r=0.05, bins=20):
        """
        Calculate option Greeks (delta, gamma) for analysis, incorporating dividend yield
        and basic filtering.

        Args:
            r (float): Risk-free rate (annualized)
            bins (int): Number of price bins for aggregating data

        Returns:
            pandas.DataFrame: DataFrame with aggregated Greeks by strike price, or None if fails.
        """
        # --- Verbesserungen starten hier ---

        # 1. Dividendenrendite holen (optional, aber empfohlen)
        try:
            # Versuche, die Dividendenrendite von yfinance zu bekommen
            # stock.info kann manchmal langsam oder unzuverlässig sein
            dividend_yield = self.stock.info.get('dividendYield', 0.0)
            if dividend_yield is None: # Manchmal gibt yfinance None zurück
                dividend_yield = 0.0
            print(f"Using dividend yield (q): {dividend_yield:.4f}")
        except Exception as e:
            print(f"Warning: Could not fetch dividend yield for {self.ticker}, assuming 0. Error: {e}")
            dividend_yield = 0.0
        q = dividend_yield # Kürzerer Name für die Formeln

        # Sicherstellen, dass Daten vorhanden sind
        if self.calls.empty or self.puts.empty:
            print("Error: Calls or Puts DataFrame is empty before Greek calculation.")
            return None

        # 2. Daten filtern (optional, aber empfohlen)
        # Entferne Optionen mit Null Open Interest oder fehlender IV
        # Du könntest auch nach Volumen oder Bid/Ask-Spread filtern
        self.calls = self.calls[self.calls['openInterest'].notna() & (self.calls['openInterest'] > 0)]
        self.calls = self.calls[self.calls['impliedVolatility'].notna() & (self.calls['impliedVolatility'] > 1e-6)] # IV > 0
        self.puts = self.puts[self.puts['openInterest'].notna() & (self.puts['openInterest'] > 0)]
        self.puts = self.puts[self.puts['impliedVolatility'].notna() & (self.puts['impliedVolatility'] > 1e-6)] # IV > 0

        if self.calls.empty or self.puts.empty:
            print("Error: Calls or Puts DataFrame is empty after filtering.")
            return None

        # 3. Griechenberechnung mit Dividendenrendite (q)
        S = self.current_price

        # --- Calls ---
        T_calls = self.calls['daysToExpiry'] / 365.0
        T_calls = T_calls.replace(0, 1e-6) # Ersetze 0 durch einen sehr kleinen Wert statt 1/365
        sigma_calls = self.calls['impliedVolatility']
        K_calls = self.calls['strike']

        # Berechne d1 und d2 mit Dividendenrendite q
        # Handle potenzielle Fehler bei Logarithmus oder Division durch Null
        with np.errstate(divide='ignore', invalid='ignore'): # Ignoriere Warnungen temporär
            d1_calls = (np.log(S / K_calls) + (r - q + 0.5 * sigma_calls**2) * T_calls) / (sigma_calls * np.sqrt(T_calls))
            d2_calls = d1_calls - sigma_calls * np.sqrt(T_calls)

            # Delta für Calls: exp(-qT) * N(d1)
            self.calls['delta'] = np.exp(-q * T_calls) * norm.cdf(d1_calls)
            # Gamma für Calls und Puts (ist gleich): exp(-qT) * N'(d1) / (S * sigma * sqrt(T))
            self.calls['gamma'] = np.exp(-q * T_calls) * norm.pdf(d1_calls) / (S * sigma_calls * np.sqrt(T_calls))

        # --- Puts ---
        T_puts = self.puts['daysToExpiry'] / 365.0
        T_puts = T_puts.replace(0, 1e-6) # Ersetze 0 durch einen sehr kleinen Wert statt 1/365
        sigma_puts = self.puts['impliedVolatility']
        K_puts = self.puts['strike']

        with np.errstate(divide='ignore', invalid='ignore'): # Ignoriere Warnungen temporär
            d1_puts = (np.log(S / K_puts) + (r - q + 0.5 * sigma_puts**2) * T_puts) / (sigma_puts * np.sqrt(T_puts))
            d2_puts = d1_puts - sigma_puts * np.sqrt(T_puts)

            # Delta für Puts: exp(-qT) * (N(d1) - 1)
            self.puts['delta'] = np.exp(-q * T_puts) * (norm.cdf(d1_puts) - 1)
            # Gamma für Calls und Puts (ist gleich): exp(-qT) * N'(d1) / (S * sigma * sqrt(T))
            self.puts['gamma'] = np.exp(-q * T_puts) * norm.pdf(d1_puts) / (S * sigma_puts * np.sqrt(T_puts))

        # --- Ersetze NaN/inf Werte, die durch Berechnungsfehler entstehen könnten ---
        greeks_cols = ['delta', 'gamma']
        self.calls.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.puts.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.calls.dropna(subset=greeks_cols, inplace=True)
        self.puts.dropna(subset=greeks_cols, inplace=True)

        if self.calls.empty or self.puts.empty:
            print("Error: Calls or Puts DataFrame is empty after calculating and cleaning Greeks.")
            return None

        # --- Verbesserungen enden hier ---

        # 4. Exposure berechnen (wie zuvor)
        self.calls['delta_exposure'] = self.calls['delta'] * self.calls['openInterest'] * 100 # Pro Kontrakt (100 Aktien)
        self.puts['delta_exposure'] = self.puts['delta'] * self.puts['openInterest'] * 100 # Pro Kontrakt (100 Aktien)
        self.calls['gamma_exposure'] = self.calls['gamma'] * self.calls['openInterest'] * 100 # Pro Kontrakt (100 Aktien)
        self.puts['gamma_exposure'] = self.puts['gamma'] * self.puts['openInterest'] * 100 # Pro Kontrakt (100 Aktien)

        # Skaliere Gamma Exposure oft mit (Underlying Price)^2 / 10^9 für bessere Lesbarkeit (optional)
        # self.calls['gamma_exposure_scaled'] = self.calls['gamma_exposure'] * S**2 / 1e9
        # self.puts['gamma_exposure_scaled'] = self.puts['gamma_exposure'] * S**2 / 1e9

        # Save enhanced data
        self.calls.to_csv(os.path.join(self.data_dir, self.ticker, f"{self.ticker}_calls_with_greeks.csv"), index=False)
        self.puts.to_csv(os.path.join(self.data_dir, self.ticker, f"{self.ticker}_puts_with_greeks.csv"), index=False)

        # 5. Aggregation nach Strike Bins (wie zuvor, aber mit den neuen Daten)
        min_strike = min(self.calls['strike'].min(), self.puts['strike'].min())
        max_strike = max(self.calls['strike'].max(), self.puts['strike'].max())

        # Stelle sicher, dass min < max
        if min_strike >= max_strike:
             # Fallback: Erstelle Bins um den aktuellen Preis herum, wenn Strikes zu eng sind
             print(f"Warning: min_strike ({min_strike}) >= max_strike ({max_strike}). Creating bins around current price.")
             center = self.current_price
             width = max(50, self.current_price * 0.3) # Mindestens 50 Punkte oder 30% Breite
             min_strike = center - width / 2
             max_strike = center + width / 2
             if min_strike >= max_strike: # Letzter Ausweg
                  print("Error: Cannot create valid strike range for aggregation.")
                  return None

        strike_range = np.linspace(min_strike, max_strike, bins + 1) # bins+1 Kanten für 'bins' Intervalle

        def aggregate_greeks_by_strike(df, strike_range):
            result = []
            for i in range(len(strike_range) - 1):
                low = strike_range[i]
                high = strike_range[i+1]
                # Wichtig: beim letzten Bin auch die Obergrenze einschließen
                if i == len(strike_range) - 2:
                    mask = (df['strike'] >= low) & (df['strike'] <= high)
                else:
                    mask = (df['strike'] >= low) & (df['strike'] < high)

                if mask.any():
                    subset = df[mask]
                    # Stelle sicher, dass Spalten existieren, bevor sum() aufgerufen wird
                    delta_exp = subset['delta_exposure'].sum() if 'delta_exposure' in subset else 0
                    gamma_exp = subset['gamma_exposure'].sum() if 'gamma_exposure' in subset else 0
                    oi = subset['openInterest'].sum() if 'openInterest' in subset else 0
                    vol = subset['volume'].sum() if 'volume' in subset else 0

                    result.append({
                        'strike_low': low,
                        'strike_high': high,
                        'strike_mid': (low + high) / 2,
                        'delta_exposure': delta_exp,
                        'gamma_exposure': gamma_exp,
                        'open_interest': oi,
                        'volume': vol
                    })
            if not result: # Wenn keine Daten aggregiert werden konnten
                return pd.DataFrame()
            return pd.DataFrame(result)

        call_agg = aggregate_greeks_by_strike(self.calls, strike_range)
        put_agg = aggregate_greeks_by_strike(self.puts, strike_range)

        # Combine for total exposure
        if call_agg.empty and put_agg.empty:
             print("Warning: Aggregated dataframes for calls and puts are both empty.")
             return None
        elif call_agg.empty:
             print("Warning: Aggregated dataframe for calls is empty.")
             total_exposure = put_agg.rename(columns={
                  'delta_exposure': 'put_delta_exposure',
                  'gamma_exposure': 'put_gamma_exposure',
                  'open_interest': 'put_open_interest',
             })
             total_exposure['call_delta_exposure'] = 0
             total_exposure['call_gamma_exposure'] = 0
             total_exposure['call_open_interest'] = 0
             total_exposure = total_exposure.set_index('strike_mid')

        elif put_agg.empty:
            print("Warning: Aggregated dataframe for puts is empty.")
            total_exposure = call_agg.rename(columns={
                  'delta_exposure': 'call_delta_exposure',
                  'gamma_exposure': 'call_gamma_exposure',
                  'open_interest': 'call_open_interest',
             })
            total_exposure['put_delta_exposure'] = 0
            total_exposure['put_gamma_exposure'] = 0
            total_exposure['put_open_interest'] = 0
            total_exposure = total_exposure.set_index('strike_mid')
        else:
            # Merge data using outer join to keep all strikes
            total_exposure = pd.merge(
                call_agg[['strike_mid', 'delta_exposure', 'gamma_exposure', 'open_interest']],
                put_agg[['strike_mid', 'delta_exposure', 'gamma_exposure', 'open_interest']],
                on='strike_mid',
                how='outer',
                suffixes=('_call', '_put')
            ).fillna(0) # Fülle fehlende Werte mit 0 auf

            # Umbenennen für Klarheit
            total_exposure.rename(columns={
                'delta_exposure_call': 'call_delta_exposure',
                'gamma_exposure_call': 'call_gamma_exposure',
                'open_interest_call': 'call_open_interest',
                'delta_exposure_put': 'put_delta_exposure',
                'gamma_exposure_put': 'put_gamma_exposure',
                'open_interest_put': 'put_open_interest',
                'strike_mid': 'strike' # Umbenennen für Konsistenz mit vorherigem Code
            }, inplace=True)
            total_exposure = total_exposure.set_index('strike') # Setze Strike als Index
            total_exposure.sort_index(inplace=True) # Sortiere nach Strike


        # Calculate net exposure
        total_exposure['net_delta_exposure'] = total_exposure['call_delta_exposure'] + total_exposure['put_delta_exposure']
        total_exposure['net_gamma_exposure'] = total_exposure['call_gamma_exposure'] + total_exposure['put_gamma_exposure']

        # Save exposure data
        total_exposure.reset_index().to_csv( # reset_index to save strike column
            os.path.join(self.results_dir, self.ticker, f"{self.ticker}_greeks_exposure_{self.today.strftime('%Y%m%d')}.csv"),
            index=False
        )

        return total_exposure.reset_index() # Gib DF mit Strike als Spalte zurück
        

    def detect_gamma_flip(self):
        """
        Detect gamma flip points where gamma exposure changes from positive to negative.
        
        Returns:
            dict: Information about gamma flip points
        """
        # Calculate Greeks if not already done
        exposure_df = self.calculate_greeks()
        
        if exposure_df is None or exposure_df.empty:
            return None
        
        # Find where net gamma exposure changes sign
        exposure_df['sign_change'] = exposure_df['net_gamma_exposure'].apply(np.sign).diff().fillna(0)
        
        # Get gamma flip points (where sign changes from positive to negative or vice versa)
        flip_points = exposure_df[exposure_df['sign_change'] != 0].copy()
        
        if flip_points.empty:
            print("No gamma flip points detected")
            return None
        
        # For each flip point, determine if it's near the current price
        flip_points['distance_from_spot'] = abs(flip_points['strike'] - self.current_price)
        flip_points['distance_percent'] = (flip_points['distance_from_spot'] / self.current_price) * 100
        
        # Find the nearest flip point to current price
        try:
            nearest_flip = flip_points.loc[flip_points['distance_from_spot'].idxmin()]
            
            # Determine if flip is from positive to negative or vice versa
            if nearest_flip['sign_change'] > 0:
                flip_direction = "negative to positive (supportive market dynamics)"
            else:
                flip_direction = "positive to negative (potential for increased volatility)"
            
            # Safely get index positions
            flip_index_pos = exposure_df.index.get_loc(nearest_flip.name)
            
            # Safely get before/after gamma values
            if flip_index_pos > 0:
                gamma_before = exposure_df.iloc[flip_index_pos - 1]['net_gamma_exposure']
            else:
                gamma_before = 0
                
            if flip_index_pos < len(exposure_df) - 1:
                gamma_after = exposure_df.iloc[flip_index_pos + 1]['net_gamma_exposure']
            else:
                gamma_after = 0
            
            flip_results = {
                "ticker": self.ticker,
                "spot_price": self.current_price,
                "date": self.today.strftime("%Y-%m-%d"),
                "nearest_flip_strike": nearest_flip['strike'],
                "flip_direction": flip_direction,
                "distance_from_spot": nearest_flip['distance_from_spot'],
                "distance_percent": nearest_flip['distance_percent'],
                "gamma_before_flip": gamma_before,
                "gamma_after_flip": gamma_after,
            }
        except Exception as e:
            print(f"Error finding gamma flip points: {str(e)}")
            return None
        
        # Save results
        pd.DataFrame([flip_results]).to_csv(
            os.path.join(self.results_dir, self.ticker, f"{self.ticker}_gamma_flip_{self.today.strftime('%Y%m%d')}.csv"),
            index=False
        )
        
        return flip_results

    def plot_gamma_exposure(self):
        """Create visualization of gamma exposure profile."""
        # Calculate Greeks if not already done
        exposure_df = self.calculate_greeks()
        
        if exposure_df is None or exposure_df.empty:
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot call and put gamma exposure
        ax.bar(exposure_df['strike'], exposure_df['call_gamma_exposure'], color='green', alpha=0.5, label='Call Gamma Exposure')
        ax.bar(exposure_df['strike'], exposure_df['put_gamma_exposure'], color='red', alpha=0.5, label='Put Gamma Exposure')
        
        # Plot net gamma exposure line
        ax.plot(exposure_df['strike'], exposure_df['net_gamma_exposure'], 'k-', linewidth=2, label='Net Gamma Exposure')
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add current price line
        ax.axvline(x=self.current_price, color='blue', linestyle='-', label=f'Current Price: ${self.current_price:.2f}')
        
        # Add annotations for major gamma levels
        try:
            # Use .idxmax() with proper error handling
            if not exposure_df['call_gamma_exposure'].empty and exposure_df['call_gamma_exposure'].max() > 0:
                call_max_idx = exposure_df['call_gamma_exposure'].idxmax()
                call_max_row = exposure_df.loc[call_max_idx]
                
                ax.annotate(f"Call Gamma Peak\n${call_max_row['strike']:.2f}", 
                           xy=(call_max_row['strike'], call_max_row['call_gamma_exposure']),
                           xytext=(10, 20), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', lw=1.5))
            
            if not exposure_df['put_gamma_exposure'].empty and exposure_df['put_gamma_exposure'].min() < 0:
                put_max_idx = exposure_df['put_gamma_exposure'].idxmin()  # For puts, we want the most negative
                put_max_row = exposure_df.loc[put_max_idx]
                
                ax.annotate(f"Put Gamma Peak\n${put_max_row['strike']:.2f}", 
                           xy=(put_max_row['strike'], put_max_row['put_gamma_exposure']),
                           xytext=(-10, -20), textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', lw=1.5))
        except Exception as e:
            print(f"Warning in gamma exposure annotation: {str(e)}")
        
        # Detect and mark gamma flip points
        try:
            flip_points = exposure_df[exposure_df['net_gamma_exposure'].apply(np.sign).diff().fillna(0) != 0]
            
            if not flip_points.empty:
                for i, point in flip_points.iterrows():
                    ax.axvline(x=point['strike'], color='purple', linestyle='--', alpha=0.5)
                    ax.annotate(f"Gamma Flip\n${point['strike']:.2f}", 
                               xy=(point['strike'], 0),
                               xytext=(0, 30), textcoords='offset points',
                               arrowprops=dict(arrowstyle='->', lw=1.5),
                               ha='center')
        except Exception as e:
            print(f"Warning in gamma flip annotation: {str(e)}")
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Gamma Exposure')
        ax.set_title(f'{self.ticker} Gamma Exposure Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
        plt.savefig(os.path.join(self.charts_dir, self.ticker, f"{self.ticker}_gamma_profile_{self.today.strftime('%Y%m%d')}.png"), dpi=300)
        plt.close()

    def plot_delta_exposure(self):
        """Create visualization of delta exposure profile."""
        # Calculate Greeks if not already done
        exposure_df = self.calculate_greeks()
        
        if exposure_df is None or exposure_df.empty:
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot call and put delta exposure
        ax.bar(exposure_df['strike'], exposure_df['call_delta_exposure'], color='green', alpha=0.5, label='Call Delta Exposure')
        ax.bar(exposure_df['strike'], exposure_df['put_delta_exposure'], color='red', alpha=0.5, label='Put Delta Exposure')
        
        # Plot net delta exposure line
        ax.plot(exposure_df['strike'], exposure_df['net_delta_exposure'], 'k-', linewidth=2, label='Net Delta Exposure')
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add current price line
        ax.axvline(x=self.current_price, color='blue', linestyle='-', label=f'Current Price: ${self.current_price:.2f}')
        
        # Maximum delta exposure around current price
        try:
            window = (exposure_df['strike'] >= self.current_price * 0.9) & (exposure_df['strike'] <= self.current_price * 1.1)
            if window.any():
                window_df = exposure_df[window]
                if not window_df.empty:
                    # Find the max using .iloc instead of .idxmax()
                    max_idx = window_df['net_delta_exposure'].abs().idxmax()
                    max_row = window_df.loc[max_idx]
                    
                    ax.annotate(f"Major Delta Level\n${max_row['strike']:.2f}", 
                               xy=(max_row['strike'], max_row['net_delta_exposure']),
                               xytext=(10, 20), textcoords='offset points',
                               arrowprops=dict(arrowstyle='->', lw=1.5))
        except Exception as e:
            print(f"Warning in delta exposure plot: {str(e)}")
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Delta Exposure')
        ax.set_title(f'{self.ticker} Delta Exposure Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
        plt.savefig(os.path.join(self.charts_dir, self.ticker, f"{self.ticker}_delta_profile_{self.today.strftime('%Y%m%d')}.png"), dpi=300)
        plt.close()

    def plot_volatility_surface(self, plot_type='3d', plot_both=False):
        """
        Create a visualization of the implied volatility surface.
        
        Args:
            plot_type (str): Type of plot to create ('3d' or 'heatmap')
            plot_both (bool): Whether to plot both calls and puts
        """
        # Prepare data for visualization
        if plot_both:
            plot_data = [
                {"data": self.calls, "option_type": "Calls"},
                {"data": self.puts, "option_type": "Puts"}
            ]
        else:
            plot_data = [{"data": self.calls, "option_type": "Calls"}]
            
        for data in plot_data:
            df = data["data"]
            option_type = data["option_type"]
            
            # Convert implied volatility to percentage
            df['iv_percent'] = df['impliedVolatility'] * 100
            
            # Create timestamp for title
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if plot_type == '3d':
                # ----------------------------------------
                # 3D Surface plot with Plotly
                # ----------------------------------------
                
                # Prepare grid for surface plot
                # We'll use moneyness (strike/spot - 1) on x-axis and days to expiry on y-axis
                
                # Create meshgrid
                moneyness_range = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 100)
                days_range = np.linspace(df['daysToExpiry'].min(), df['daysToExpiry'].max(), 100)
                X, Y = np.meshgrid(moneyness_range, days_range)
                
                # Interpolate IV values onto grid
                points = np.column_stack((df['moneyness'], df['daysToExpiry']))
                values = df['iv_percent']
                
                # Grid data with linear interpolation
                Z = griddata(points, values, (X, Y), method='linear')
                
                # For missing values, try nearest interpolation
                mask = np.isnan(Z)
                if mask.any():
                    Z[mask] = griddata(points, values, (X[mask], Y[mask]), method='nearest')
                
                # Create visualization
                fig = go.Figure(data=[go.Surface(
                    x=X * 100,  # Convert moneyness to percentage
                    y=Y,  # Days to expiry
                    z=Z,  # Implied volatility (%)
                    colorscale='Viridis',
                    colorbar=dict(title="IV (%)"),
                    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.6, fresnel=0.8),
                    contours={
                        "x": {"show": True, "color":"rgba(0,0,0,0.3)"},
                        "y": {"show": True, "color":"rgba(0,0,0,0.3)"},
                        "z": {"show": True, "color":"rgba(0,0,0,0.3)"}
                    }
                )])
                
                # Add term structure and skew curves similar to reference image
                # Term structure (line at constant moneyness across expiries)
                atm_moneyness = 0.0  # At-the-money
                nearest_idx = np.abs(moneyness_range - atm_moneyness).argmin()
                
                fig.add_trace(go.Scatter3d(
                    x=np.ones(len(days_range)) * moneyness_range[nearest_idx] * 100,
                    y=days_range,
                    z=Z[:, nearest_idx],
                    mode='lines',
                    line=dict(color='white', width=5),
                    name='Term Structure'
                ))
                
                # Skew curve (line at constant expiry across strikes)
                nearest_expiry_idx = np.abs(days_range - df['daysToExpiry'].min()).argmin()
                
                fig.add_trace(go.Scatter3d(
                    x=moneyness_range * 100,
                    y=np.ones(len(moneyness_range)) * days_range[nearest_expiry_idx],
                    z=Z[nearest_expiry_idx, :],
                    mode='lines',
                    line=dict(color='white', width=5),
                    name='Volatility Skew'
                ))
                
                # Add labels for key areas on the surface
                fig.add_trace(go.Scatter3d(
                    x=[0],
                    y=[days_range.mean()],
                    z=[Z[np.abs(days_range - days_range.mean()).argmin(), nearest_idx] + 10],
                    mode='text',
                    text=['Term Structure'],
                    textfont=dict(color='white', size=12),
                    name='',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[moneyness_range.mean() * 100],
                    y=[days_range[nearest_expiry_idx]],
                    z=[Z[nearest_expiry_idx, np.abs(moneyness_range - moneyness_range.mean()).argmin()] + 10],
                    mode='text',
                    text=['Skew Curve'],
                    textfont=dict(color='white', size=12),
                    name='',
                    showlegend=False
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'Volatility Surface for {self.ticker} {option_type}<br>Timestamp: {timestamp}',
                    scene=dict(
                        xaxis_title='Moneyness (%)',
                        yaxis_title='Expiry (days)',
                        zaxis_title='Implied Vol (%)',
                        xaxis=dict(tickformat='.0f', showbackground=False, gridcolor='rgba(255,255,255,0.2)'),
                        yaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.2)'),
                        zaxis=dict(tickformat='.0f', showbackground=False, gridcolor='rgba(255,255,255,0.2)')
                    ),
                    template='plotly_dark',
                    margin=dict(l=0, r=0, t=100, b=0),
                    autosize=True,
                    width=1000,
                    height=800,
                )
                
                # Add dark background
                fig.update_layout(
                    paper_bgcolor='black',
                    plot_bgcolor='black',
                    font=dict(color='white')
                )
                
                # Save the figure
                os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
                pio.write_image(fig, os.path.join(self.charts_dir, self.ticker, 
                                                f"{self.ticker}_{option_type.lower()}_vol_surface_{self.today.strftime('%Y%m%d')}.png"),
                                scale=2)
                pio.write_html(fig, os.path.join(self.charts_dir, self.ticker, 
                                               f"{self.ticker}_{option_type.lower()}_vol_surface_{self.today.strftime('%Y%m%d')}.html"))
            
            elif plot_type == 'heatmap':
                # ----------------------------------------
                # Heatmap visualization
                # ----------------------------------------
                
                # Pivot data to create strike x expiry grid
                pivot = df.pivot_table(index='strike', columns='daysToExpiry', values='iv_percent')
                
                # Create a heatmap
                plt.figure(figsize=(12, 8))
                
                cmap = cm.get_cmap('viridis')
                sns_plot = sns.heatmap(pivot, cmap=cmap, annot=False, fmt=".1f", 
                                       cbar_kws={'label': 'Implied Volatility (%)'})
                
                plt.title(f'{self.ticker} {option_type} Implied Volatility Surface\nTimestamp: {timestamp}')
                plt.xlabel('Days to Expiry')
                plt.ylabel('Strike Price')
                
                # Mark current price
                current_price_idx = np.abs(np.array(pivot.index) - self.current_price).argmin()
                plt.axhline(y=current_price_idx, color='red', linestyle='--', alpha=0.7, 
                            label=f'Current Price: ${self.current_price:.2f}')
                
                plt.legend()
                plt.tight_layout()
                
                # Save the figure
                os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
                plt.savefig(os.path.join(self.charts_dir, self.ticker, 
                                        f"{self.ticker}_{option_type.lower()}_vol_heatmap_{self.today.strftime('%Y%m%d')}.png"), 
                           dpi=300)
                plt.close()

    def identify_support_resistance(self):
        """
        Identify support and resistance levels from options open interest.
        
        Returns:
            dict: Support and resistance levels
        """
        # For resistance: look for high call open interest above current price
        calls_above = self.calls[self.calls['strike'] > self.current_price]
        puts_below = self.puts[self.puts['strike'] < self.current_price]
        
        if calls_above.empty or puts_below.empty:
            print("Not enough data to identify support/resistance levels")
            return None
        
        # Group by strike and sum open interest
        call_resistance = calls_above.groupby('strike')['openInterest'].sum().reset_index()
        put_support = puts_below.groupby('strike')['openInterest'].sum().reset_index()
        
        # Sort by open interest to find highest levels
        call_resistance = call_resistance.sort_values('openInterest', ascending=False)
        put_support = put_support.sort_values('openInterest', ascending=False)
        
        # Get top 3 resistance and support levels
        top_resistance = call_resistance.head(3)
        top_support = put_support.head(3)
        
        # Filter to ensure levels are significant (have minimum open interest)
        min_oi_threshold = max(100, self.calls['openInterest'].mean() * 2)
        
        significant_resistance = top_resistance[top_resistance['openInterest'] > min_oi_threshold]
        significant_support = top_support[top_support['openInterest'] > min_oi_threshold]
        
        # Calculate distance from current price
        if not significant_resistance.empty:
            significant_resistance['distance'] = significant_resistance['strike'] - self.current_price
            significant_resistance['distance_percent'] = (significant_resistance['distance'] / self.current_price) * 100
        
        if not significant_support.empty:
            significant_support['distance'] = self.current_price - significant_support['strike']
            significant_support['distance_percent'] = (significant_support['distance'] / self.current_price) * 100
        
        # Prepare results
        results = {
            "ticker": self.ticker,
            "spot_price": self.current_price,
            "date": self.today.strftime("%Y-%m-%d"),
            "resistance_levels": significant_resistance[['strike', 'openInterest', 'distance', 'distance_percent']].to_dict('records') if not significant_resistance.empty else [],
            "support_levels": significant_support[['strike', 'openInterest', 'distance', 'distance_percent']].to_dict('records') if not significant_support.empty else []
        }
        
        # Save results
        pd.DataFrame([results]).to_csv(
            os.path.join(self.results_dir, self.ticker, f"{self.ticker}_support_resistance_{self.today.strftime('%Y%m%d')}.csv"),
            index=False
        )
        
        return results

    def plot_support_resistance(self):
        """Plot support and resistance levels from options open interest."""
        # Get support and resistance data
        levels = self.identify_support_resistance()
        
        if not levels:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot open interest by strike for both calls and puts
        call_oi = self.calls.groupby('strike')['openInterest'].sum().reset_index()
        put_oi = self.puts.groupby('strike')['openInterest'].sum().reset_index()
        
        ax.bar(call_oi['strike'], call_oi['openInterest'], color='green', alpha=0.5, label='Call Open Interest')
        ax.bar(put_oi['strike'], put_oi['openInterest'], color='red', alpha=0.5, label='Put Open Interest')
        
        # Add current price line
        ax.axvline(x=self.current_price, color='blue', linestyle='-', label=f'Current Price: ${self.current_price:.2f}')
        
        # Add resistance levels
        for level in levels["resistance_levels"]:
            ax.axvline(x=level['strike'], color='red', linestyle='--', alpha=0.7)
            ax.annotate(f"Resistance: ${level['strike']:.2f}\nOI: {level['openInterest']}", 
                       xy=(level['strike'], 0),
                       xytext=(0, 30), textcoords='offset points',
                       ha='center', va='bottom', 
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        # Add support levels
        for level in levels["support_levels"]:
            ax.axvline(x=level['strike'], color='green', linestyle='--', alpha=0.7)
            ax.annotate(f"Support: ${level['strike']:.2f}\nOI: {level['openInterest']}", 
                       xy=(level['strike'], 0),
                       xytext=(0, 60), textcoords='offset points',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Open Interest')
        ax.set_title(f'{self.ticker} Support and Resistance Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits around current price
        ax.set_xlim(self.current_price * 0.7, self.current_price * 1.3)
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
        plt.savefig(os.path.join(self.charts_dir, self.ticker, f"{self.ticker}_support_resistance_{self.today.strftime('%Y%m%d')}.png"), dpi=300)
        plt.close()

    def create_multi_expiry_gex_plot(self):
        """
        Create a multi-panel visualization of GEX (Gamma Exposure) for multiple expirations.
        Based on the Gnovdm-WwAAoTlm.jpeg reference image.
        """
        # Only attempt this if we have sufficient data
        if len(self.expirations) < 4:
            print("Not enough expirations for multi-expiry GEX plot")
            return
        
        # Calculate Greeks for all data
        self.calculate_greeks()
        
        # Select 4 key expirations for visualization
        # 1. Nearest expiration
        # 2. Next expiration
        # 3. Expiration with highest GEX
        # 4. Second highest GEX
        
        # Calculate total gamma exposure by expiration
        exp_gex = {}
        for exp in self.expirations:
            calls_exp = self.calls[self.calls['expirationDate'] == exp]
            puts_exp = self.puts[self.puts['expirationDate'] == exp]
            
            if calls_exp.empty or puts_exp.empty:
                continue
                
            total_gamma = (calls_exp['gamma_exposure'].sum() + puts_exp['gamma_exposure'].sum())
            exp_gex[exp] = abs(total_gamma)  # Use absolute value for ranking
        
        if len(exp_gex) < 4:
            print("Not enough valid expirations for multi-expiry GEX plot")
            return
        
        # Sort expirations by date
        sorted_exp = sorted(exp_gex.keys())
        
        # Get nearest two expirations
        nearest_exp = sorted_exp[0]
        next_exp = sorted_exp[1]
        
        # Get highest GEX expirations (excluding the first two)
        remaining_exp = sorted_exp[2:]
        sorted_by_gex = sorted([(exp, exp_gex[exp]) for exp in remaining_exp], key=lambda x: x[1], reverse=True)
        
        highest_gex_exp = sorted_by_gex[0][0] if len(sorted_by_gex) > 0 else None
        second_highest_gex_exp = sorted_by_gex[1][0] if len(sorted_by_gex) > 1 else None
        
        # Selected expirations for plotting
        selected_exp = [nearest_exp, next_exp]
        
        if highest_gex_exp:
            selected_exp.append(highest_gex_exp)
        
        if second_highest_gex_exp:
            selected_exp.append(second_highest_gex_exp)
        
        # Create multi-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Colors for consistent visualization
        colors = {
            'gex_profile': 'gold',
            'positive_gex': 'green',
            'negative_gex': 'red',
            'spot_price': 'white',
            'call_resistance': 'orangered',
            'put_support': 'limegreen',
            'hvl': 'yellow'  # Highest volume level
        }
        
        # Process each selected expiration
        for i, exp in enumerate(selected_exp):
            if i >= 4:  # Only use the first 4
                break
                
            # Get data for this expiration
            calls_exp = self.calls[self.calls['expirationDate'] == exp]
            puts_exp = self.puts[self.puts['expirationDate'] == exp]
            
            if calls_exp.empty or puts_exp.empty:
                continue
            
            # Calculate days to expiry
            exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
            days_to_exp = (exp_date - self.today).days
            
            # Group by strike
            call_by_strike = calls_exp.groupby('strike')[['gamma_exposure', 'openInterest', 'volume']].sum()
            put_by_strike = puts_exp.groupby('strike')[['gamma_exposure', 'openInterest', 'volume']].sum()
            
            # Combine and calculate net gamma
            all_strikes = sorted(set(call_by_strike.index).union(set(put_by_strike.index)))
            
            # Create a DataFrame with all strikes
            gex_data = pd.DataFrame(index=all_strikes)
            gex_data['call_gamma'] = call_by_strike['gamma_exposure'].reindex(all_strikes, fill_value=0)
            gex_data['put_gamma'] = put_by_strike['gamma_exposure'].reindex(all_strikes, fill_value=0)
            gex_data['net_gamma'] = gex_data['call_gamma'] + gex_data['put_gamma']
            
            # Calculate positive and negative GEX for visualization
            gex_data['positive_gex'] = gex_data['net_gamma'].apply(lambda x: max(0, x))
            gex_data['negative_gex'] = gex_data['net_gamma'].apply(lambda x: min(0, x))
            
            # Find resistance, support, and highest volume levels
            call_oi = calls_exp.groupby('strike')['openInterest'].sum()
            put_oi = puts_exp.groupby('strike')['openInterest'].sum()
            
            # Find highest call OI above current price
            calls_above = call_oi[call_oi.index > self.current_price]
            if not calls_above.empty:
                call_resistance = calls_above.idxmax()
            else:
                call_resistance = None
                
            # Find highest put OI below current price
            puts_below = put_oi[put_oi.index < self.current_price]
            if not puts_below.empty:
                put_support = puts_below.idxmax()
            else:
                put_support = None
                
            # Find highest volume level
            total_volume = (calls_exp.groupby('strike')['volume'].sum() + 
                           puts_exp.groupby('strike')['volume'].sum())
            if not total_volume.empty:
                hvl = total_volume.idxmax()
            else:
                hvl = None
            
            # Plotting
            ax = axes[i]
            
            # Plot positive and negative GEX as green/red bars
            ax.bar(gex_data.index, gex_data['positive_gex'], color=colors['positive_gex'], alpha=0.7, label='Positive GEX')
            ax.bar(gex_data.index, gex_data['negative_gex'], color=colors['negative_gex'], alpha=0.7, label='Negative GEX')
            
            # Plot GEX profile curve
            sorted_index = sorted(gex_data.index)
            ax.plot(sorted_index, gex_data.loc[sorted_index, 'net_gamma'], color=colors['gex_profile'], linewidth=2, label='GEX Profile')
            
            # Add vertical lines for key levels
            ax.axvline(x=self.current_price, color=colors['spot_price'], linestyle='--', label=f'Spot: {self.current_price:.2f}')
            
            if call_resistance:
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=gex_data['net_gamma'].max() * 0.7, color='gray', linestyle='-', alpha=0.3)
                ax.axhline(y=gex_data['net_gamma'].min() * 0.7, color='gray', linestyle='-', alpha=0.3)
                
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.axvline(x=call_resistance, color=colors['call_resistance'], linestyle='--', 
                          label=f'Call Resistance: {call_resistance:.2f}')
            
            if put_support:
                ax.axvline(x=put_support, color=colors['put_support'], linestyle='--', 
                          label=f'Put Support: {put_support:.2f}')
            
            if hvl:
                ax.axvline(x=hvl, color=colors['hvl'], linestyle=':', 
                          label=f'HVL: {hvl:.2f}')
            
            # Set title and styling for each panel
            gex_sum = gex_data['net_gamma'].sum()
            title_parts = []
            
            if i == 0:
                title_parts.append("First Expiration")
            elif i == 1:
                title_parts.append("Next Expiration")
            elif i == 2:
                title_parts.append("Expiration with\nHighest GEX")
            elif i == 3:
                title_parts.append("Expiration with\n2nd Highest GEX")
            
            title_parts.append(f"{exp}")
            title_parts.append(f"GEX Expiring: {gex_sum:.2f}%")
            
            ax.set_title("\n".join(title_parts))
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('GEX')
            
            # Set limits focused around current price
            price_range = max(50, self.current_price * 0.2)  # At least 50 points or 20% of current price
            ax.set_xlim(self.current_price - price_range, self.current_price + price_range)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper right', fontsize='small')
        
        # Overall title
        plt.suptitle(f'NET GEX Multi Expirations for {self.ticker}\nTimestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Style for dark background to match reference
        fig.set_facecolor('black')
        for ax in axes:
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Save the figure
        os.makedirs(os.path.join(self.charts_dir, self.ticker), exist_ok=True)
        plt.savefig(
            os.path.join(self.charts_dir, self.ticker, f"{self.ticker}_multi_expiry_gex_{self.today.strftime('%Y%m%d')}.png"), 
            dpi=300, facecolor='black'
        )
        plt.close()

    def run_full_analysis(self):
        """Run a complete options analysis and generate all visualizations and data."""
        print(f"Starting full options analysis for {self.ticker}...")
        
        try:
            # 1. Load options data (already done in __init__)
            
            # 2. Calculate Put/Call ratio
            pcr_results = self.analyze_put_call_ratio()
            print(f"Put/Call Ratio Analysis: Volume PCR = {pcr_results['pcr_volume']:.2f}, OI PCR = {pcr_results['pcr_open_interest']:.2f}")
            
            # 3. Visualize Put/Call ratio
            self.plot_put_call_ratio()
            print("Generated Put/Call ratio visualization")
            
            # 4. Calculate Greeks and exposures
            _ = self.calculate_greeks()
            print("Calculated option Greeks and exposures")
            
            # 5. Detect gamma flip points
            gamma_flip_results = self.detect_gamma_flip()
            if gamma_flip_results:
                print(f"Detected gamma flip at strike {gamma_flip_results['nearest_flip_strike']:.2f} ({gamma_flip_results['flip_direction']})")
            
            # 6. Visualize gamma exposure
            try:
                self.plot_gamma_exposure()
                print("Generated gamma exposure visualization")
            except Exception as e:
                print(f"Warning: Could not generate gamma exposure plot: {str(e)}")
            
            # 7. Visualize delta exposure
            try:
                self.plot_delta_exposure()
                print("Generated delta exposure visualization")
            except Exception as e:
                print(f"Warning: Could not generate delta exposure plot: {str(e)}")
            
            # 8. Generate volatility surface
            try:
                self.plot_volatility_surface(plot_type='3d', plot_both=True)
                print("Generated 3D volatility surface visualizations")
            except Exception as e:
                print(f"Warning: Could not generate volatility surface: {str(e)}")
            
            # 9. Identify and visualize support/resistance levels
            try:
                sr_levels = self.identify_support_resistance()
                if sr_levels:
                    resistance_count = len(sr_levels['resistance_levels'])
                    support_count = len(sr_levels['support_levels'])
                    print(f"Identified {resistance_count} resistance and {support_count} support levels")
                
                self.plot_support_resistance()
                print("Generated support/resistance visualization")
            except Exception as e:
                print(f"Warning: Could not generate support/resistance analysis: {str(e)}")
            
            # 10. Create multi-expiry GEX plot
            try:
                self.create_multi_expiry_gex_plot()
                print("Generated multi-expiration GEX visualization")
            except Exception as e:
                print(f"Warning: Could not generate multi-expiry GEX plot: {str(e)}")
            
            print(f"Full analysis for {self.ticker} completed. Results saved to:")
            print(f"  - Data: {self.data_dir}/{self.ticker}/")
            print(f"  - Results: {self.results_dir}/{self.ticker}/")
            print(f"  - Charts: {self.charts_dir}/{self.ticker}/")
            
        except Exception as e:
            print(f"Error in full analysis for {self.ticker}: {str(e)}")
            print("Partial results may have been saved.")


# Example usage
if __name__ == "__main__":
    # Create a sample settings file
    import json
    
    sample_config = {
        "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
        "days_to_expiry": 90,
        "output_directory": None  # Use default directories
    }
    
    with open('options_analysis_config.json', 'w') as f:
        json.dump(sample_config, f, indent=4)
    
    print("Created sample configuration file: options_analysis_config.json")
    
    # Create README
    readme_content = """# Options Analysis Module

This module implements options analysis techniques based on financial theory, including:

## Features

- **Volatility Surface Analysis**: Create 3D visualizations of implied volatility surfaces
- **Put/Call Ratio Analysis**: Calculate and visualize put/call ratios for sentiment analysis
- **Greek Exposure**: Calculate and visualize delta and gamma exposure
- **Gamma Flip Detection**: Identify points where gamma exposure changes sign
- **Support/Resistance**: Identify option-derived support and resistance levels
- **Multi-Expiration GEX**: Visualize gamma exposure across multiple expirations

## Usage

```python
from options_analysis import OptionsAnalyzer

# Initialize analyzer for a ticker
analyzer = OptionsAnalyzer("AAPL", days_to_expiry=60)

# Run full analysis
analyzer.run_full_analysis()

# Or run individual analyses
pcr = analyzer.analyze_put_call_ratio()
analyzer.plot_volatility_surface()
analyzer.plot_gamma_exposure()
```

## Documentation

### Data Storage

The module automatically organizes outputs into the following directories:

- `data/`: Raw options data by ticker
- `results/`: Analysis results in CSV format
- `charts/`: Visualizations in PNG and HTML formats

### Configuration

Create a configuration file (`options_analysis_config.json`) with the following structure:

```json
{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "days_to_expiry": 90,
    "output_directory": null
}
```

### Theory References

This implementation is based on theoretical concepts from financial literature:

1. **Put/Call Ratio**: A sentiment indicator measuring market positioning
2. **Gamma Exposure**: Measures second-order price sensitivity in options
3. **Volatility Surface**: 3D representation of implied volatility by strike and expiry
4. **Support/Resistance Levels**: Price levels with significant option open interest

## Requirements

- numpy
- pandas
- matplotlib
- plotly
- yfinance
- scipy
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created README.md")
    
    # Run a sample analysis if requested
    run_sample = input("Run a sample analysis for AAPL? (y/n): ")
    if run_sample.lower() == 'y':
        print("\nRunning analysis for AAPL...")
        analyzer = OptionsAnalyzer("AAPL", days_to_expiry=60)
        analyzer.run_full_analysis()