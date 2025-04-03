import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = yf.Ticker(ticker).history(period='1y')
        self.buy_signals = []
        self.sell_signals = []  # Hinzugefügt

    def generate_daily_signals(self):
        """Analysiere für jeden Tag, ob ein Kaufsignal vorliegt"""
        self.buy_signals = []
        buy_condition_active = False

        for i in range(50, len(self.data)):
            window_data = self.data.iloc[:i+1].copy()
            self.data_slice = window_data  # temporär überschreiben
            current_price = float(window_data['Close'].iloc[-1])
            sma20_today = float(window_data['Close'].rolling(window=20).mean().iloc[-1])
            sma50_today = float(window_data['Close'].rolling(window=50).mean().iloc[-1])
            
            if i >= 51:
                current_price_prev = float(window_data['Close'].iloc[-2])
                sma20_prev = float(window_data['Close'].rolling(window=20).mean().iloc[-2])
                sma50_prev = float(window_data['Close'].rolling(window=50).mean().iloc[-2])
            else:
                current_price_prev = current_price
                sma20_prev = sma20_today
                sma50_prev = sma50_today

            condition = (
                current_price > sma20_today and sma20_today > sma50_today and
                current_price > current_price_prev and
                sma20_today > sma20_prev and
                sma50_today > sma50_prev
            )

            if condition and not buy_condition_active:
                self.buy_signals.append(window_data.index[-1])
                print(f"Kaufsignal erkannt am {window_data.index[-1].date()}")
                buy_condition_active = True
            elif not condition and buy_condition_active:
                buy_condition_active = False

        self.data_slice = None

    def generate_daily_sell_signals(self):
        """Analysiere für jeden Tag, ob ein Verkaufssignal vorliegt"""
        self.sell_signals = []
        sell_condition_active = False

        for i in range(50, len(self.data)):
            window_data = self.data.iloc[:i+1].copy()
            self.data_slice = window_data  # temporär überschreiben
            current_price = float(window_data['Close'].iloc[-1])
            sma20_today = float(window_data['Close'].rolling(window=20).mean().iloc[-1])
            sma50_today = float(window_data['Close'].rolling(window=50).mean().iloc[-1])
            
            if i >= 51:
                current_price_prev = float(window_data['Close'].iloc[-2])
                sma20_prev = float(window_data['Close'].rolling(window=20).mean().iloc[-2])
                sma50_prev = float(window_data['Close'].rolling(window=50).mean().iloc[-2])
            else:
                current_price_prev = current_price
                sma20_prev = sma20_today
                sma50_prev = sma50_today

            condition = (
                current_price < sma20_today and sma20_today < sma50_today and
                current_price < current_price_prev and
                sma20_today < sma20_prev and
                sma50_today < sma50_prev
            )

            if condition and not sell_condition_active:
                self.sell_signals.append(window_data.index[-1])
                print(f"Verkaufssignal erkannt am {window_data.index[-1].date()}")
                sell_condition_active = True
            elif not condition and sell_condition_active:
                sell_condition_active = False

        self.data_slice = None

    def calculate_strategy_return(self):
        """Berechne die Rendite basierend auf Buy/Sell-Signalen"""
        if not self.buy_signals or not self.sell_signals:
            print("Nicht genug Signale zur Berechnung der Rendite.")
            return

        trades = []
        buy_iter = iter(self.buy_signals)
        sell_iter = iter(self.sell_signals)
        current_buy = next(buy_iter, None)
        current_sell = next(sell_iter, None)

        while current_buy and current_sell:
            if current_sell > current_buy:
                buy_price = self.data.loc[current_buy]['Close']
                sell_price = self.data.loc[current_sell]['Close']
                trades.append((buy_price, sell_price))
                current_buy = next(buy_iter, None)
                current_sell = next(sell_iter, None)
            else:
                current_sell = next(sell_iter, None)

        if trades:
            returns = [(sell - buy) / buy for buy, sell in trades]
            total_return = np.prod([1 + r for r in returns]) - 1
            print(f"Gesamtrendite über alle Trades: {total_return * 100:.2f}%")
        else:
            print("Keine gültigen Trade-Paare gefunden.")

    def plot_analysis(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label='Schlusskurs')
        
        # Zeichne rote senkrechte Linien für Verkaufssignale
        if hasattr(self, "sell_signals"):
            if len(self.sell_signals) == 0:
                print("Keine Verkaufssignale zum Plotten vorhanden.")
            for i, signal_date in enumerate(self.sell_signals):
                plt.axvline(x=signal_date, color='red', linestyle='--', alpha=0.6,
                            label='Verkaufssignal' if i == 0 else None)

        # Zeichne grüne senkrechte Linien für Kaufsignale
        if hasattr(self, "buy_signals"):
            if len(self.buy_signals) == 0:
                print("Keine Kaufsignale zum Plotten vorhanden.")
            for i, signal_date in enumerate(self.buy_signals):
                plt.axvline(x=signal_date, color='green', linestyle='--', alpha=0.6,
                            label='Kaufsignal' if i == 0 else None)
        
        plt.title(f'Aktienanalyse für {self.ticker}')
        plt.xlabel('Datum')
        plt.ylabel('Preis (USD)')
        plt.legend()
        plt.grid()
        plt.show()

# Hauptfunktion zur Analyse von Microsoft
def analyze_microsoft():
    analyzer = StockAnalyzer("MSFT")
    analyzer.generate_daily_signals()
    analyzer.generate_daily_sell_signals()  # Hinzugefügt
    analyzer.calculate_strategy_return()
    analyzer.plot_analysis()

analyze_microsoft()