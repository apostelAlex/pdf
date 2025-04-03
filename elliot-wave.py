import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class AnalysisParameters:
    ticker: str = "MSFT"
    period: str = "1y"
    sma_short: int = 20
    sma_long: int = 50
    extrema_window: int = 10
    fibonacci_levels: list = field(default_factory=lambda: [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618])

class ElliottWaveAnalyzer:

    def __init__(self, params: AnalysisParameters = None):
        self.params = params or AnalysisParameters()
        self.ticker = self.params.ticker
        self.period = self.params.period
        self.data = None
        self.wave_patterns = None
        self.fibonacci_levels = self.params.fibonacci_levels
        self.buy_signals = []
        self.sell_signals = []

    def fetch_data(self):
        """Lade Microsoft-Aktiendaten herunter"""
        print(f"Lade Daten für {self.ticker}...")
        self.data = yf.download(self.ticker, period=self.period)
        return self.data
    
    def identify_local_extrema(self, window=10):
        """Identifiziere lokale Maxima und Minima als potenzielle Wellenpunkte"""
        prices = self.data['Close'].values  # Convert to numpy array
        dates = self.data.index
        peaks = []
        troughs = []
        print(self.data['Close'])
        
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window+1)):
                peaks.append((i, prices[i]))
            
            if all(prices[i] < prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, window+1)):
                troughs.append((i, prices[i]))
        
        return peaks, troughs
    
    def detect_elliott_waves(self):
        """Versuche, Elliott-Wellenmuster in den Daten zu erkennen"""
        peaks, troughs = self.identify_local_extrema()
        extrema = sorted(peaks + troughs, key=lambda x: x[0])
        
        # Einfache Implementierung zur Identifikation potenzieller 5-3 Elliott-Wellen-Muster
        patterns = []
        for i in range(len(extrema) - 7):
            potential_pattern = extrema[i:i+8]
            
            # Prüfe, ob ein potenzielles Impuls-Korrektur-Muster vorhanden ist
            # (Vereinfachte Version, echte Elliott-Wellen-Analyse ist komplexer)
            if self._is_potential_elliott_pattern(potential_pattern):
                patterns.append(potential_pattern)
        
        self.wave_patterns = patterns
        return patterns
    
    def _is_potential_elliott_pattern(self, pattern):
        """Überprüfe, ob die Punkte einem potenziellen Elliott-Wellenmuster entsprechen können"""
        # Vereinfachte Bedingungen für ein 5-3-Muster
        # In einer echten Analyse wären hier komplexere Regeln
        
        # Wellen 1, 3, 5 sollten aufwärts gehen (für Bullenmarkt)
        # Wellen 2, 4, A, C sollten abwärts gehen
        # Welle B sollte aufwärts gehen
        
        values = [p[1] for p in pattern]
        
        # Prüfe grundlegende Muster für Impulswellen
        impulsive = (values[0] < values[2] > values[1] < values[4] > values[3] < values[6])
        
        # Prüfe grundlegende Muster für Korrekturwellen
        corrective = (values[4] > values[6] < values[5] > values[7])
        
        # Prüfe Fibonacci-Verhältnisse (vereinfacht)
        fib_check = self._check_fibonacci_relations(values)
        
        return impulsive and corrective and fib_check
    
    def _check_fibonacci_relations(self, values):
        """Überprüfe, ob die Wellen Fibonacci-Verhältnisse einhalten"""
        # Vereinfachte Version
        # Beispiel: Welle 2 sollte eine 0.5-0.618 Retracements von Welle 1 sein
        wave1 = abs(values[2] - values[0])
        wave2_retrace = abs(values[1] - values[2]) / wave1
        
        # Welle 3 sollte 1.618 * Welle 1 sein
        wave3 = abs(values[4] - values[1])
        wave3_relation = wave3 / wave1
        
        # Einfache Prüfung einiger Fibonacci-Verhältnisse
        return (0.5 <= wave2_retrace <= 0.618) and (1.5 <= wave3_relation <= 1.8)
    
    def calculate_fibonacci_levels(self, start_price, end_price, trend="up"):
        """Berechne Fibonacci-Retracement-Levels"""
        levels = {}
        
        if trend == "up":
            diff = end_price - start_price
            for fib in self.fibonacci_levels:
                levels[fib] = end_price - diff * fib
        else:
            diff = start_price - end_price
            for fib in self.fibonacci_levels:
                levels[fib] = end_price + diff * fib
                
        return levels
    
    def analyze_current_position(self):
        """Analysiere die aktuelle Position im Elliott-Wellen-Zyklus"""
        if not self.wave_patterns:
            self.detect_elliott_waves()
            
        if not self.wave_patterns:
            return "Keine klaren Elliott-Wellenmuster gefunden"
        
        # Wähle das neueste erkannte Muster
        latest_pattern = self.wave_patterns[-1]
        latest_points = [p[0] for p in latest_pattern]
        
        # Prüfe, wo wir im aktuellen Zyklus stehen
        latest_point = latest_points[-1]
        current_idx = len(self.data) - 1
        
        # Bestimme die aktuelle Position basierend auf dem letzten erkannten Muster
        position_in_cycle = None
        if current_idx - latest_point < 10:  # Wenn das letzte Muster kürzlich endete
            # Schätze, ob wir in einem neuen Zyklus sind oder nicht
            if self.data['Close'].iloc[-5:].mean() > self.data['Close'].iloc[-10:-5].mean():
                position_in_cycle = "Möglicherweise Beginn neuer Impuls (Welle 1)"
            else:
                position_in_cycle = "Möglicherweise in Korrekturwelle (A-B-C)"
        else:
            # Wenn das letzte erkannte Muster älter ist, analysiere den neueren Trend
            short_trend = self.data['Close'].iloc[-10:].pct_change().mean()
            if short_trend > 0:
                position_in_cycle = "Aufwärtstrend, potenziell in Impulswelle"
            else:
                position_in_cycle = "Abwärtstrend, potenziell in Korrekturwelle"
        
        return position_in_cycle
    
    def make_buy_recommendation(self):
        """Generiere eine Kaufempfehlung basierend auf der Elliott-Wellen-Analyse"""
        # Extract scalar values from pandas Series
        current_price = float(self.data['Close'].iloc[-1])
        position = self.analyze_current_position()
        
        # Empfehlung basierend auf der Position im Elliott-Wellen-Zyklus
        if position and ("Beginn neuer Impuls" in position or "in Impulswelle" in position):
            if "Welle 1" in position:
                return f"KAUFEN: Potenzieller Beginn einer neuen Impulsphase bei {current_price}. Guter Einstiegspunkt."
            elif "Welle 3" in position:
                return f"STARK KAUFEN: Möglicherweise in Welle 3 bei {current_price}. Bester Zeitpunkt für Kauf."
            else:
                return f"HALTEN/KAUFEN: In Impulsphase bei {current_price}. Positive Aussichten."
        elif position and "Korrekturwelle" in position:
            if "C" in position:
                return f"ABWARTEN: Ende der Korrekturwelle C bei {current_price} abwarten. Kaufgelegenheit könnte bevorstehen."
            else:
                return f"NICHT KAUFEN: In Korrekturphase bei {current_price}. Besseren Einstiegspunkt abwarten."
        else:
            # Wenn keine klare Elliott-Wellenmuster erkannt wurden
            # Benutze einfache technische Indikatoren als Fallback
            sma20 = float(self.data['Close'].rolling(window=20).mean().iloc[-1])
            sma50 = float(self.data['Close'].rolling(window=50).mean().iloc[-1])
            
            if current_price > sma20 and sma20 > sma50:
                return f"KAUFEN: Positive Trendindikatoren bei {current_price}, obwohl kein klares Elliott-Wellenmuster erkannt wurde."
            elif current_price < sma20 and sma20 < sma50:
                return f"NICHT KAUFEN: Negative Trendindikatoren bei {current_price}. Warten auf Trendumkehr."
            else:
                return f"NEUTRAL: Gemischte Signale bei {current_price}. Weitere Analysen empfohlen."

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
            
            # Hole Vortageswerte (sofern verfügbar)
            if i >= 51:
                current_price_prev = float(window_data['Close'].iloc[-2])
                sma20_prev = float(window_data['Close'].rolling(window=20).mean().iloc[-2])
                sma50_prev = float(window_data['Close'].rolling(window=50).mean().iloc[-2])
            else:
                current_price_prev = current_price
                sma20_prev = sma20_today
                sma50_prev = sma50_today

            # Bedingung: aktueller Preis über SMA20 und SMA20 über SMA50
            # und zusätzlich steigen alle drei Werte im Vergleich zum Vortag
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

        self.data_slice = None  # Reset
    def generate_daily_sell_signals(self):
        self.sell_signals = []
        sell_condition_active = False

        for i in range(50, len(self.data)):
            window_data = self.data.iloc[:i+1].copy()
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
    
    def plot_analysis(self):
        """Plotte die Aktiendaten mit identifizierten Elliott-Wellen"""
        if self.wave_patterns is None or len(self.wave_patterns) == 0:
            print("Keine Wellenmuster zum Anzeigen gefunden")
            # Still plot the price chart even if no patterns are found
            plt.figure(figsize=(14, 8))
            plt.plot(self.data.index, self.data['Close'], label=f'{self.ticker} Aktienkurs')
            plt.title(f'{self.ticker} Kurs ohne erkannte Elliott-Wellen')
            plt.xlabel('Datum')
            plt.ylabel('Preis ($)')
            plt.grid(True, alpha=0.3)
            
            # Add SMA lines for reference
            sma20 = self.data['Close'].rolling(window=self.params.sma_short).mean()
            sma50 = self.data['Close'].rolling(window=self.params.sma_long).mean()
            plt.plot(self.data.index, sma20, 'g--', label='SMA 20')
            plt.plot(self.data.index, sma50, 'r--', label='SMA 50')
            
            # Zeichne grüne senkrechte Linien für Kaufsignale
            if hasattr(self, "buy_signals"):
                if len(self.buy_signals) == 0:
                    print("Keine Kaufsignale zum Plotten vorhanden.")
                for i, signal_date in enumerate(self.buy_signals):
                    plt.axvline(x=signal_date, color='green', linestyle='--', alpha=0.6,
                                label='Kaufsignal' if i == 0 else None)

            if hasattr(self, "sell_signals"):
                if len(self.sell_signals) == 0:
                    print("Keine Verkaufssignale zum Plotten vorhanden.")
                for i, signal_date in enumerate(self.sell_signals):
                    plt.axvline(x=signal_date, color='red', linestyle='--', alpha=0.6,
                                label='Verkaufssignal' if i == 0 else None) 
            plt.legend()
            
            # Show current recommendation
            recommendation = self.make_buy_recommendation()
            plt.figtext(0.5, 0.01, recommendation, ha='center', fontsize=12, 
                       bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            plt.tight_layout()
            plt.show()
            return
        
        plt.figure(figsize=(14, 8))
        plt.plot(self.data.index, self.data['Close'], label=f'{self.ticker} Aktienkurs')
        
        # Plotte das neueste erkannte Muster
        latest_pattern = self.wave_patterns[-1]
        
        # Extrahiere x (Datums-Indizes) und y (Preis) Werte
        x_indices = [p[0] for p in latest_pattern]
        y_values = [p[1] for p in latest_pattern]
        
        # Konvertiere Indizes in Datumsangaben
        x_dates = [self.data.index[idx] for idx in x_indices]
        
        # Plotte die Punkte des Elliott-Wellenmusters
        plt.scatter(x_dates, y_values, color='red', zorder=5)
        plt.plot(x_dates, y_values, 'r--', alpha=0.7)
        
        # Beschrifte die Wellen
        wave_labels = ['0', '1', '2', '3', '4', '5', 'A', 'B', 'C']
        for i, (date, price) in enumerate(zip(x_dates, y_values)):
            if i < len(wave_labels):
                plt.annotate(wave_labels[i], (date, price), xytext=(5, 5), 
                            textcoords='offset points', fontsize=10)
        
        # Berechne und plotte Fibonacci-Retracement-Levels für Welle 5->A->B->C
        if len(latest_pattern) >= 6:  # Wir haben mindestens bis zur Welle 5
            wave5_idx = x_indices[5] if len(x_indices) > 5 else x_indices[-1]
            wave5_date = self.data.index[wave5_idx]
            wave5_price = y_values[5] if len(y_values) > 5 else y_values[-1]
            
            current_date = self.data.index[-1]
            current_price = float(self.data['Close'].iloc[-1])
            
            # Berechne Fibonacci-Retracements von Welle 5
            fib_levels = self.calculate_fibonacci_levels(y_values[0], wave5_price)
            
            # Plotte horizontale Linien für Fibonacci-Levels
            for fib, price in fib_levels.items():
                plt.axhline(y=price, linestyle='--', alpha=0.5, color='green',
                           label=f'Fib {fib}' if fib in [0.382, 0.5, 0.618] else "")
        
        # Zeichne grüne senkrechte Linien für Kaufsignale
        if hasattr(self, "buy_signals"):
            if len(self.buy_signals) == 0:
                print("Keine Kaufsignale zum Plotten vorhanden.")
            for i, signal_date in enumerate(self.buy_signals):
                plt.axvline(x=signal_date, color='green', linestyle='--', alpha=0.1,
                            label='Kaufsignal' if i == 0 else None)
        
        plt.title(f'{self.ticker} Elliott-Wellen-Analyse')
        plt.xlabel('Datum')
        plt.ylabel('Preis ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Zeige die aktuelle Empfehlung im Plot
        recommendation = self.make_buy_recommendation()
        plt.figtext(0.5, 0.01, recommendation, ha='center', fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.show()
        
    def calculate_strategy_return(self):
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

def analyze_microsoft():

    params = AnalysisParameters(
        ticker="MSFT",
        period="1y",
        sma_short=20,
        sma_long=50,
        extrema_window=10
    )
    analyzer = ElliottWaveAnalyzer(params)
    analyzer.fetch_data()
    patterns = analyzer.detect_elliott_waves()
    
    print("\n--- Elliott-Wellen-Analyse für Microsoft ---")
    
    if patterns:
        print(f"Anzahl gefundener Elliott-Wellenmuster: {len(patterns)}")
        print("\nAktuelle Position im Elliott-Wellenzyklus:")
        position = analyzer.analyze_current_position()
        print(position)
        
        print("\nKaufempfehlung basierend auf Elliott-Wellen:")
        recommendation = analyzer.make_buy_recommendation()
        print(recommendation)
    else:
        print("Keine klaren Elliott-Wellenmuster gefunden.")
        print("\nKaufempfehlung basierend auf alternativen Indikatoren:")
        recommendation = analyzer.make_buy_recommendation()
        print(recommendation)
    
    analyzer.generate_daily_signals()
    analyzer.generate_daily_sell_signals()
    analyzer.calculate_strategy_return()
    
    print("\nDiese Analyse kombiniert die 'Wellen-Teilchen-Dualität' in der Finanzwelt:")
    print("- Wellenförmige Analyse: Elliott-Wellen-Muster und Zyklen")
    print("- Teilchenartige Analyse: Konkrete Preis-Wendepunkte und Fibonacci-Levels")
    
    # Plot der Analyse erstellen
    analyzer.plot_analysis()
    
    return recommendation

def parameter_grid_search():
    import os
    from itertools import product

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH"]  # Beispiel: 10 Ticker – ggf. 100 ergänzen
    sma_short_options = [10, 15, 20]
    sma_long_options = [30, 50, 70]
    extrema_window_options = [5, 10, 15]

    os.makedirs("simulation_logs", exist_ok=True)

    total_runs = 0
    for ticker in tickers:
        for sma_short, sma_long, extrema_window in product(sma_short_options, sma_long_options, extrema_window_options):
            if sma_short >= sma_long:
                continue  # Ignoriere unsinnige Kombinationen

            params = AnalysisParameters(
                ticker=ticker,
                period="1y",
                sma_short=sma_short,
                sma_long=sma_long,
                extrema_window=extrema_window
            )
            analyzer = ElliottWaveAnalyzer(params)
            try:
                analyzer.fetch_data()
                analyzer.detect_elliott_waves()
                analyzer.generate_daily_signals()
                analyzer.generate_daily_sell_signals()

                logfile = f"simulation_logs/{ticker}_{sma_short}_{sma_long}_{extrema_window}.log"
                with open(logfile, "w") as f:
                    f.write(f"Ticker: {ticker}\n")
                    f.write(f"SMA short: {sma_short}, SMA long: {sma_long}, Extrema Window: {extrema_window}\n")
                    if not analyzer.buy_signals or not analyzer.sell_signals:
                        f.write("Unvollständige Signale – keine Rendite berechnet.\n")
                    else:
                        trades = []
                        buy_iter = iter(analyzer.buy_signals)
                        sell_iter = iter(analyzer.sell_signals)
                        current_buy = next(buy_iter, None)
                        current_sell = next(sell_iter, None)

                        while current_buy and current_sell:
                            if current_sell > current_buy:
                                buy_price = float(analyzer.data.at[current_buy, 'Close'])
                                sell_price = float(analyzer.data.at[current_sell, 'Close'])
                                trades.append((buy_price, sell_price))
                                current_buy = next(buy_iter, None)
                                current_sell = next(sell_iter, None)
                            else:
                                current_sell = next(sell_iter, None)

                        if trades:
                            returns = [(sell - buy) / buy for buy, sell in trades]
                            total_return = np.prod([1 + r for r in returns]) - 1
                            f.write(f"Gesamtrendite: {total_return * 100:.2f}%\n")
                            f.write(f"Anzahl Trades: {len(trades)}\n")
                            for i, (buy, sell) in enumerate(trades):
                                f.write(f"Trade {i+1}: Buy={buy:.2f}, Sell={sell:.2f}, Return={(sell - buy) / buy * 100:.2f}%\n")
                        else:
                            f.write("Keine gültigen Trade-Paare gefunden.\n")

                total_runs += 1

            except Exception as e:
                with open(f"simulation_logs/{ticker}_error.log", "a") as f:
                    f.write(f"Fehler bei {ticker} mit Parametern SMA {sma_short}/{sma_long}, Extrema {extrema_window}: {e}\n")
                continue

    print(f"Parameter-Simulation abgeschlossen: {total_runs} Kombinationen ausgeführt.")


if __name__ == "__main__":
    # analyze_microsoft()  # Deaktivieren für Simulation
    parameter_grid_search()