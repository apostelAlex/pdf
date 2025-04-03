import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

class ElliottWaveAnalyzer:
    def __init__(self, ticker="MSFT", period="1y"):
        self.ticker = ticker
        self.period = period
        self.data = None
        self.wave_patterns = None
        self.fibonacci_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]
    
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
            sma20 = self.data['Close'].rolling(window=20).mean()
            sma50 = self.data['Close'].rolling(window=50).mean()
            plt.plot(self.data.index, sma20, 'g--', label='SMA 20')
            plt.plot(self.data.index, sma50, 'r--', label='SMA 50')
            
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


def analyze_microsoft():
    analyzer = ElliottWaveAnalyzer(ticker="MSFT", period="1y")
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
    
    print("\nDiese Analyse kombiniert die 'Wellen-Teilchen-Dualität' in der Finanzwelt:")
    print("- Wellenförmige Analyse: Elliott-Wellen-Muster und Zyklen")
    print("- Teilchenartige Analyse: Konkrete Preis-Wendepunkte und Fibonacci-Levels")
    
    # Plot der Analyse erstellen
    analyzer.plot_analysis()
    
    return recommendation


if __name__ == "__main__":
    analyze_microsoft()