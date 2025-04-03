import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Funktion zur Berechnung des RSI (falls ta-lib nicht verfügbar ist)
def calculate_rsi(data, periods=14):
 delta = data['Close'].diff()
 gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
 loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
 rs = gain / loss
 rsi = 100 - (100 / (1 + rs))
 return rsi

# Funktion zur Identifikation von Elliott-Wellen (vereinfacht)
def identify_elliott_waves(data, window=20):
    # Vereinfachte Logik: Wir suchen nach Hoch- und Tiefpunkten, um Wellen zu identifizieren
    highs = data['High'].rolling(window =window).max()
    lows = data['Low'].rolling(window=window).min()
    
    # Markiere Hoch- und Tiefpunkte
    data['is_high'] = (data['High'] == highs)
    data['is_low'] = (data['Low'] == lows)
    
    # Identifiziere Wellenmuster (Impuls: 5 Wellen, Korrektur: 3 Wellen)
    wave_points = []
    for i in range(1, len(data)):
        if data['is_high'].iloc[i]:
            wave_points.append(('H', data.index[i], data['High'].iloc[i]))
        elif data['is_low'].iloc[i]:
            wave_points.append(('L', data.index[i], data['Low'].iloc[i]))
    
    return wave_points

# Funktion zur Berechnung von Fibonacci-Retracements
def calculate_fibonacci_levels(high, low):
    diff = high - low
    levels = {
    '23.6%': high - diff * 0.236,
    '38.2%': high - diff * 0.382,
    '50.0%': high - diff * 0.5,
    '61.8%': high - diff * 0.618,
    }
    return levels

# Hauptfunktion zur Kauf-/Verkaufsentscheidung
def trading_decision(data, wave_points):
    # Letzter Preis
    last_price = data['Close'].iloc [-1]
    
    # Berechne RSI
    data['RSI'] = calculate_rsi(data)
    last_rsi = data['RSI'].iloc[-1]
    
    # Identifiziere die letzten Wellen
    if len(wave_points) < 2:
        return "Nicht genug Daten für eine Analyse."
 
    # Letzte zwei Punkte für Fibonacci-Retracements
    last_two_points = wave_points[-2:]
    if last_two_points[0][0] == 'H' and last_two_points[1][0] == 'L':
        high = last_two_points[0][2]
        low = last_two_points[1][2]
    elif last_two_points[0][0] == 'L' and last_two_points[1][0] == 'H':
        low = last_two_points[0][2]
        high = last_two_points[1][2]
    else:
        return "Kein klares Wellenmuster erkannt."
    
    # Berechne Fibonacci-Levels
    fib_levels = calculate_fibonacci_levels(high, low)
    
    # Entscheidungslogik
    decision = "Halten"
    reasoning = []
    
    # Prüfe, ob der Preis in der Nähe eines Fibonacci-Levels ist
    for level, value in fib_levels.items():
        if abs(last_price - value) / value < 0.01: # Preis ist innerhalb von 1% des Levels
            reasoning.append(f"Preis in der Nähe des Fibonacci-Levels {level} ({value:.2f})")
        if last_rsi < 30: # RSI zeigt Überverkauft an
            decision = "Kaufen"
            reasoning.append("RSI zeigt Überverkauft an (< 30)")
        elif last_rsi > 70: # RSI zeigt Überkauft an
            decision = "Verkaufen"
            reasoning.append("RSI zeigt Überkauft an (> 70)")
    
    # Prüfe Wellenmuster (vereinfacht)
    if len(wave_points) >= 5:
        last_five = wave_points[-5:]
    # Prüfe, ob wir in einer Korrekturphase (Welle 4 oder A) sind
    if last_five[-1][0] == 'L' and last_five[-2][0] == 'H':
        reasoning.append("Möglicherweise in Korrekturphase (Welle 4 oder A)")
    if last_price <= fib_levels['61.8%'] and last_rsi < 40:
        decision = "Kaufen"
        reasoning.append("Günstiger Einstiegspunkt in Korrekturphase")
        
    return decision, reasoning

# Daten herunterladen (Microsoft)
ticker = yf.Ticker("MSFT")

data = ticker.history(period='1y')

# Elliott-Wellen identifizieren
wave_points = identify_elliott_waves(data)

# Kauf-/Verkaufsentscheidung
print(data)
result = trading_decision(data, wave_points)
print(result)
exit()
# Ergebnis ausgeben
print(f"\nAnalyse für Microsoft (MSFT) am {end_date.strftime('%Y-%m-%d')}:")
print(f"Letzter Preis: {data['Close'].iloc[-1]:.2f}")
print(f"RSI: {data['RSI'].iloc[-1]:.2f}")
print(f"Entscheidung: {decision}")
print("Begründung:")
for reason in reasoning:
 print(f"- {reason}")

# Optional: Visualisierung
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Schlusskurs')
for point in wave_points:
    if point[0] == 'H':
        plt.plot(point[1], point[2], 'ro', label='Hochpunkt' if 'Hochpunkt' not in plt.gca().get_legend_handles_labels()[1] else "")
    elif point[0] == 'L':
        plt.plot(point[1], point[2], 'go', label='Tiefpunkt' if 'Tiefpunkt' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title('Microsoft Aktienkurs mit Elliott-Wellen')
plt.xlabel('Datum')
plt.ylabel(' Preis (USD)')
plt.legend()
plt.grid()
plt.show()