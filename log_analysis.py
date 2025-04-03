import os
import re
import pandas as pd

log_dir = "simulation_logs"

log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]

results = []

for log_file in log_files:
    with open(os.path.join(log_dir, log_file), "r") as f:
        content = f.read()

    ticker_match = re.search(r"Ticker: (.+)", content)
    sma_short_match = re.search(r"SMA short: (\d+)", content)
    sma_long_match = re.search(r"SMA long: (\d+)", content)
    ext_win_match = re.search(r"Extrema Window: (\d+)", content)
    return_match = re.search(r"Gesamtrendite: ([\-\d\.]+)%", content)
    trades_match = re.search(r"Anzahl Trades: (\d+)", content)

    if all([ticker_match, sma_short_match, sma_long_match, ext_win_match, return_match, trades_match]):
        results.append({
            "Ticker": ticker_match.group(1),
            "SMA_short": int(sma_short_match.group(1)),
            "SMA_long": int(sma_long_match.group(1)),
            "Extrema_Window": int(ext_win_match.group(1)),
            "Return_%": float(return_match.group(1)),
            "Trades": int(trades_match.group(1))
        })

df = pd.DataFrame(results)

# Gruppieren nach Parametern und Mittelwerte berechnen
grouped = df.groupby(["SMA_short", "SMA_long", "Extrema_Window"]).agg(
    avg_return_pct=("Return_%", "mean"),
    avg_trades=("Trades", "mean"),
    count=("Ticker", "count")
).reset_index()

grouped = grouped.sort_values(by="avg_return_pct", ascending=False)

# Anzeige
print(grouped.to_string(index=False))
