# Options Analysis and Visualization Suite

This project provides tools for comprehensive options market analysis and visualization based on financial derivatives theory. It implements concepts from quantitative finance including volatility surfaces, gamma exposure, delta hedging, and market sentiment indicators.

## Key Features

- **Volatility Surface Analysis**: 3D visualizations of implied volatility surfaces across strikes and expirations
- **Put/Call Ratio Analysis**: Calculate and visualize sentiment indicators based on options volume and open interest
- **Greek Exposure Metrics**: Calculate delta and gamma exposure to understand market maker hedging behavior
- **Gamma Flip Detection**: Identify critical points where gamma exposure changes sign
- **Support/Resistance Identification**: Find price levels with significant option interest that may act as barriers
- **Multi-Expiration Analysis**: Compare gamma exposure across different option expiration dates

## Implementation Concepts

The implementation is based on several key financial concepts detailed in the resources:

### Put/Call Ratio and Volume Analysis

The Put/Call Ratio (PCR) is a market sentiment indicator defined as:

```
PCR = Put Volume / Call Volume
```

- **High PCR (>1.2)**: Indicates bearish sentiment (high put activity)
- **Low PCR (<0.7)**: Indicates bullish sentiment (high call activity)
- **Analyzed by**: Volume and open interest, across different expirations

### Call and Put Levels as Price Barriers

- **Put Levels (below current price)**: Act as support where market makers must buy the underlying to hedge
- **Call Levels (above current price)**: Act as resistance where market makers sell the underlying to hedge
- **Mechanism**: Based on delta hedging requirements by market makers

### Gamma and Delta Exposure

Delta Exposure measures first-order price sensitivity:
```
Delta Exposure = Sum(Delta_i * Position_Size_i)
```

Gamma Exposure measures the rate of change of delta:
```
Gamma Exposure = Sum(Gamma_i * Position_Size_i * (ΔS)^2)
```

- **Positive Gamma**: Market makers must trade contrary to price movement (stabilizing)
- **Negative Gamma**: Market makers must trade with price movement (destabilizing)

### Gamma Flip Points

A Gamma Flip occurs when aggregate gamma exposure changes sign:
- Indicates a potential transition between market regimes
- Often corresponds to changes in volatility behavior
- Critical for understanding potential market stability zones

## Directory Structure

- `data/`: Raw option chain data organized by ticker
- `results/`: Analysis results including PCR, gamma exposure, and flip points
- `charts/`: Generated visualizations including volatility surfaces and gamma profiles
- `resources/`: Reference documentation on option theory

## Usage

### Running the Analysis

```bash
# Run with interactive ticker selection
python run_options_analysis.py

# Run for a specific ticker
python run_options_analysis.py --ticker AAPL

# Run for all configured tickers
python run_options_analysis.py --all

# Specify days to expiry
python run_options_analysis.py --ticker MSFT --days 60
```

### Configuration

Edit `options_analysis_config.json` to customize tickers and settings:

```json
{
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "days_to_expiry": 90,
    "output_directory": null
}
```

## Visualization Examples

### Volatility Surface

The 3D volatility surface shows implied volatility across strikes (moneyness) and expiration dates. Features include:
- Term structure (volatility across time)
- Volatility skew (volatility across strikes)
- Colored by volatility level

### Gamma Exposure Profile

The gamma exposure visualization shows:
- Positive and negative gamma by strike price
- Net gamma exposure line
- Current price and key gamma levels
- Gamma flip points where exposure changes sign

### Multi-Expiry GEX Visualization

Shows gamma exposure across four key expirations:
- Nearest expiration
- Next expiration
- Expiration with highest gamma exposure
- Expiration with second highest exposure

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- plotly
- yfinance
- scipy

## References

Based on financial theory resources including:
- Gamma and Delta Exposure analysis
- Put/Call Ratio interpretation
- Volatility surface modeling
- Gamma flip dynamics

## License

MIT License