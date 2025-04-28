import pandas as pd
from openbb import obb
import plotly.graph_objects as go
import datetime

import numpy as np
from scipy.interpolate import griddata

def fetch_option_data(symbol, min_moneyness=0.5, max_moneyness=1.5):
    """
    Fetch raw option chains for given symbol and filter by moneyness.
    Returns df_puts, df_calls.
    """
    oc = obb.derivatives.options.chains(symbol)
    dd = oc.to_dict()
    options = dd["CboeOptionsChainsData"]
    df = pd.DataFrame(options)
    df_puts = df[df["option_type"] == "put"].copy()
    df_calls = df[df["option_type"] == "call"].copy()
    # Filter by moneyness
    df_calls = df_calls[
        (df_calls["strike"] >= df_calls["underlying_price"] * min_moneyness) &
        (df_calls["strike"] <= df_calls["underlying_price"] * max_moneyness)
    ].copy()
    df_puts = df_puts[
        (df_puts["strike"] >= df_puts["underlying_price"] * min_moneyness) &
        (df_puts["strike"] <= df_puts["underlying_price"] * max_moneyness)
    ].copy()
    return df_puts, df_calls

def get_volatility_surface(df):
    """
    Create a pivot table of implied volatility indexed by DTE and strike.
    """
    return df.pivot_table(
        values='implied_volatility',
        index='dte',
        columns='strike',
        aggfunc='mean'
    )

def smooth_surface_rolling(surface, window_dte=3, window_strike=3, fill_median=True):
    """
    Apply a rolling mean smoothing on both DTE and strike axes.
    """
    surf = surface.copy()
    surf = surf.rolling(window=window_dte, axis=0, min_periods=1, center=True).mean()
    surf = surf.rolling(window=window_strike, axis=1, min_periods=1, center=True).mean()
    if fill_median:
        surf = surf.fillna(surface.stack().median())
    return surf

def interpolate_surface(surface, method='linear'):
    """
    Interpolate missing values on the volatility surface using griddata.
    """
    strikes = surface.columns.values
    dtes = surface.index.values
    X, Y = np.meshgrid(strikes, dtes)
    mask = ~surface.isna().values
    points = np.column_stack([X[mask], Y[mask]])
    values = surface.values[mask]
    target = np.column_stack([X.ravel(), Y.ravel()])
    interp_vals = griddata(points, values, target, method=method)
    surf_interp = pd.DataFrame(interp_vals.reshape(surface.shape), index=dtes, columns=strikes)
    # Fill any remaining NaNs with nearest neighbor
    missing = surf_interp.isna().values
    if missing.any():
        nearest = griddata(points, values, target, method='nearest')
        surf_interp.values[missing] = nearest.reshape(surface.shape)[missing]
    return surf_interp

def plot_surface(surface, title='Volatilitätsoberfläche'):
    """
    Plot a 3D volatility surface using Plotly.
    """
    fig = go.Figure(data=[go.Surface(
        z=surface.values,
        x=surface.columns.to_list(),
        y=surface.index.to_list(),
        colorscale='Viridis'
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Strike Preis',
            yaxis_title='Verfallsdatum',
            zaxis_title='Implizite Volatilität',
            xaxis=dict(tickformat=".0f"),
            yaxis=dict(tickvals=surface.index.to_list(), autorange='reversed'),
            camera=dict(eye=dict(x=1.2, y=-1.2, z=0.8))
        ),
        margin=dict(l=20, r=20, b=20, t=40)
    )
    fig.show()

if __name__ == '__main__':
    df_puts, df_calls = fetch_option_data('AAPL')
    surface_calls = get_volatility_surface(df_calls)
    smoothed = smooth_surface_rolling(surface_calls)
    interpolated = interpolate_surface(smoothed)
    plot_surface(interpolated, title='AAPL Call Volatility Surface')
