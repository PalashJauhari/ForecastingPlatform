"""Vetted code patterns for Time Series and Forecasting."""

# Pattern: Time Series Gap Filling
# Use this when the date column has missing steps.
def FillTimeGaps(df, date_col, freq='D', fill_value=0):
    df[date_col] = pd.to_datetime(df[date_col])
    full_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq=freq)
    df = df.set_index(date_col).reindex(full_range).fillna(fill_value).reset_index()
    df.rename(columns={'index': date_col}, inplace=True)
    return df

# Pattern: Rolling Statistics (Feature Engineering)
# Use this to create trend-based features.
def AddRollingFeatures(df, value_col, windows=[7, 30]):
    for w in windows:
        df[f'{value_col}_rolling_mean_{w}'] = df[value_col].rolling(window=w).mean()
        df[f'{value_col}_rolling_std_{w}'] = df[value_col].rolling(window=w).std()
    return df

# Pattern: Simple Prophet Forecast
# Use this for a robust baseline forecast with seasonality.
def RunProphetForecast(df, date_col, value_col, horizon=30):
    from prophet import Prophet
    pdf = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(pdf)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
