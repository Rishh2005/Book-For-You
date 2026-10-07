import numpy as np
import pandas as pd


def weather_to_numeric(weather: str) -> int:
    mapping = {
        'Clear': 0,
        'Rain': 1,
        'Storm': 2,
        'Fog': 3,
        'Traffic Jam': 4,
    }
    return mapping.get(weather, 0)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features used by the ML models."""
    features = df.copy()

    features['pickup_hour_sin'] = np.sin(2 * np.pi * features['pickup_hour'] / 24)
    features['pickup_hour_cos'] = np.cos(2 * np.pi * features['pickup_hour'] / 24)
    features['pickup_day_sin'] = np.sin(2 * np.pi * features['pickup_day'] / 7)
    features['pickup_day_cos'] = np.cos(2 * np.pi * features['pickup_day'] / 7)

    features['weather_code'] = features['weather_condition'].apply(weather_to_numeric)
    features['area_code'] = features['pickup_area'].map({
        'North': 0,
        'South': 1,
        'East': 2,
        'West': 3,
        'Central': 4,
        'North-East': 5,
        'South-West': 6,
    })

    features['distance_to_driver_wait'] = features['distance_km'] / (features['driver_count'] + 1)
    features['traffic_wait_score'] = features['traffic_density'] * features['rider_waiting_minutes']
    features['demand_pressure'] = features['demand_score'] * (1 + features['traffic_density'])

    selected_cols = [
        'pickup_hour', 'pickup_day', 'traffic_density', 'weather_code', 'area_code',
        'distance_km', 'driver_count', 'rider_waiting_minutes', 'pickup_hour_sin',
        'pickup_hour_cos', 'pickup_day_sin', 'pickup_day_cos', 'distance_to_driver_wait',
        'traffic_wait_score', 'demand_pressure', 'trip_duration_minutes'
    ]
    return features[selected_cols]


def prepare_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    demand_target = df['demand_score']
    booking_target = (df['booking_probability'] >= 0.6).astype(int)
    pricing_target = df['surge_multiplier']
    return demand_target, booking_target, pricing_target
