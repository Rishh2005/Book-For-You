import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_rides(path: str | Path, n_samples: int = 6000) -> pd.DataFrame:
    """Generate realistic synthetic ride data for cab-demand forecasting."""
    rng = np.random.default_rng(42)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    weather_conditions = ['Clear', 'Rain', 'Storm', 'Fog', 'Traffic Jam']
    areas = ['North', 'South', 'East', 'West', 'Central', 'North-East', 'South-West']

    records = []
    for _ in range(n_samples):
        pickup_hour = int(rng.integers(0, 24))
        pickup_day = int(rng.integers(0, 7))
        traffic_density = float(rng.uniform(0.2, 1.0))
        weather = rng.choice(weather_conditions)
        area = rng.choice(areas)
        distance_km = float(rng.uniform(2.0, 30.0))
        driver_count = int(rng.integers(8, 60))
        rider_waiting_minutes = int(rng.integers(2, 22))

        weather_factor = {'Clear': 0.9, 'Rain': 1.2, 'Storm': 1.5, 'Fog': 1.25, 'Traffic Jam': 1.7}[weather]
        time_factor = 1.3 if 7 <= pickup_hour <= 10 or 17 <= pickup_hour <= 21 else 0.9
        demand_factor = 1.0 + (1.0 - driver_count / 60.0) + (traffic_density * 0.8)
        demand_score = np.clip((distance_km * 0.06) + weather_factor + time_factor + demand_factor * 0.25, 0.1, 1.0)

        surge_multiplier = float(np.clip(0.9 + (1.0 - (driver_count / 60.0)) * 0.9 + (traffic_density * 0.6), 0.9, 2.5))
        trip_duration = float(distance_km * rng.uniform(2.8, 4.6) * (1 + traffic_density * 0.9))
        base_fare = float(distance_km * 18.5 + trip_duration * 3.2)
        estimated_fare = float(base_fare * surge_multiplier)

        booking_probability = float(np.clip((demand_score * 0.78) + (driver_count / 80.0) * 0.18 + (1 / (1 + rider_waiting_minutes / 20.0)) * 0.15, 0.15, 0.97))
        cancellation_probability = float(np.clip((weather_factor * 0.1) + (traffic_density * 0.12) + (rider_waiting_minutes / 120.0), 0.02, 0.45))

        pickup_lat = float(rng.uniform(28.4, 28.9))
        pickup_lon = float(rng.uniform(77.0, 77.5))
        dropoff_lat = float(np.clip(pickup_lat + rng.uniform(-0.04, 0.06), 28.2, 29.2))
        dropoff_lon = float(np.clip(pickup_lon + rng.uniform(-0.08, 0.08), 76.9, 77.8))

        records.append({
            'pickup_hour': pickup_hour,
            'pickup_day': pickup_day,
            'traffic_density': round(traffic_density, 3),
            'weather_condition': weather,
            'pickup_area': area,
            'distance_km': round(distance_km, 2),
            'driver_count': driver_count,
            'rider_waiting_minutes': rider_waiting_minutes,
            'pickup_lat': round(pickup_lat, 5),
            'pickup_lon': round(pickup_lon, 5),
            'dropoff_lat': round(dropoff_lat, 5),
            'dropoff_lon': round(dropoff_lon, 5),
            'trip_duration_minutes': round(trip_duration, 2),
            'base_fare': round(base_fare, 2),
            'surge_multiplier': round(surge_multiplier, 3),
            'estimated_fare': round(estimated_fare, 2),
            'demand_score': round(float(demand_score), 3),
            'booking_probability': round(float(booking_probability), 3),
            'cancellation_probability': round(float(cancellation_probability), 3),
        })

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return df
