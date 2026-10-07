# Cab Booking AI & ML Platform

A professional cab booking intelligence system built with Python, using machine learning to forecast rider demand, optimize fare pricing, estimate pickup times, and recommend driver matching. This project is designed as an advanced AI/ML application for a real-world transportation platform.

## Project Features

- Demand forecasting for taxi bookings across time slots
- Dynamic surge pricing recommendations
- Estimated trip fare calculation
- Driver-rider matching and route prioritization
- Real-time prediction API using FastAPI
- Synthetic data generation for experimentation and model training
- Scalable Python project structure for future deployment

## Business Use Cases

- Predict demand spikes during peak hours
- Estimate trip fares based on distance, weather, and traffic
- Suggest surge prices to balance supply and demand
- Rank drivers by proximity, trip complexity, and availability
- Improve customer experience with faster booking predictions

## Architecture

The project follows a production-style ML architecture:

- Data generation layer: synthetic taxi trip data
- Feature engineering layer: time, weather, traffic, area, distance, and trip attributes
- Model training layer: Random Forest Regressor and Classifier models
- Optimization layer: surge pricing and driver matching recommendations
- API layer: FastAPI endpoints for predictions

## Tech Stack

- Python 3.11+
- pandas
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- joblib

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── train_model.py
├── cab_booking_ai/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── optimizer.py
│   └── service.py
├── tests/
│   └── test_service.py
└── data/
    └── rides_dataset.csv
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Models

```bash
python train_model.py
```

This creates synthetic taxi ride data and trains the demand, booking probability, and pricing models.

## Run the API

```bash
python main.py
```

Then open:

- http://localhost:8000/docs

## Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_lat": 28.6139,
    "pickup_lon": 77.2090,
    "dropoff_lat": 28.6271,
    "dropoff_lon": 77.2103,
    "pickup_hour": 18,
    "pickup_day": 5,
    "traffic_density": 0.82,
    "weather_condition": "Rain",
    "distance_km": 8.5,
    "driver_count": 23,
    "rider_waiting_minutes": 6
  }'
```

## Example Response

```json
{
  "demand_score": 0.89,
  "booking_probability": 0.84,
  "estimated_fare": 504.23,
  "surge_multiplier": 1.46,
  "estimated_pickup_minutes": 5.2,
  "recommended_driver_zone": "North-East"
}
```

## Future Enhancements

- Real booking data ingestion from APIs and databases
- Deep learning models with LSTM or XGBoost for advanced forecasting
- Reservation and ride cancellation prediction
- Kafka streaming integration for live demand signals
- Docker deployment and cloud hosting
- Dashboard for monitoring ML model performance

## License

This project is intended for learning and professional portfolio use.
