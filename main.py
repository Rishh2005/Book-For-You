from fastapi import FastAPI
from pydantic import BaseModel

from cab_booking_ai.service import CabBookingService

app = FastAPI(title='Cab Booking AI API', version='1.0.0')
service = CabBookingService()


class RideRequest(BaseModel):
    pickup_lat: float = 28.6139
    pickup_lon: float = 77.2090
    dropoff_lat: float = 28.6271
    dropoff_lon: float = 77.2103
    pickup_hour: int = 18
    pickup_day: int = 5
    traffic_density: float = 0.82
    weather_condition: str = 'Rain'
    pickup_area: str = 'Central'
    distance_km: float = 8.5
    driver_count: int = 23
    rider_waiting_minutes: int = 6
    trip_duration_minutes: float = 22.0


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'cab-booking-ai'}


@app.post('/predict')
def predict(request: RideRequest):
    output = service.predict(request.model_dump())
    return output
