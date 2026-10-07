from cab_booking_ai.config import MODEL_DIR
from cab_booking_ai.models import CabAIModelManager
from cab_booking_ai.optimizer import RideOptimizer


class CabBookingService:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_manager = CabAIModelManager(model_dir)
        self.optimizer = RideOptimizer()

    def predict(self, request: dict) -> dict:
        payload = {
            'pickup_hour': request.get('pickup_hour', 12),
            'pickup_day': request.get('pickup_day', 1),
            'traffic_density': request.get('traffic_density', 0.5),
            'weather_condition': request.get('weather_condition', 'Clear'),
            'pickup_area': request.get('pickup_area', 'Central'),
            'distance_km': request.get('distance_km', 8.0),
            'driver_count': request.get('driver_count', 20),
            'rider_waiting_minutes': request.get('rider_waiting_minutes', 5),
            'pickup_lat': request.get('pickup_lat', 28.61),
            'pickup_lon': request.get('pickup_lon', 77.20),
            'dropoff_lat': request.get('dropoff_lat', 28.63),
            'dropoff_lon': request.get('dropoff_lon', 77.21),
            'trip_duration_minutes': request.get('trip_duration_minutes', 22.0),
        }

        prediction = self.model_manager.predict(payload)
        demand_score = prediction['demand_score']
        booking_probability = prediction['booking_probability']
        surge_multiplier = prediction['surge_multiplier']

        estimated_fare = payload['distance_km'] * 18.5 * surge_multiplier + payload['trip_duration_minutes'] * 2.2
        recommended_driver_zone = self.optimizer.recommend_driver_zone(
            payload['pickup_area'], demand_score, payload['traffic_density']
        )
        estimated_pickup_minutes = self.optimizer.estimate_pickup_minutes(
            payload['rider_waiting_minutes'], payload['traffic_density'], demand_score
        )

        return {
            'demand_score': round(float(demand_score), 4),
            'booking_probability': round(float(booking_probability), 4),
            'estimated_fare': round(float(estimated_fare), 2),
            'surge_multiplier': round(float(surge_multiplier), 4),
            'estimated_pickup_minutes': float(estimated_pickup_minutes),
            'recommended_driver_zone': recommended_driver_zone,
        }
