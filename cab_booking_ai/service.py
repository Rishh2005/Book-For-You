class RideOptimizer:
    def __init__(self):
        self.zone_priority = {
            'North': 1.15,
            'South': 1.0,
            'East': 1.08,
            'West': 1.03,
            'Central': 1.22,
            'North-East': 1.3,
            'South-West': 1.18,
        }

    def recommend_driver_zone(self, pickup_area: str, demand_score: float, traffic_density: float) -> str:
        weighted_score = self.zone_priority.get(pickup_area, 1.0) * (0.6 + demand_score) * (1 + traffic_density)
        if weighted_score > 1.9:
            return 'North-East'
        if weighted_score > 1.6:
            return 'Central'
        if weighted_score > 1.35:
            return 'North'
        return 'South'

    def estimate_pickup_minutes(self, rider_waiting_minutes: int, traffic_density: float, demand_score: float) -> float:
        return round(max(2.0, rider_waiting_minutes * 0.7 + traffic_density * 4.5 - demand_score * 2.2), 2)
