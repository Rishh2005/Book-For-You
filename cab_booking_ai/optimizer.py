import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from cab_booking_ai.config import MODEL_DIR
from cab_booking_ai.data_generation import generate_synthetic_rides
from cab_booking_ai.feature_engineering import prepare_features, prepare_targets


class CabAIModelManager:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.demand_model = None
        self.booking_model = None
        self.pricing_model = None

    def _train_regressor(self, X_train, y_train, random_state=42):
        model = RandomForestRegressor(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model

    def _train_classifier(self, X_train, y_train, random_state=42):
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model

    def train(self, dataset_path=None):
        if dataset_path is not None:
            df = pd.read_csv(dataset_path)
        else:
            df = generate_synthetic_rides(self.model_dir / '..' / 'data' / 'rides_dataset.csv')

        X = prepare_features(df)
        demand_target, booking_target, pricing_target = prepare_targets(df)

        X_train, X_test, y_train_d, y_test_d = train_test_split(X, demand_target, test_size=0.2, random_state=42)
        _, _, y_train_b, y_test_b = train_test_split(X, booking_target, test_size=0.2, random_state=42)
        _, _, y_train_p, y_test_p = train_test_split(X, pricing_target, test_size=0.2, random_state=42)

        self.demand_model = self._train_regressor(X_train, y_train_d)
        self.booking_model = self._train_classifier(X_train, y_train_b)
        self.pricing_model = self._train_regressor(X_train, y_train_p)

        demand_preds = self.demand_model.predict(X_test)
        booking_preds = self.booking_model.predict(X_test)
        pricing_preds = self.pricing_model.predict(X_test)

        demand_rmse = mean_squared_error(y_test_d, demand_preds, squared=False)
        booking_accuracy = (booking_preds == y_test_b).mean()
        pricing_rmse = mean_squared_error(y_test_p, pricing_preds, squared=False)

        self.save()
        return {
            'demand_rmse': round(float(demand_rmse), 4),
            'booking_accuracy': round(float(booking_accuracy), 4),
            'pricing_rmse': round(float(pricing_rmse), 4),
            'r2': round(float(r2_score(y_test_d, demand_preds)), 4),
        }

    def save(self):
        joblib.dump(self.demand_model, self.model_dir / 'demand_model.joblib')
        joblib.dump(self.booking_model, self.model_dir / 'booking_model.joblib')
        joblib.dump(self.pricing_model, self.model_dir / 'pricing_model.joblib')

    def load(self):
        self.demand_model = joblib.load(self.model_dir / 'demand_model.joblib')
        self.booking_model = joblib.load(self.model_dir / 'booking_model.joblib')
        self.pricing_model = joblib.load(self.model_dir / 'pricing_model.joblib')

    def predict(self, payload: dict) -> dict:
        if self.demand_model is None or self.booking_model is None or self.pricing_model is None:
            self.load()

        feature_row = pd.DataFrame([payload])
        features = prepare_features(feature_row)

        demand_score = float(np.clip(self.demand_model.predict(features)[0], 0.0, 1.0))
        booking_probability = float(np.clip(self.booking_model.predict_proba(features)[0][1], 0.0, 1.0))
        surge_multiplier = float(np.clip(self.pricing_model.predict(features)[0], 0.9, 2.5))

        return {
            'demand_score': round(demand_score, 4),
            'booking_probability': round(booking_probability, 4),
            'surge_multiplier': round(surge_multiplier, 4),
        }
