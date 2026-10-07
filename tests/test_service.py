from cab_booking_ai.config import DATASET_PATH
from cab_booking_ai.data_generation import generate_synthetic_rides
from cab_booking_ai.models import CabAIModelManager


def train_pipeline():
    data = generate_synthetic_rides(DATASET_PATH)
    model_manager = CabAIModelManager()
    metrics = model_manager.train(DATASET_PATH)
    print('Generated rows:', len(data))
    print('Training metrics:', metrics)


if __name__ == '__main__':
    train_pipeline()
