from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_PATH = DATA_DIR / 'rides_dataset.csv'
MODEL_DIR = PROJECT_ROOT / 'models'

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
