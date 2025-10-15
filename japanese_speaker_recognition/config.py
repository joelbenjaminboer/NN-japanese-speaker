from pathlib import Path
# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels/"
TRAIN_FILE = "data/ae.train"
TEST_FILE = "data/ae.test"
MAX_LEN = 29   # longest utterance length (7–29)
N_FEATURES = 12
OUTPUT_DIR = Path("data/processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)