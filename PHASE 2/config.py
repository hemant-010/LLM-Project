import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "models" / "artifacts")))

EVENT_TYPES = [
    "insider_selling", "analyst_downgrade", "earnings_miss",
    "fraud_investigation", "product_recall", "merger_acquisition",
    "bankruptcy_distress", "positive_earnings", "other"
]

DIRECTIONS = ["increase", "decrease", "no_significant_change"]
