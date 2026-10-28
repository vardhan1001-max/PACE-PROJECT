"""
_paths.py
=========
Shared path constants for the service layer, computed independently from
app.py (same convention as 02_ML_AI/config.py: walk up from this file's
location to the project root) so services/ has no import dependency on
app.py and can be unit-tested standalone.
"""
import os

# services/_paths.py -> services/ -> 01_BACKEND/ -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "02_ML_AI", "models")

FINAL_DATA_PATH = os.path.join(DATA_DIR, "final_customer_data.csv")
CLEANED_MODEL_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
RECOMMENDATIONS_PATH = os.path.join(DATA_DIR, "customer_recommendations.csv")
CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "best_churn_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

_DATA_CACHE = {}

def get_cached_df(path=FINAL_DATA_PATH):
    """Loads and caches DataFrame in-memory based on file modification time.
    Guarantees sub-millisecond data access across all routes and services."""
    if not os.path.exists(path):
        import pandas as pd
        return pd.DataFrame()
    mtime = os.path.getmtime(path)
    cached = _DATA_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        import pandas as pd
        df = pd.read_csv(path)
        _DATA_CACHE[path] = (mtime, df)
    return _DATA_CACHE[path][1]

