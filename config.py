"""
Configuration settings for the TU/e LinkedIn Graduate Analyzer.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# API keys
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "fa66a92b5aa84b4e8080f537871c33745b7cb4c3a577487179cd533b899de698")
PROXYCURL_API_KEY = os.environ.get("PROXYCURL_API_KEY", "KogNZkrYkWYPQpn0WpQr5Q")

# Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Default filenames
DEFAULT_URLS_FILE = DATA_DIR / "tue_graduate_urls.txt"
DEFAULT_PROFILES_FILE = DATA_DIR / "tue_graduate_profiles.csv"
DEFAULT_ANALYSIS_FILE = DATA_DIR / "tue_graduate_analysis.csv"
SURNAME_DATASET_FILE = BASE_DIR / "sample_data" / "final_surname_dataset.csv"

# LinkedIn search parameters
LINKEDIN_SEARCH_QUERIES = [
    'site:linkedin.com/in "Eindhoven University of Technology" "Bachelor"',
    'site:linkedin.com/in "Eindhoven University of Technology" "Master"',
    'site:linkedin.com/in "Eindhoven University of Technology" "BSc"',
    'site:linkedin.com/in "Eindhoven University of Technology" "MSc"',
]

# University name variations for detecting TU/e graduates
TUE_VARIATIONS = [
    "eindhoven university of technology", 
    "tu/e", 
    "technische universiteit eindhoven",
    "tu eindhoven"
]

# API settings
API_REQUEST_DELAY = 2  # Seconds between API requests to avoid rate limiting