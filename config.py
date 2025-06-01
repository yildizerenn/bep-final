"""
Configuration settings for the TU/e LinkedIn Graduate Analyzer.
Optimized for reduced credit consumption and improved accuracy.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# API keys - REMOVE HARDCODED KEYS FOR SECURITY
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
PROXYCURL_API_KEY = os.environ.get("PROXYCURL_API_KEY")

if not SERPAPI_API_KEY:
    print("Warning: SERPAPI_API_KEY not found in environment variables")
if not PROXYCURL_API_KEY:
    print("Warning: PROXYCURL_API_KEY not found in environment variables")

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

# OPTIMIZED LinkedIn search parameters - Reduced credit consumption
LINKEDIN_SEARCH_QUERIES = [
    # Single comprehensive query instead of 4 separate ones
    'site:linkedin.com/in ("Eindhoven University of Technology" OR "TU/e" OR "Technische Universiteit Eindhoven" OR "TU Eindhoven")'
]

# Credit-saving limits
MAX_RESULTS_PER_QUERY = 10  # Limit to first 10 results only
MAX_PAGES_PER_QUERY = 1     # Only first page

# University name variations for detecting TU/e graduates (expanded for better accuracy)
TUE_VARIATIONS = [
    "eindhoven university of technology",
    "tu/e", 
    "technische universiteit eindhoven",
    "tue"
]

# API settings
API_REQUEST_DELAY = 2  # Seconds between API requests to avoid rate limiting