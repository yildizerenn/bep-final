# TU/e LinkedIn Graduate Analyzer

A comprehensive tool for scraping, collecting, and analyzing LinkedIn profiles of graduates from Eindhoven University of Technology (TU/e).

## Features

- **LinkedIn URL Scraper**: Uses Google Search to find LinkedIn profiles of TU/e graduates
- **Profile Data Collection**: Extracts detailed profile information using Proxycurl API
- **Graduate Analysis**: Processes education and work experience data
- **Nationality Classification**: Uses machine learning to identify Dutch vs. international graduates based on surnames

## Project Structure

```
tue-linkedin-analyzer/
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── config.py                     # Configuration settings
├── main.py                       # Main pipeline script
├── scraper/                      # Scraping modules
├── analysis/                     # Analysis modules
├── utils/                        # Utility functions
├── data/                         # Data storage
├── models/                       # ML models
└── tests/                        # Test scripts
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tue-linkedin-analyzer.git
cd tue-linkedin-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API keys:
Edit `config.py` and add your SerpAPI and Proxycurl API keys.

## Usage

### Running the Complete Pipeline

To run the entire pipeline (scrape URLs, collect profile data, and analyze):

```bash
python main.py
```

### Running Individual Components

To only scrape LinkedIn URLs:
```bash
python -m scraper.link_scraper
```

To collect profile data from existing URLs:
```bash
python -m scraper.profile_scraper --url_file path/to/urls.txt
```

To analyze already collected profile data:
```bash
python -m analysis.profile_analyzer --data_file path/to/profiles.csv
```

### Testing

To test the scraper with 10 random LinkedIn profiles:
```bash
python tests/test_random_profiles.py
```

## Requirements

- Python 3.8+
- SerpAPI account and API key
- Proxycurl account and API key
- Required Python packages (see requirements.txt)

## Ethical Considerations

This tool is designed for academic and research purposes only. Please be mindful of:
- LinkedIn's terms of service
- Proxycurl's API usage limits and terms
- Privacy concerns when collecting and storing personal data
- Appropriate data anonymization for research publications

## License

This project is licensed under the MIT License - see the LICENSE file for details.