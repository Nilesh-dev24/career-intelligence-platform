import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search"


def fetch_jobs(page=1, results_per_page=20):
    """Fetch one page of jobs from the Adzuna API."""

    url = f"{BASE_URL}/{page}"

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "what": "data scientist",
        "where": "India",
        "content-type": "application/json"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        print(f"Timeout while downloading page {page}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed for page {page}: {e}")
        return None


# Store all downloaded jobs
all_jobs = []

# Download 5 pages
for page in range(1, 6):

    print(f"Downloading page {page}...")

    data = fetch_jobs(
        page=page,
        results_per_page=20
    )

    if data is None:
        print(f"Skipping page {page}")
        continue

    jobs = data.get("results", [])

    print(f"Jobs received: {len(jobs)}")

    all_jobs.extend(jobs)


# Check whether we received any jobs
if not all_jobs:
    print("\nNo jobs were downloaded.")
    print("Please check your API credentials or internet connection.")
    exit()


# Convert nested JSON into a Pandas DataFrame
df = pd.json_normalize(all_jobs)


print("\nTotal jobs downloaded:", len(all_jobs))

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# Save raw data
output_path = "data/raw/adzuna_jobs.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"\nData saved to: {output_path}")


# Display selected columns
selected_columns = [
    "title",
    "company.display_name",
    "location.display_name",
    "category.label",
    "description"
]

# Only use columns that actually exist
available_columns = [
    column for column in selected_columns
    if column in df.columns
]

print("\nFirst 5 jobs:")
print(df[available_columns].head())