import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

BASE_URL = "https://api.adzuna.com/v1/api/jobs/in/search"

# Job roles we want to analyze
JOB_ROLES = [
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "AI engineer",
    "data engineer",
    "business analyst"
]

RESULTS_PER_PAGE = 20
PAGES_PER_ROLE = 3


def fetch_jobs(role, page=1, results_per_page=20):
    """Fetch one page of jobs for a specific role."""

    url = f"{BASE_URL}/{page}"

    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": results_per_page,
        "what": role,
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
        print(f"Timeout: role='{role}', page={page}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: role='{role}', page={page}")
        print(f"Error: {e}")
        return None


# Store all downloaded jobs
all_jobs = []

print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - JOB INGESTION")
print("=" * 60)

# Search every job role
for role in JOB_ROLES:

    print(f"\nSearching for: {role}")

    for page in range(1, PAGES_PER_ROLE + 1):

        print(f"  Downloading page {page}...")

        data = fetch_jobs(
            role=role,
            page=page,
            results_per_page=RESULTS_PER_PAGE
        )

        if data is None:
            print("  Skipping page.")
            continue

        jobs = data.get("results", [])

        print(f"  Jobs received: {len(jobs)}")

        # Add the search role to every job
        for job in jobs:
            job["search_role"] = role

        all_jobs.extend(jobs)


# Check whether anything was downloaded
if not all_jobs:
    print("\nNo jobs were downloaded.")
    print("Check your API credentials or internet connection.")
    exit()


# Convert nested JSON into DataFrame
df = pd.json_normalize(all_jobs)

print("\n" + "=" * 60)
print("INGESTION SUMMARY")
print("=" * 60)

print("\nTotal jobs downloaded:", len(df))

print("\nDataset shape:", df.shape)

print("\nJobs by search role:")
print(df["search_role"].value_counts())


print("\nColumns:")
print(df.columns.tolist())


# Remove duplicate job IDs
before = len(df)

df = df.drop_duplicates(
    subset="id"
)

after = len(df)

print("\nDuplicate jobs removed:", before - after)

print("Final job count:", len(df))


# Create output directory
os.makedirs(
    "data/raw",
    exist_ok=True
)


# Save raw dataset
output_path = "data/raw/adzuna_jobs.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"\nRaw data saved to: {output_path}")


print("\nFirst 5 jobs:")

selected_columns = [
    "title",
    "company.display_name",
    "location.display_name",
    "category.label",
    "search_role"
]

available_columns = [
    column
    for column in selected_columns
    if column in df.columns
]

print(
    df[available_columns].head()
)


print("\nIngestion completed successfully.")