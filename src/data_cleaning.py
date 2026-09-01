import pandas as pd


INPUT_PATH = "data/raw/adzuna_jobs.csv"
OUTPUT_PATH = "data/processed/clean_jobs.csv"


# Load raw data
df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 1. Remove duplicate jobs
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset="id")

after = len(df)

print("Duplicate jobs removed:", before - after)


# --------------------------------------------------
# 2. Select useful columns
# --------------------------------------------------

columns = [
    "id",
    "title",
    "description",
    "company.display_name",
    "location.display_name",
    "category.label",
    "category.tag",
    "created",
    "contract_time",
    "contract_type",
    "salary_is_predicted",
    "latitude",
    "longitude",
    "redirect_url"
]

df = df[columns]


# --------------------------------------------------
# 3. Rename columns
# --------------------------------------------------

df = df.rename(columns={
    "company.display_name": "company",
    "location.display_name": "location",
    "category.label": "category",
    "category.tag": "category_tag"
})


# --------------------------------------------------
# 4. Remove rows without descriptions
# --------------------------------------------------

before = len(df)

df = df.dropna(subset=["description"])

after = len(df)

print("Jobs without descriptions removed:", before - after)


# --------------------------------------------------
# 5. Clean text columns
# --------------------------------------------------

df["title"] = df["title"].fillna("").str.strip()

df["description"] = (
    df["description"]
    .fillna("")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["company"] = (
    df["company"]
    .fillna("Unknown")
    .str.strip()
)

df["location"] = (
    df["location"]
    .fillna("Unknown")
    .str.strip()
)


# --------------------------------------------------
# 6. Convert date
# --------------------------------------------------

df["created"] = pd.to_datetime(
    df["created"],
    errors="coerce"
)


# --------------------------------------------------
# 7. Create processed-data directory
# --------------------------------------------------

import os

os.makedirs("data/processed", exist_ok=True)


# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 9. Final information
# --------------------------------------------------

print("\nCleaned dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print(f"\nCleaned data saved to: {OUTPUT_PATH}")