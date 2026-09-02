import os
import pandas as pd

INPUT_PATH = "data/raw/adzuna_jobs.csv"
OUTPUT_PATH = "data/processed/clean_jobs.csv"


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - DATA CLEANING")
print("=" * 60)


# --------------------------------------------------
# 1. Load raw dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nOriginal dataset shape:", df.shape)


# --------------------------------------------------
# 2. Remove duplicate jobs
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset="id"
)

after = len(df)

print("Duplicate jobs removed:", before - after)


# --------------------------------------------------
# 3. Select useful columns
# --------------------------------------------------

columns = [
    "id",
    "title",
    "description",
    "company.display_name",
    "location.display_name",
    "category.label",
    "category.tag",
    "search_role",
    "created",
    "contract_time",
    "contract_type",
    "salary_min",
    "salary_max",
    "salary_is_predicted",
    "latitude",
    "longitude",
    "redirect_url"
]

# Keep only columns that exist
available_columns = [
    column
    for column in columns
    if column in df.columns
]

df = df[available_columns]


# --------------------------------------------------
# 4. Rename columns
# --------------------------------------------------

df = df.rename(
    columns={
        "company.display_name": "company",
        "location.display_name": "location",
        "category.label": "category",
        "category.tag": "category_tag"
    }
)


# --------------------------------------------------
# 5. Remove jobs without descriptions
# --------------------------------------------------

before = len(df)

df = df.dropna(
    subset=["description"]
)

after = len(df)

print(
    "Jobs without descriptions removed:",
    before - after
)


# --------------------------------------------------
# 6. Clean text fields
# --------------------------------------------------

df["title"] = (
    df["title"]
    .fillna("")
    .str.strip()
)

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

df["search_role"] = (
    df["search_role"]
    .fillna("Unknown")
    .str.strip()
)


# --------------------------------------------------
# 7. Convert date column
# --------------------------------------------------

df["created"] = pd.to_datetime(
    df["created"],
    errors="coerce"
)


# --------------------------------------------------
# 8. Create salary range feature
# --------------------------------------------------

df["salary_range"] = (
    df["salary_max"] - df["salary_min"]
)


# --------------------------------------------------
# 9. Create average salary feature
# --------------------------------------------------

df["salary_avg"] = (
    df["salary_min"] + df["salary_max"]
) / 2


# --------------------------------------------------
# 10. Create description length
# --------------------------------------------------

df["description_length"] = (
    df["description"].str.len()
)


# --------------------------------------------------
# 11. Create output directory
# --------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


# --------------------------------------------------
# 12. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 13. Data quality report
# --------------------------------------------------

print("\nCleaned dataset shape:", df.shape)

print("\nJobs by search role:")
print(
    df["search_role"].value_counts()
)

print("\nMissing values:")
print(
    df.isnull().sum()
    .sort_values(ascending=False)
)

print("\nSalary coverage:")

salary_count = df["salary_avg"].notna().sum()

print(
    f"Jobs with salary information: "
    f"{salary_count}/{len(df)}"
)

print(
    f"\nCleaned data saved to: {OUTPUT_PATH}"
)

print("\nData cleaning completed successfully.")