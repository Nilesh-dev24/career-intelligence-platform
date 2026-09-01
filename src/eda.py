import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "data/processed/clean_jobs.csv"

df = pd.read_csv(INPUT_PATH)


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - EDA")
print("=" * 60)


# --------------------------------------------------
# 1. Dataset overview
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 2. Most common job titles
# --------------------------------------------------

print("\nTop 20 job titles:")

print(
    df["title"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# 3. Companies hiring
# --------------------------------------------------

print("\nTop 20 companies:")

print(
    df["company"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# 4. Locations
# --------------------------------------------------

print("\nTop locations:")

print(
    df["location"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# 5. Categories
# --------------------------------------------------

print("\nJob categories:")

print(
    df["category"]
    .value_counts()
)


# --------------------------------------------------
# 6. Salary prediction availability
# --------------------------------------------------

print("\nSalary prediction distribution:")

print(
    df["salary_is_predicted"]
    .value_counts()
)


# --------------------------------------------------
# 7. Description length
# --------------------------------------------------

df["description_length"] = df["description"].str.len()

print("\nDescription statistics:")

print(
    df["description_length"].describe()
)


# --------------------------------------------------
# 8. Visualize top locations
# --------------------------------------------------

top_locations = (
    df["location"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 6))

top_locations.sort_values().plot(
    kind="barh"
)

plt.title("Top Job Locations")
plt.xlabel("Number of Jobs")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig(
    "data/processed/top_locations.png",
    dpi=150
)

plt.show()


print("\nEDA completed successfully.")