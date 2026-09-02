import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "data/processed/clean_jobs.csv"
OUTPUT_DIR = "data/processed/eda"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - EDA V2")
print("=" * 60)


print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 2. Basic dataset information
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 3. Jobs by career role
# --------------------------------------------------

role_counts = df["search_role"].value_counts()

print("\nJobs by career role:")
print(role_counts)


plt.figure(figsize=(10, 6))

role_counts.sort_values().plot(
    kind="barh"
)

plt.title("Jobs by Career Role")
plt.xlabel("Number of Jobs")
plt.ylabel("Career Role")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/jobs_by_role.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# 4. Top job titles
# --------------------------------------------------

print("\nTop 20 job titles:")

print(
    df["title"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# 5. Top companies
# --------------------------------------------------

print("\nTop 20 companies:")

print(
    df["company"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# 6. Top locations
# --------------------------------------------------

print("\nTop 20 locations:")

print(
    df["location"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# 7. Job categories
# --------------------------------------------------

print("\nJob categories:")

print(
    df["category"]
    .value_counts()
)


# --------------------------------------------------
# 8. Description length
# --------------------------------------------------

print("\nDescription length statistics:")

print(
    df["description_length"]
    .describe()
)


print("\nDescription length by career role:")

print(
    df.groupby("search_role")["description_length"]
    .describe()
)


# --------------------------------------------------
# 9. Salary analysis
# --------------------------------------------------

salary_data = df[
    df["salary_avg"].notna()
]

print("\nSalary information:")

print(
    f"Jobs with salary data: "
    f"{len(salary_data)}/{len(df)}"
)


if len(salary_data) > 0:

    print("\nSalary statistics:")

    print(
        salary_data[
            ["salary_min", "salary_max", "salary_avg"]
        ].describe()
    )

    print("\nAverage salary by career role:")

    print(
        salary_data
        .groupby("search_role")["salary_avg"]
        .agg(["count", "mean", "min", "max"])
        .sort_values("mean", ascending=False)
    )


# --------------------------------------------------
# 10. Salary coverage by role
# --------------------------------------------------

salary_coverage = (
    df.groupby("search_role")["salary_avg"]
    .apply(lambda x: x.notna().sum())
    .sort_values(ascending=False)
)

print("\nSalary coverage by role:")

print(salary_coverage)


# --------------------------------------------------
# 11. Contract information
# --------------------------------------------------

print("\nContract type distribution:")

print(
    df["contract_type"]
    .value_counts(dropna=False)
)


print("\nContract time distribution:")

print(
    df["contract_time"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# 12. Save role summary
# --------------------------------------------------

role_summary = (
    df.groupby("search_role")
    .agg(
        job_count=("id", "count"),
        avg_description_length=(
            "description_length",
            "mean"
        ),
        salary_jobs=(
            "salary_avg",
            lambda x: x.notna().sum()
        )
    )
    .sort_values(
        "job_count",
        ascending=False
    )
)

role_summary.to_csv(
    f"{OUTPUT_DIR}/role_summary.csv"
)


print(
    f"\nRole summary saved to: "
    f"{OUTPUT_DIR}/role_summary.csv"
)


# --------------------------------------------------
# 13. Top locations visualization
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
    f"{OUTPUT_DIR}/top_locations.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# 14. Jobs by role visualization
# --------------------------------------------------

role_counts = (
    df["search_role"]
    .value_counts()
)


plt.figure(figsize=(10, 6))

role_counts.sort_values().plot(
    kind="barh"
)

plt.title("Career Role Distribution")
plt.xlabel("Number of Jobs")
plt.ylabel("Career Role")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/career_role_distribution.png",
    dpi=150
)

plt.show()


print("\n" + "=" * 60)
print("EDA V2 COMPLETED SUCCESSFULLY")
print("=" * 60)