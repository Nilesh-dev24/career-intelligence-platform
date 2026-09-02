import os
import ast
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/jobs_with_skills.csv"
OUTPUT_DIR = "data/processed/skill_analysis"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - SKILL ANALYSIS V1")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Convert skills column back to Python lists
# --------------------------------------------------

df["skills"] = df["skills"].apply(
    ast.literal_eval
)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# 1. Overall skill frequency
# --------------------------------------------------

skill_frequency = {}

for skills in df["skills"]:

    for skill in skills:

        skill_frequency[skill] = (
            skill_frequency.get(skill, 0) + 1
        )


skill_frequency = (
    pd.Series(skill_frequency)
    .sort_values(ascending=False)
)


print("\nTop 20 skills across all jobs:")

print(
    skill_frequency.head(20)
)


# --------------------------------------------------
# 2. Skill percentage
# --------------------------------------------------

skill_percentage = (
    skill_frequency / len(df) * 100
)


skill_report = pd.DataFrame({
    "skill": skill_frequency.index,
    "job_count": skill_frequency.values,
    "percentage": skill_percentage.values
})


skill_report["percentage"] = (
    skill_report["percentage"]
    .round(2)
)


skill_report.to_csv(
    f"{OUTPUT_DIR}/overall_skill_report.csv",
    index=False
)


# --------------------------------------------------
# 3. Skill frequency by career role
# --------------------------------------------------

role_skill_records = []

for role in df["search_role"].unique():

    role_df = df[
        df["search_role"] == role
    ]

    role_skill_counts = {}

    for skills in role_df["skills"]:

        for skill in skills:

            role_skill_counts[skill] = (
                role_skill_counts.get(skill, 0) + 1
            )

    for skill, count in role_skill_counts.items():

        percentage = (
            count / len(role_df) * 100
        )

        role_skill_records.append({
            "role": role,
            "skill": skill,
            "job_count": count,
            "percentage": round(
                percentage,
                2
            )
        })


role_skill_report = pd.DataFrame(
    role_skill_records
)


role_skill_report.to_csv(
    f"{OUTPUT_DIR}/role_skill_report.csv",
    index=False
)


# --------------------------------------------------
# 4. Display top skills for each role
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP SKILLS BY CAREER ROLE")
print("=" * 60)


for role in sorted(
    df["search_role"].unique()
):

    print(f"\n{role.upper()}")

    role_data = (
        role_skill_report[
            role_skill_report["role"] == role
        ]
        .sort_values(
            "job_count",
            ascending=False
        )
        .head(10)
    )

    for _, row in role_data.iterrows():

        print(
            f"  {row['skill']}: "
            f"{row['job_count']} jobs "
            f"({row['percentage']}%)"
        )


# --------------------------------------------------
# 5. Create role × skill matrix
# --------------------------------------------------

role_skill_matrix = (
    role_skill_report
    .pivot_table(
        index="role",
        columns="skill",
        values="percentage",
        fill_value=0
    )
)


role_skill_matrix.to_csv(
    f"{OUTPUT_DIR}/role_skill_matrix.csv"
)


# --------------------------------------------------
# 6. Top 15 skills visualization
# --------------------------------------------------

top_skills = (
    skill_frequency
    .head(15)
    .sort_values()
)


plt.figure(figsize=(10, 7))

top_skills.plot(
    kind="barh"
)

plt.title(
    "Top 15 Skills in Job Market"
)

plt.xlabel(
    "Number of Jobs"
)

plt.ylabel(
    "Skill"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/top_skills.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# 7. Top skills for each role
# --------------------------------------------------

for role in sorted(
    df["search_role"].unique()
):

    role_data = (
        role_skill_report[
            role_skill_report["role"] == role
        ]
        .sort_values(
            "job_count",
            ascending=False
        )
        .head(10)
        .sort_values("job_count")
    )

    if role_data.empty:
        continue

    plt.figure(figsize=(10, 6))

    plt.barh(
        role_data["skill"],
        role_data["job_count"]
    )

    plt.title(
        f"Top Skills - {role.title()}"
    )

    plt.xlabel(
        "Number of Jobs"
    )

    plt.ylabel(
        "Skill"
    )

    plt.tight_layout()

    safe_role = (
        role.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_role}_skills.png",
        dpi=150
    )

    plt.show()


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print("SKILL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nReports saved to:",
    OUTPUT_DIR
)
