import os
import re
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/clean_jobs.csv"
OUTPUT_PATH = "data/processed/jobs_with_skills.csv"


# --------------------------------------------------
# Skill taxonomy
# --------------------------------------------------

SKILLS = {
    "Python": [
        "python"
    ],

    "SQL": [
        "sql"
    ],

    "R": [
        "r programming",
        "r language",
        r"\br\b"
    ],

    "Java": [
        r"\bjava\b"
    ],

    "C++": [
        r"\bc\+\+\b"
    ],

    "C#": [
        r"\bc#\b",
        "c sharp"
    ],

    "Pandas": [
        "pandas"
    ],

    "NumPy": [
        "numpy"
    ],

    "Scikit-learn": [
        "scikit-learn",
        "scikit learn"
    ],

    "TensorFlow": [
        "tensorflow"
    ],

    "PyTorch": [
        "pytorch"
    ],

    "Machine Learning": [
        "machine learning",
        "machine-learning"
    ],

    "Deep Learning": [
        "deep learning",
        "deep-learning"
    ],

    "NLP": [
        "natural language processing",
        r"\bnlp\b"
    ],

    "Computer Vision": [
        "computer vision"
    ],

    "Generative AI": [
        "generative ai",
        r"\bgenai\b",
        r"\bgen ai\b"
    ],

    "LLM": [
        "large language model",
        "large language models",
        r"\bllm\b",
        r"\bllms\b"
    ],

    "AWS": [
        r"\baws\b",
        "amazon web services"
    ],

    "Azure": [
        r"\bazure\b",
        "microsoft azure"
    ],

    "GCP": [
        r"\bgcp\b",
        "google cloud",
        "google cloud platform"
    ],

    "Docker": [
        r"\bdocker\b"
    ],

    "Kubernetes": [
        r"\bkubernetes\b"
    ],

    "Apache Spark": [
        "apache spark",
        r"\bspark\b"
    ],

    "Hadoop": [
        r"\bhadoop\b"
    ],

    "Databricks": [
        "databricks"
    ],

    "Git": [
        r"\bgit\b",
        r"\bgithub\b",
        r"\bgitlab\b",
        r"\bbitbucket\b",
        r"\bgit\s*/\s*github\b",
        r"\bgithub\s*/\s*git\b",
        r"\bgit\s+github\b",
        r"\bgithub\s+git\b"
    ],

    "Power BI": [
        "power bi",
        "powerbi"
    ],

    "Tableau": [
        "tableau"
    ],

    "Excel": [
        r"\bexcel\b",
        "microsoft excel"
    ],

    "Statistics": [
        "statistics",
        "statistical analysis",
        "statistical modeling"
    ],

    "Data Visualization": [
        "data visualization",
        "data visualisation"
    ],

    "ETL": [
        r"\betl\b",
        "extract transform load",
        "extract, transform, load"
    ],

    "Airflow": [
        r"\bairflow\b",
        "apache airflow"
    ],

    "Kafka": [
        r"\bkafka\b",
        "apache kafka"
    ],

    "Snowflake": [
        "snowflake"
    ],

    "BigQuery": [
        "bigquery",
        "big query"
    ]
}


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - SKILL EXTRACTION V2")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Skill extraction function
# --------------------------------------------------

def extract_skills(text):
    """
    Extract skills from a job description
    using boundary-aware pattern matching.
    """

    if pd.isna(text):
        return []

    text = str(text).lower()

    found_skills = []

    for skill, patterns in SKILLS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):
                found_skills.append(skill)
                break

    return found_skills


# --------------------------------------------------
# Extract skills
# --------------------------------------------------

print("\nExtracting skills...")

df["skills"] = df["description"].apply(
    extract_skills
)


# --------------------------------------------------
# Skill count per job
# --------------------------------------------------

df["skill_count"] = (
    df["skills"].apply(len)
)


print("\nSkill extraction completed.")

print("\nSkill count statistics:")

print(
    df["skill_count"].describe()
)


# --------------------------------------------------
# Show examples
# --------------------------------------------------

print("\nExample extracted skills:")

for _, row in df.head(10).iterrows():

    print("-" * 50)

    print("Job:", row["title"])

    print("Role:", row["search_role"])

    print("Skills:", row["skills"])


# --------------------------------------------------
# Skill frequency
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


print("\nTop skills:")

print(
    skill_frequency.head(20)
)


# --------------------------------------------------
# Save skill frequency
# --------------------------------------------------

skill_frequency_df = (
    skill_frequency
    .reset_index()
)

skill_frequency_df.columns = [
    "skill",
    "job_count"
]

os.makedirs(
    "data/processed",
    exist_ok=True
)

skill_frequency_df.to_csv(
    "data/processed/skill_frequency.csv",
    index=False
)


# --------------------------------------------------
# Save dataset with extracted skills
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nDataset with skills saved to: {OUTPUT_PATH}"
)

print(
    "Skill frequency saved to: "
    "data/processed/skill_frequency.csv"
)


print("\nSkill extraction V2 completed successfully.")