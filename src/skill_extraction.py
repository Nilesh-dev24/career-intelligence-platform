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
        r"\br\b",
        "r programming",
        "r language"
    ],

    "Java": [
        "java"
    ],

    "C++": [
        "c++"
    ],

    "C#": [
        "c#",
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
        "nlp"
    ],

    "Computer Vision": [
        "computer vision"
    ],

    "Generative AI": [
        "generative ai",
        "genai",
        "gen ai"
    ],

    "LLM": [
        "large language model",
        "large language models",
        "llm",
        "llms"
    ],

    "AWS": [
        "aws",
        "amazon web services"
    ],

    "Azure": [
        "azure",
        "microsoft azure"
    ],

    "GCP": [
        "gcp",
        "google cloud",
        "google cloud platform"
    ],

    "Docker": [
        "docker"
    ],

    "Kubernetes": [
        "kubernetes"
    ],

    "Apache Spark": [
        "apache spark",
        "spark"
    ],

    "Hadoop": [
        "hadoop"
    ],

    "Databricks": [
        "databricks"
    ],

    "Git": [
        "git",
        "github"
    ],

    "Power BI": [
        "power bi",
        "powerbi"
    ],

    "Tableau": [
        "tableau"
    ],

    "Excel": [
        "excel",
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
        "etl",
        "extract transform load",
        "extract, transform, load"
    ],

    "Airflow": [
        "airflow",
        "apache airflow"
    ],

    "Kafka": [
        "kafka",
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
print("CAREER INTELLIGENCE PLATFORM - SKILL EXTRACTION V1")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Skill extraction function
# --------------------------------------------------

def extract_skills(text):
    """
    Extract skills from a job description.

    Returns a list of matched skills.
    """

    if pd.isna(text):
        return []

    text = str(text).lower()

    found_skills = []

    for skill, patterns in SKILLS.items():

        for pattern in patterns:

            # Escape literal skill names so characters such as
            # +, #, and . are treated as normal characters.
            safe_pattern = re.escape(pattern)

            if re.search(
                safe_pattern,
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


print("\nSkill extraction completed successfully.")