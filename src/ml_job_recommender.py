import ast
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


JOBS_PATH = "data/processed/jobs_with_skills.csv"
ROLE_SKILLS_PATH = "data/processed/skill_analysis/role_skill_report.csv"
OUTPUT_PATH = "data/processed/recommendations/ml_job_recommendations.csv"

TARGET_ROLE = "data scientist"

USER_SKILLS = {
    "python",
    "sql",
    "pandas"
}


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - ML JOB RECOMMENDER V3")
print("=" * 60)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

jobs_df = pd.read_csv(JOBS_PATH)
role_skills_df = pd.read_csv(ROLE_SKILLS_PATH)

print("\nJobs dataset:", jobs_df.shape)
print("Role skill report:", role_skills_df.shape)


# ---------------------------------------------------------
# CONVERT SKILLS
# ---------------------------------------------------------

jobs_df["skills"] = jobs_df["skills"].apply(
    lambda x: ast.literal_eval(x)
    if isinstance(x, str)
    else []
)


# ---------------------------------------------------------
# NORMALIZE USER SKILLS
# ---------------------------------------------------------

user_skills = {
    skill.lower().strip()
    for skill in USER_SKILLS
}


# ---------------------------------------------------------
# TARGET ROLE SKILL PROFILE
# ---------------------------------------------------------
#
# We use the skills discovered from the actual job
# dataset for the target career.
#
# This gives frequently requested skills more importance.
#

target_role_skills = role_skills_df[
    role_skills_df["role"].str.lower() == TARGET_ROLE.lower()
].copy()


target_role_skills = target_role_skills[
    target_role_skills["percentage"] >= 5
]


skill_weights = {
    row["skill"].lower(): row["percentage"]
    for _, row in target_role_skills.iterrows()
}


print("\nTarget role:")
print(TARGET_ROLE.title())


print("\nTarget-role skill weights:")

for skill, weight in sorted(
    skill_weights.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(
        f"  {skill.title()}: "
        f"{weight:.2f}%"
    )


# ---------------------------------------------------------
# CREATE WEIGHTED JOB TEXT
# ---------------------------------------------------------
#
# Important job information is repeated so TF-IDF gives
# stronger representation to:
#
#   title
#   role
#   detected skills
#
# The description is still included.
#


def create_job_text(row):

    title = str(row["title"])
    role = str(row["search_role"])

    skills = " ".join(
        str(skill)
        for skill in row["skills"]
    )

    description = str(
        row["description"]
    )

    return (
        title + " "
        + title + " "
        + role + " "
        + role + " "
        + skills + " "
        + skills + " "
        + description
    )


jobs_df["job_text"] = jobs_df.apply(
    create_job_text,
    axis=1
)


# ---------------------------------------------------------
# CREATE USER PROFILE
# ---------------------------------------------------------

skill_text = " ".join(
    sorted(USER_SKILLS)
)


user_profile = (
    TARGET_ROLE + " "
    + TARGET_ROLE + " "
    + skill_text + " "
    + skill_text
)


print("\nCandidate profile:")
print(user_profile)


# ---------------------------------------------------------
# CONTENT TF-IDF
# ---------------------------------------------------------

print("\nBuilding content TF-IDF model...")


content_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)


content_documents = jobs_df[
    "job_text"
].tolist()


content_documents.append(
    user_profile
)


content_matrix = content_vectorizer.fit_transform(
    content_documents
)


job_content_vectors = content_matrix[:-1]
user_content_vector = content_matrix[-1]


content_similarity = cosine_similarity(
    user_content_vector,
    job_content_vectors
).flatten()


jobs_df["content_similarity"] = (
    content_similarity * 100
)


print(
    "Content vocabulary size:",
    len(content_vectorizer.vocabulary_)
)


# ---------------------------------------------------------
# ROLE TF-IDF
# ---------------------------------------------------------
#
# Separate model for role matching.
#

print("\nBuilding role similarity model...")


role_documents = (
    jobs_df["title"].fillna("").astype(str)
    + " "
    + jobs_df["search_role"].fillna("").astype(str)
)


role_documents = role_documents.tolist()

role_documents.append(
    TARGET_ROLE
)


role_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)


role_matrix = role_vectorizer.fit_transform(
    role_documents
)


job_role_vectors = role_matrix[:-1]
target_role_vector = role_matrix[-1]


role_similarity = cosine_similarity(
    target_role_vector,
    job_role_vectors
).flatten()


jobs_df["role_similarity"] = (
    role_similarity * 100
)


# ---------------------------------------------------------
# SKILL MATCHING
# ---------------------------------------------------------

def get_normalized_job_skills(job_skills):

    return {
        skill.lower().strip()
        for skill in job_skills
    }


def get_matched_skills(job_skills):

    normalized_job_skills = (
        get_normalized_job_skills(job_skills)
    )

    return (
        normalized_job_skills
        & user_skills
    )


def get_missing_skills(job_skills):

    normalized_job_skills = (
        get_normalized_job_skills(job_skills)
    )

    return (
        normalized_job_skills
        - user_skills
    )


# ---------------------------------------------------------
# DEMAND-WEIGHTED SKILL MATCH
# ---------------------------------------------------------
#
# Instead of:
#
#     matched skills / detected skills
#
# we calculate how much of the target role's
# skill demand is covered by the candidate.
#
# Example:
#
# Python = 25%
# SQL = 11.67%
#
# Candidate has both:
#
# 36.67% demand covered
#
# This is more meaningful than simply saying 2/5.
#


total_skill_demand = sum(
    skill_weights.values()
)


def calculate_skill_match(job_skills):

    matched = get_matched_skills(
        job_skills
    )

    if total_skill_demand == 0:
        return 0

    matched_demand = sum(
        skill_weights.get(
            skill,
            0
        )
        for skill in matched
    )

    return (
        matched_demand
        / total_skill_demand
        * 100
    )


jobs_df["skill_match"] = jobs_df[
    "skills"
].apply(
    calculate_skill_match
)


# ---------------------------------------------------------
# JOB-SPECIFIC SKILL FIT
# ---------------------------------------------------------
#
# This measures whether the candidate's skills match
# the skills detected in this particular job.
#
# Demand weighting is used when the skill is part of
# the target-role profile.
#


def calculate_job_skill_fit(job_skills):

    normalized_job_skills = (
        get_normalized_job_skills(
            job_skills
        )
    )

    relevant_job_skills = (
        normalized_job_skills
        & set(skill_weights.keys())
    )

    if not relevant_job_skills:
        return 0

    matched_skills = (
        relevant_job_skills
        & user_skills
    )

    job_demand = sum(
        skill_weights[skill]
        for skill in relevant_job_skills
    )

    matched_demand = sum(
        skill_weights[skill]
        for skill in matched_skills
    )

    if job_demand == 0:
        return 0

    return (
        matched_demand
        / job_demand
        * 100
    )


jobs_df["job_skill_fit"] = jobs_df[
    "skills"
].apply(
    calculate_job_skill_fit
)


# ---------------------------------------------------------
# FINAL HYBRID ML SCORE
# ---------------------------------------------------------
#
# Main signals:
#
# 45% -> content similarity
# 25% -> role similarity
# 20% -> target-role skill match
# 10% -> job-specific skill fit
#
# The first two are ML/NLP signals.
# The skill signals make the recommendation
# explainable and candidate-specific.
#

jobs_df["ml_recommendation_score"] = (
    jobs_df["content_similarity"] * 0.45
    +
    jobs_df["role_similarity"] * 0.25
    +
    jobs_df["skill_match"] * 0.20
    +
    jobs_df["job_skill_fit"] * 0.10
)


# ---------------------------------------------------------
# EXPLAINABILITY COLUMNS
# ---------------------------------------------------------

jobs_df["matched_skills"] = jobs_df[
    "skills"
].apply(
    lambda skills: ", ".join(
        sorted(
            get_matched_skills(skills)
        )
    )
)


jobs_df["missing_skills"] = jobs_df[
    "skills"
].apply(
    lambda skills: ", ".join(
        sorted(
            get_missing_skills(skills)
        )
    )
)


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

recommendations = (
    jobs_df
    .sort_values(
        "ml_recommendation_score",
        ascending=False
    )
    .reset_index(drop=True)
)


recommendations[
    "ml_recommendation_rank"
] = (
    recommendations.index + 1
)


# ---------------------------------------------------------
# SAVE COLUMNS
# ---------------------------------------------------------

output_columns = [
    "ml_recommendation_rank",
    "ml_recommendation_score",
    "content_similarity",
    "role_similarity",
    "skill_match",
    "job_skill_fit",
    "title",
    "company",
    "location",
    "search_role",
    "matched_skills",
    "missing_skills",
    "skills",
    "description"
]


recommendations_output = recommendations[
    output_columns
].copy()


# ---------------------------------------------------------
# DISPLAY TOP 10
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOP ML JOB RECOMMENDATIONS")
print("=" * 60)


for _, row in recommendations.head(10).iterrows():

    print(
        f"\n#{int(row['ml_recommendation_rank'])} "
        f"{row['title']}"
    )

    print(
        f"   Company: "
        f"{row['company']}"
    )

    print(
        f"   Location: "
        f"{row['location']}"
    )

    print(
        f"   Role: "
        f"{row['search_role']}"
    )

    print(
        f"   ML recommendation score: "
        f"{row['ml_recommendation_score']:.2f}%"
    )

    print(
        f"   Content similarity: "
        f"{row['content_similarity']:.2f}%"
    )

    print(
        f"   Role similarity: "
        f"{row['role_similarity']:.2f}%"
    )

    print(
        f"   Target-role skill match: "
        f"{row['skill_match']:.2f}%"
    )

    print(
        f"   Job-specific skill fit: "
        f"{row['job_skill_fit']:.2f}%"
    )

    print(
        f"   Matched skills: "
        f"{row['matched_skills'] or 'None'}"
    )

    print(
        f"   Missing detected skills: "
        f"{row['missing_skills'] or 'None'}"
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ML RECOMMENDER SUMMARY")
print("=" * 60)


print(
    f"Jobs analyzed: "
    f"{len(recommendations)}"
)


print(
    f"Average content similarity: "
    f"{recommendations['content_similarity'].mean():.2f}%"
)


print(
    f"Average role similarity: "
    f"{recommendations['role_similarity'].mean():.2f}%"
)


print(
    f"Average target-role skill match: "
    f"{recommendations['skill_match'].mean():.2f}%"
)


print(
    f"Average job-specific skill fit: "
    f"{recommendations['job_skill_fit'].mean():.2f}%"
)


print(
    f"Average ML recommendation score: "
    f"{recommendations['ml_recommendation_score'].mean():.2f}%"
)


print(
    f"Best ML recommendation score: "
    f"{recommendations['ml_recommendation_score'].max():.2f}%"
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

os.makedirs(
    "data/processed/recommendations",
    exist_ok=True
)


recommendations_output.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nML recommendations saved to:"
    f" {OUTPUT_PATH}"
)


print(
    "\nML Job Recommender V3 completed successfully."
)