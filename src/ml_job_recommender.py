import ast
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


JOBS_PATH = "data/processed/jobs_with_skills.csv"
ROLE_SKILLS_PATH = "data/processed/skill_analysis/role_skill_report.csv"
OUTPUT_PATH = "data/processed/recommendations/ml_job_recommendations.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    jobs_df = pd.read_csv(JOBS_PATH)
    role_skills_df = pd.read_csv(ROLE_SKILLS_PATH)

    jobs_df["skills"] = jobs_df["skills"].apply(
        lambda x: ast.literal_eval(x)
        if isinstance(x, str)
        else []
    )

    return jobs_df, role_skills_df


# ---------------------------------------------------------
# CREATE WEIGHTED JOB TEXT
# ---------------------------------------------------------

def create_job_text(row):
    title = str(row["title"])
    role = str(row["search_role"])

    skills = " ".join(
        str(skill)
        for skill in row["skills"]
    )

    description = str(row["description"])

    return (
        title + " "
        + title + " "
        + role + " "
        + role + " "
        + skills + " "
        + skills + " "
        + description
    )


# ---------------------------------------------------------
# NORMALIZE SKILLS
# ---------------------------------------------------------

def normalize_skills(skills):
    return {
        str(skill).lower().strip()
        for skill in skills
        if str(skill).strip()
    }


def get_normalized_job_skills(job_skills):
    return normalize_skills(job_skills)


# ---------------------------------------------------------
# TARGET ROLE SKILL PROFILE
# ---------------------------------------------------------

def get_skill_weights(role_skills_df, target_role):
    target_role_skills = role_skills_df[
        role_skills_df["role"].astype(str).str.lower()
        == target_role.lower()
    ].copy()

    target_role_skills = target_role_skills[
        target_role_skills["percentage"] >= 5
    ]

    return {
        row["skill"].lower().strip(): row["percentage"]
        for _, row in target_role_skills.iterrows()
    }


# ---------------------------------------------------------
# MAIN RECOMMENDATION FUNCTION
# ---------------------------------------------------------

def recommend_jobs(
    target_role,
    user_skills,
    top_k=10,
    save_output=False
):
    """
    Generate personalized job recommendations.

    Parameters
    ----------
    target_role : str
        Career role the user is targeting.

    user_skills : list/set
        Skills currently possessed by the user.

    top_k : int
        Number of recommendations to return.

    save_output : bool
        Whether to save the recommendations CSV.

    Returns
    -------
    pandas.DataFrame
        Ranked job recommendations.
    """

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    jobs_df, role_skills_df = load_data()

    # -----------------------------------------------------
    # NORMALIZE USER PROFILE
    # -----------------------------------------------------

    target_role = str(target_role).lower().strip()

    user_skills = normalize_skills(user_skills)

    # -----------------------------------------------------
    # TARGET ROLE SKILL WEIGHTS
    # -----------------------------------------------------

    skill_weights = get_skill_weights(
        role_skills_df,
        target_role
    )

    total_skill_demand = sum(
        skill_weights.values()
    )

    # -----------------------------------------------------
    # CREATE WEIGHTED JOB TEXT
    # -----------------------------------------------------

    jobs_df["job_text"] = jobs_df.apply(
        create_job_text,
        axis=1
    )

    skill_text = " ".join(
        sorted(user_skills)
    )

    user_profile = (
        target_role + " "
        + target_role + " "
        + skill_text + " "
        + skill_text
    )

    # -----------------------------------------------------
    # CONTENT TF-IDF
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ROLE TF-IDF
    # -----------------------------------------------------

    role_documents = (
        jobs_df["title"].fillna("").astype(str)
        + " "
        + jobs_df["search_role"].fillna("").astype(str)
    ).tolist()

    role_documents.append(
        target_role
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

    # -----------------------------------------------------
    # SKILL MATCHING
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # DEMAND-WEIGHTED SKILL MATCH
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # JOB-SPECIFIC SKILL FIT
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # FINAL HYBRID ML SCORE
    # -----------------------------------------------------
    #
    # 45% -> content similarity
    # 25% -> role similarity
    # 20% -> target-role skill match
    # 10% -> job-specific skill fit
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

    # -----------------------------------------------------
    # EXPLAINABILITY
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # SORT AND RANK
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # OUTPUT COLUMNS
    # -----------------------------------------------------

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
        "description",
        "redirect_url"
    ]

    available_columns = [
        column
        for column in output_columns
        if column in recommendations.columns
    ]

    recommendations_output = recommendations[
        available_columns
    ].copy()

    # -----------------------------------------------------
    # TOP K
    # -----------------------------------------------------

    recommendations_output = (
        recommendations_output
        .head(top_k)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # SAVE OPTIONAL OUTPUT
    # -----------------------------------------------------

    if save_output:

        os.makedirs(
            "data/processed/recommendations",
            exist_ok=True
        )

        recommendations_output.to_csv(
            OUTPUT_PATH,
            index=False
        )

    return recommendations_output


# ---------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(
        "CAREER INTELLIGENCE PLATFORM "
        "- ML JOB RECOMMENDER"
    )
    print("=" * 60)

    default_role = "data scientist"

    default_skills = {
        "python",
        "sql",
        "pandas"
    }

    print("\nTarget role:")
    print(default_role.title())

    print("\nCandidate skills:")
    print(
        ", ".join(
            sorted(default_skills)
        )
    )

    recommendations = recommend_jobs(
        target_role=default_role,
        user_skills=default_skills,
        top_k=10,
        save_output=True
    )

    print("\n" + "=" * 60)
    print("TOP ML JOB RECOMMENDATIONS")
    print("=" * 60)

    for _, row in recommendations.iterrows():

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

    print("\n" + "=" * 60)
    print("ML RECOMMENDER SUMMARY")
    print("=" * 60)

    print(
        f"Jobs analyzed: "
        f"{len(recommendations)}"
    )

    print(
        f"Average recommendation score: "
        f"{recommendations['ml_recommendation_score'].mean():.2f}%"
    )

    print(
        f"Best recommendation score: "
        f"{recommendations['ml_recommendation_score'].max():.2f}%"
    )

    print(
        f"\nML recommendations saved to: "
        f"{OUTPUT_PATH}"
    )