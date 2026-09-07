import ast
import os
import pandas as pd


JOBS_PATH = "data/processed/jobs_with_skills.csv"
ROLE_SKILLS_PATH = "data/processed/skill_analysis/role_skill_report.csv"
OUTPUT_PATH = "data/processed/recommendations/job_recommendations.csv"

TARGET_ROLE = "data scientist"


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - JOB RECOMMENDATION V4.1")
print("=" * 60)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

jobs_df = pd.read_csv(JOBS_PATH)
role_skills_df = pd.read_csv(ROLE_SKILLS_PATH)

print("\nJob dataset:", jobs_df.shape)
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
# USER PROFILE
# ---------------------------------------------------------

user_skills = {
    "python",
    "sql",
    "pandas"
}


print("\nTarget role:")
print(TARGET_ROLE.title())

print("\nUser skills:")

for skill in sorted(user_skills):
    print(f"  ✓ {skill.title()}")


# ---------------------------------------------------------
# TARGET ROLE SKILL PROFILE
# ---------------------------------------------------------

target_skills = role_skills_df[
    role_skills_df["role"].str.lower() == TARGET_ROLE.lower()
].copy()


# Only consider skills appearing in at least 5%
# of jobs for the target role.
target_skills = target_skills[
    target_skills["percentage"] >= 5
]


target_skill_weights = {
    row["skill"].lower(): row["percentage"]
    for _, row in target_skills.iterrows()
}


print("\nTarget-role skill profile:")

for skill, weight in sorted(
    target_skill_weights.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(
        f"  {skill.title()}: "
        f"{weight:.2f}%"
    )


# ---------------------------------------------------------
# ROLE RELEVANCE
# ---------------------------------------------------------

ROLE_RELEVANCE = {
    "data scientist": 100,
    "machine learning engineer": 90,
    "ai engineer": 85,
    "data analyst": 60,
    "data engineer": 55,
    "business analyst": 35
}


def calculate_role_relevance(job_role):

    role = str(job_role).lower().strip()

    return ROLE_RELEVANCE.get(role, 30)


# ---------------------------------------------------------
# EVIDENCE CONFIDENCE
# ---------------------------------------------------------

def calculate_evidence_confidence(skill_count):

    if skill_count == 0:
        return 0

    if skill_count == 1:
        return 40

    if skill_count == 2:
        return 55

    if skill_count == 3:
        return 70

    if skill_count == 4:
        return 80

    return 100


# ---------------------------------------------------------
# JOB SKILL MATCH
# ---------------------------------------------------------

def calculate_skill_match(job_skills):

    job_skills_normalized = {
        skill.lower()
        for skill in job_skills
    }


    # Skills detected in this job that are also
    # part of the target-role skill profile.
    relevant_job_skills = (
        job_skills_normalized
        & set(target_skill_weights.keys())
    )


    if not relevant_job_skills:

        return {
            "skill_match": 0,
            "matched_skills": [],
            "detected_missing_skills": []
        }


    matched_skills = (
        relevant_job_skills
        & user_skills
    )


    # -----------------------------------------------------
    # DEMAND-WEIGHTED MATCH
    # -----------------------------------------------------

    total_target_weight = sum(
        target_skill_weights.values()
    )


    matched_weight = sum(
        target_skill_weights[skill]
        for skill in matched_skills
    )


    demand_match = (
        matched_weight
        / total_target_weight
        * 100
    )


    # -----------------------------------------------------
    # JOB-SPECIFIC COVERAGE
    # -----------------------------------------------------

    job_weight = sum(
        target_skill_weights[skill]
        for skill in relevant_job_skills
    )


    if job_weight > 0:

        job_coverage = (
            matched_weight
            / job_weight
            * 100
        )

    else:

        job_coverage = 0


    # -----------------------------------------------------
    # COMBINED SKILL SCORE
    # -----------------------------------------------------

    skill_match = (
        demand_match * 0.70
        +
        job_coverage * 0.30
    )


    # These are skills actually detected in the job
    # but not present in the user's profile.
    detected_missing_skills = (
        relevant_job_skills
        - user_skills
    )


    return {
        "skill_match": skill_match,
        "matched_skills": sorted(
            matched_skills
        ),
        "detected_missing_skills": sorted(
            detected_missing_skills
        )
    }


# ---------------------------------------------------------
# PROCESS JOBS
# ---------------------------------------------------------

recommendations = []


for _, row in jobs_df.iterrows():

    job_skills = row["skills"]


    skill_result = calculate_skill_match(
        job_skills
    )


    role_relevance = calculate_role_relevance(
        row["search_role"]
    )


    evidence_confidence = (
        calculate_evidence_confidence(
            len(job_skills)
        )
    )


    # -----------------------------------------------------
    # FINAL RECOMMENDATION SCORE
    # -----------------------------------------------------
    #
    # 45% -> skill match
    # 40% -> target role relevance
    # 15% -> evidence confidence
    #

    final_score = (
        skill_result["skill_match"] * 0.45
        +
        role_relevance * 0.40
        +
        evidence_confidence * 0.15
    )


    recommendations.append({

        "job_id": row["id"],

        "title": row["title"],

        "company": row["company"],

        "location": row["location"],

        "search_role": row["search_role"],

        "recommendation_score":
            final_score,

        "skill_match":
            skill_result["skill_match"],

        "role_relevance":
            role_relevance,

        "evidence_confidence":
            evidence_confidence,

        "detected_skill_count":
            len(job_skills),

        "matched_skills":
            ", ".join(
                skill_result["matched_skills"]
            ),

        "detected_missing_skills":
            ", ".join(
                skill_result[
                    "detected_missing_skills"
                ]
            )
    })


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

recommendations_df = pd.DataFrame(
    recommendations
)


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

recommendations_df = (
    recommendations_df
    .sort_values(
        "recommendation_score",
        ascending=False
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# RANK
# ---------------------------------------------------------

recommendations_df[
    "recommendation_rank"
] = (
    recommendations_df.index + 1
)


# ---------------------------------------------------------
# TOP RECOMMENDATIONS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOP JOB RECOMMENDATIONS")
print("=" * 60)


for _, row in recommendations_df.head(10).iterrows():

    print(
        f"\n#{int(row['recommendation_rank'])} "
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
        f"   Recommendation score: "
        f"{row['recommendation_score']:.1f}%"
    )

    print(
        f"   Skill match: "
        f"{row['skill_match']:.1f}%"
    )

    print(
        f"   Role relevance: "
        f"{row['role_relevance']:.1f}%"
    )

    print(
        f"   Evidence confidence: "
        f"{row['evidence_confidence']:.1f}%"
    )

    print(
        f"   Detected skills: "
        f"{int(row['detected_skill_count'])}"
    )

    print(
        f"   Matched skills: "
        f"{row['matched_skills'] or 'None'}"
    )

    print(
        f"   Detected missing skills: "
        f"{row['detected_missing_skills'] or 'None'}"
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RECOMMENDATION SUMMARY")
print("=" * 60)


print(
    f"Jobs analyzed: "
    f"{len(jobs_df)}"
)


print(
    f"Jobs with detected skills: "
    f"{(jobs_df['skills'].apply(len) > 0).sum()}"
)


print(
    f"Average recommendation score: "
    f"{recommendations_df['recommendation_score'].mean():.1f}%"
)


print(
    f"Best recommendation score: "
    f"{recommendations_df['recommendation_score'].max():.1f}%"
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

os.makedirs(
    "data/processed/recommendations",
    exist_ok=True
)


recommendations_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nRecommendations saved to:"
    f" {OUTPUT_PATH}"
)


print(
    "\nJob Recommendation V4.1 completed successfully."
)