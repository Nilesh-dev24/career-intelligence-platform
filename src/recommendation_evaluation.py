import pandas as pd


INPUT_PATH = (
    "data/processed/recommendations/"
    "ml_job_recommendations.csv"
)

OUTPUT_PATH = (
    "data/processed/recommendations/"
    "recommendation_evaluation.csv"
)

TARGET_ROLE = "data scientist"


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM -")
print("RECOMMENDATION EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# LOAD RECOMMENDATIONS
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nRecommendation dataset loaded.")
print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# DEFINE ROLE RELEVANCE
# ---------------------------------------------------------
#
# This is an evaluation signal, NOT part of the model.
#
# We use the existing search_role field to determine
# whether a recommendation belongs to the target career.
#

def is_target_role(role):

    if pd.isna(role):
        return False

    role = str(role).lower().strip()

    return role == TARGET_ROLE.lower()


df["is_target_role"] = df[
    "search_role"
].apply(is_target_role)


# ---------------------------------------------------------
# TOP-K ROLE RELEVANCE
# ---------------------------------------------------------
#
# Measures how many recommendations in the top K belong
# to the target career.
#
# Example:
#
# Top 10:
# 9 Data Scientist
# 1 other role
#
# Precision@10 = 90%
#

def calculate_precision_at_k(data, k):

    top_k = data.head(k)

    if len(top_k) == 0:
        return 0

    return (
        top_k["is_target_role"].sum()
        / len(top_k)
        * 100
    )


# ---------------------------------------------------------
# SKILL OVERLAP
# ---------------------------------------------------------
#
# We measure whether recommended jobs contain at least
# one skill the candidate already has.
#
# This provides an additional evaluation signal.
#

def has_skill_overlap(value):

    if pd.isna(value):
        return False

    value = str(value).strip()

    if value == "":
        return False

    return True


df["has_skill_overlap"] = df[
    "matched_skills"
].apply(has_skill_overlap)


def calculate_skill_overlap_at_k(data, k):

    top_k = data.head(k)

    if len(top_k) == 0:
        return 0

    return (
        top_k["has_skill_overlap"].sum()
        / len(top_k)
        * 100
    )


# ---------------------------------------------------------
# TOP-K EVALUATION
# ---------------------------------------------------------

k_values = [5, 10, 20, 50]


evaluation_rows = []


print("\n" + "=" * 60)
print("TOP-K EVALUATION")
print("=" * 60)


for k in k_values:

    precision = calculate_precision_at_k(
        df,
        k
    )

    skill_overlap = calculate_skill_overlap_at_k(
        df,
        k
    )

    actual_k = min(k, len(df))

    evaluation_rows.append({

        "k": k,

        "recommendation_count":
            actual_k,

        "target_role_precision":
            precision,

        "skill_overlap_rate":
            skill_overlap
    })


    print(
        f"\nPrecision@{k}: "
        f"{precision:.2f}%"
    )

    print(
        f"Skill overlap@{k}: "
        f"{skill_overlap:.2f}%"
    )


evaluation_df = pd.DataFrame(
    evaluation_rows
)


# ---------------------------------------------------------
# RANKING DISTRIBUTION
# ---------------------------------------------------------
#
# This helps us understand how recommendations are
# distributed across career roles.
#

print("\n" + "=" * 60)
print("ROLE DISTRIBUTION")
print("=" * 60)


role_distribution = (
    df["search_role"]
    .value_counts()
)


print(
    role_distribution.to_string()
)


# ---------------------------------------------------------
# TOP 10 DETAILS
# ---------------------------------------------------------

top_10 = df.head(10)


target_role_count = (
    top_10["is_target_role"].sum()
)


skill_overlap_count = (
    top_10["has_skill_overlap"].sum()
)


print("\n" + "=" * 60)
print("TOP 10 SUMMARY")
print("=" * 60)


print(
    f"Target-role jobs in top 10: "
    f"{target_role_count}/10"
)


print(
    f"Jobs with candidate skill overlap "
    f"in top 10: "
    f"{skill_overlap_count}/10"
)


print(
    f"Top-10 target-role precision: "
    f"{calculate_precision_at_k(df, 10):.2f}%"
)


print(
    f"Top-10 skill overlap rate: "
    f"{calculate_skill_overlap_at_k(df, 10):.2f}%"
)


# ---------------------------------------------------------
# SCORE STATISTICS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RECOMMENDATION SCORE STATISTICS")
print("=" * 60)


print(
    f"Average score: "
    f"{df['ml_recommendation_score'].mean():.2f}%"
)


print(
    f"Median score: "
    f"{df['ml_recommendation_score'].median():.2f}%"
)


print(
    f"Top-10 average score: "
    f"{df.head(10)['ml_recommendation_score'].mean():.2f}%"
)


print(
    f"Top-20 average score: "
    f"{df.head(20)['ml_recommendation_score'].mean():.2f}%"
)


# ---------------------------------------------------------
# SAVE EVALUATION
# ---------------------------------------------------------

evaluation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nEvaluation report saved to:"
    f" {OUTPUT_PATH}"
)


print(
    "\nRecommendation Evaluation completed successfully."
)