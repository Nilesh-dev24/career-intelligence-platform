import os
import pandas as pd

INPUT_PATH = "data/processed/skill_analysis/role_skill_report.csv"
OUTPUT_PATH = "data/processed/skill_analysis/skill_gap_report.csv"


print("=" * 60)
print("CAREER INTELLIGENCE PLATFORM - SKILL GAP ENGINE V2")
print("=" * 60)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nSkill report loaded.")
print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# GET SKILLS REQUIRED FOR A ROLE
# ---------------------------------------------------------

def get_role_skills(role, minimum_percentage=5):

    role_data = df[
        df["role"].str.lower() == role.lower()
    ].copy()

    if role_data.empty:
        return pd.DataFrame()

    role_data = role_data[
        role_data["percentage"] >= minimum_percentage
    ]

    role_data = role_data.sort_values(
        "percentage",
        ascending=False
    )

    return role_data[
        ["skill", "job_count", "percentage"]
    ]


# ---------------------------------------------------------
# CALCULATE SKILL GAP
# ---------------------------------------------------------

def calculate_skill_gap(
    role,
    user_skills,
    minimum_percentage=5
):

    role_skills = get_role_skills(
        role=role,
        minimum_percentage=minimum_percentage
    )

    if role_skills.empty:
        print(f"\nNo skill data found for role: {role}")
        return None

    # Normalize user skills
    user_skills_normalized = {
        skill.strip().lower()
        for skill in user_skills
    }

    # Determine whether user has each skill
    role_skills["status"] = role_skills["skill"].apply(
        lambda skill:
        "Have"
        if skill.lower() in user_skills_normalized
        else "Missing"
    )

    # Skill importance is based on job percentage
    role_skills["priority_score"] = role_skills["percentage"]

    # Assign priority level
    def get_priority(percentage):

        if percentage >= 15:
            return "High"

        elif percentage >= 8:
            return "Medium"

        else:
            return "Low"

    role_skills["priority"] = (
        role_skills["percentage"]
        .apply(get_priority)
    )

    # Missing skills first
    role_skills["status_order"] = (
        role_skills["status"]
        .map({
            "Missing": 0,
            "Have": 1
        })
    )

    # Sort missing skills by importance
    role_skills = role_skills.sort_values(
        ["status_order", "priority_score"],
        ascending=[True, False]
    )

    role_skills = role_skills.drop(
        columns=["status_order"]
    )

    return role_skills


# ---------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------

user_skills = [
    "Python",
    "SQL",
    "Pandas"
]

target_role = "data scientist"


print("\nTarget career:")
print(target_role.title())

print("\nUser skills:")

for skill in user_skills:
    print(f"  ✓ {skill}")


# ---------------------------------------------------------
# RUN SKILL GAP ANALYSIS
# ---------------------------------------------------------

result = calculate_skill_gap(
    role=target_role,
    user_skills=user_skills,
    minimum_percentage=5
)


if result is not None:

    # -----------------------------------------------------
    # DISPLAY COMPLETE ANALYSIS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SKILL GAP ANALYSIS")
    print("=" * 60)

    display_columns = [
        "skill",
        "job_count",
        "percentage",
        "status",
        "priority"
    ]

    print(
        result[display_columns].to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # MISSING SKILLS
    # -----------------------------------------------------

    missing = result[
        result["status"] == "Missing"
    ].copy()

    missing = missing.reset_index(
        drop=True
    )


    print("\n" + "=" * 60)
    print("PRIORITIZED SKILL GAPS")
    print("=" * 60)


    if missing.empty:

        print(
            "Excellent! No major skill gaps detected."
        )

    else:

        for index, row in missing.iterrows():

            print(
                f"{index + 1}. "
                f"{row['skill']} "
                f"- {row['priority']} priority "
                f"({row['percentage']:.2f}% of jobs)"
            )


    # -----------------------------------------------------
    # CAREER READINESS
    # -----------------------------------------------------

    total_skills = len(result)

    matched_skills = len(
        result[
            result["status"] == "Have"
        ]
    )

    match_percentage = (
        matched_skills /
        total_skills *
        100
    )


    print("\n" + "=" * 60)
    print("CAREER READINESS")
    print("=" * 60)

    print(
        f"Skill match: "
        f"{matched_skills}/{total_skills}"
    )

    print(
        f"Readiness score: "
        f"{match_percentage:.1f}%"
    )


    # -----------------------------------------------------
    # PRIORITY SUMMARY
    # -----------------------------------------------------

    high_priority = len(
        missing[
            missing["priority"] == "High"
        ]
    )

    medium_priority = len(
        missing[
            missing["priority"] == "Medium"
        ]
    )

    low_priority = len(
        missing[
            missing["priority"] == "Low"
        ]
    )


    print("\n" + "=" * 60)
    print("GAP SUMMARY")
    print("=" * 60)

    print(
        f"High priority gaps: {high_priority}"
    )

    print(
        f"Medium priority gaps: {medium_priority}"
    )

    print(
        f"Low priority gaps: {low_priority}"
    )


    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------

    os.makedirs(
        "data/processed/skill_analysis",
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSkill gap report saved to:"
        f" {OUTPUT_PATH}"
    )


print("\nSkill Gap Engine V2 completed successfully.")