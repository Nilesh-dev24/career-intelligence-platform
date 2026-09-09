import os
import pandas as pd


INPUT_PATH = "data/processed/skill_analysis/skill_gap_report.csv"
OUTPUT_PATH = "data/processed/skill_analysis/career_roadmap.csv"


# ---------------------------------------------------------
# SKILL DEPENDENCIES
# ---------------------------------------------------------

SKILL_DEPENDENCIES = {

    "Statistics": ["Python"],

    "Machine Learning": [
        "Python",
        "Statistics"
    ],

    "Deep Learning": [
        "Python",
        "Machine Learning"
    ],

    "NLP": [
        "Python",
        "Machine Learning"
    ],

    "Generative AI": [
        "Python",
        "Machine Learning",
        "Deep Learning"
    ],

    "LLM": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "NLP"
    ],

    "Git": ["Python"],

    "AWS": ["Python"],

    "Azure": ["Python"]
}


# ---------------------------------------------------------
# LEARNING PHASES
# ---------------------------------------------------------

SKILL_PHASES = {

    "Statistics": "Foundations",

    "Machine Learning": "Core Machine Learning",

    "Deep Learning": "Advanced Machine Learning",

    "NLP": "Advanced Machine Learning",

    "Generative AI": "Modern AI",

    "LLM": "Modern AI",

    "Git": "Production Skills",

    "AWS": "Production Skills",

    "Azure": "Production Skills"
}


PHASE_ORDER = {

    "Foundations": 1,

    "Core Machine Learning": 2,

    "Advanced Machine Learning": 3,

    "Modern AI": 4,

    "Production Skills": 5
}


# ---------------------------------------------------------
# LOAD SKILL GAP DATA
# ---------------------------------------------------------

def load_skill_gap_data():

    return pd.read_csv(INPUT_PATH)


# ---------------------------------------------------------
# BUILD CAREER ROADMAP
# ---------------------------------------------------------

def build_career_roadmap(
    user_skills,
    skill_gap_df=None
):

    if skill_gap_df is None:
        skill_gap_df = load_skill_gap_data()

    if skill_gap_df.empty:
        return pd.DataFrame()

    # -----------------------------------------------------
    # NORMALIZE USER SKILLS
    # -----------------------------------------------------

    user_skills_normalized = {
        skill.strip().lower()
        for skill in user_skills
    }

    # -----------------------------------------------------
    # DETERMINE MISSING SKILLS
    # -----------------------------------------------------

    skill_gap_df = skill_gap_df.copy()

    skill_gap_df["user_has_skill"] = (
        skill_gap_df["skill"]
        .str.lower()
        .isin(user_skills_normalized)
    )

    missing_skills = skill_gap_df[
        ~skill_gap_df["user_has_skill"]
    ].copy()

    if missing_skills.empty:
        return pd.DataFrame()

    # -----------------------------------------------------
    # SKILL INFORMATION LOOKUP
    # -----------------------------------------------------

    missing_skill_names = set(
        missing_skills["skill"]
    )

    skill_info = {}

    for _, row in missing_skills.iterrows():

        skill_info[row["skill"]] = {
            "priority": row["priority"],
            "percentage": row["percentage"]
        }

    # -----------------------------------------------------
    # DEPENDENCY HELPERS
    # -----------------------------------------------------

    def get_dependencies(skill):

        return SKILL_DEPENDENCIES.get(
            skill,
            []
        )

    # -----------------------------------------------------
    # DEPENDENCY-AWARE ORDERING
    # -----------------------------------------------------

    roadmap = []
    added = set()

    def add_skill(skill):

        if skill in added:
            return

        dependencies = get_dependencies(
            skill
        )

        # Add missing prerequisites first
        for dependency in dependencies:

            if (
                dependency in missing_skill_names
                and dependency not in added
            ):
                add_skill(dependency)

        if skill in missing_skill_names:
            roadmap.append(skill)

        added.add(skill)

    # -----------------------------------------------------
    # SORT BY LEARNING PHASE
    # -----------------------------------------------------

    skills_sorted = sorted(
        missing_skill_names,
        key=lambda skill: (
            PHASE_ORDER.get(
                SKILL_PHASES.get(
                    skill,
                    "Production Skills"
                ),
                99
            ),
            -skill_info[skill]["percentage"]
        )
    )

    for skill in skills_sorted:
        add_skill(skill)

    # -----------------------------------------------------
    # BUILD ROADMAP DATAFRAME
    # -----------------------------------------------------

    roadmap_rows = []

    for order, skill in enumerate(
        roadmap,
        start=1
    ):

        info = skill_info[skill]

        phase = SKILL_PHASES.get(
            skill,
            "Production Skills"
        )

        dependencies = get_dependencies(
            skill
        )

        roadmap_rows.append({

            "learning_order": order,

            "phase": phase,

            "skill": skill,

            "priority": info["priority"],

            "job_demand_percentage":
                info["percentage"],

            "dependencies":
                ", ".join(dependencies)
        })

    return pd.DataFrame(
        roadmap_rows
    )


# ---------------------------------------------------------
# SAVE ROADMAP
# ---------------------------------------------------------

def save_career_roadmap(roadmap_df):

    if roadmap_df is None or roadmap_df.empty:
        return

    os.makedirs(
        "data/processed/skill_analysis",
        exist_ok=True
    )

    roadmap_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


# ---------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CAREER INTELLIGENCE PLATFORM - CAREER ROADMAP V5")
    print("=" * 60)

    user_skills = [
        "Python",
        "SQL",
        "Pandas"
    ]

    target_role = "Data Scientist"

    print("\nTarget career:")
    print(target_role)

    print("\nUser skills:")

    for skill in user_skills:
        print(f"  ✓ {skill}")

    # -----------------------------------------------------
    # LOAD SKILL GAP
    # -----------------------------------------------------

    skill_gap = load_skill_gap_data()

    # -----------------------------------------------------
    # BUILD ROADMAP
    # -----------------------------------------------------

    roadmap_df = build_career_roadmap(
        user_skills=user_skills,
        skill_gap_df=skill_gap
    )

    if roadmap_df.empty:

        print(
            "\nExcellent! No missing skills found."
        )

    else:

        # -------------------------------------------------
        # DISPLAY ROADMAP
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("PERSONALIZED CAREER ROADMAP")
        print("=" * 60)

        current_phase = None

        for _, row in roadmap_df.iterrows():

            if row["phase"] != current_phase:

                current_phase = row["phase"]

                print(
                    f"\n--- {current_phase} ---"
                )

            print(
                f"\n{int(row['learning_order'])}. "
                f"{row['skill']}"
            )

            print(
                f"   Priority: "
                f"{row['priority']}"
            )

            print(
                f"   Job demand: "
                f"{row['job_demand_percentage']:.2f}%"
            )

            if row["dependencies"]:

                print(
                    f"   Prerequisites: "
                    f"{row['dependencies']}"
                )

            else:

                print(
                    "   Prerequisites: None"
                )

        # -------------------------------------------------
        # SAVE ROADMAP
        # -------------------------------------------------

        save_career_roadmap(
            roadmap_df
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("ROADMAP SUMMARY")
        print("=" * 60)

        print(
            f"Total skills to learn: "
            f"{len(roadmap_df)}"
        )

        print(
            f"Starting phase: "
            f"{roadmap_df.iloc[0]['phase']}"
        )

        print(
            f"First skill: "
            f"{roadmap_df.iloc[0]['skill']}"
        )

        print(
            f"\nRoadmap saved to:"
            f" {OUTPUT_PATH}"
        )

    print(
        "\nCareer Roadmap V5 completed successfully."
    )