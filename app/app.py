import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ml_job_recommender import recommend_jobs


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Career Intelligence Platform",
    page_icon="🚀",
    layout="wide"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🚀 Career Intelligence Platform")

st.write(
    "An ML-powered platform for personalized job "
    "recommendations and career intelligence."
)

st.divider()


# ---------------------------------------------------------
# CAREER PROFILE
# ---------------------------------------------------------

st.header("👤 Build Your Career Profile")

target_role = st.selectbox(
    "What role are you targeting?",
    [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "Data Engineer",
        "AI Engineer",
        "Business Analyst",
    ]
)


skills = st.multiselect(
    "Select your current skills",
    [
        "Python",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "Generative AI",
        "LLM",
        "Statistics",
        "Git",
        "AWS",
        "Azure",
        "GCP",
        "Power BI",
        "Tableau",
        "Excel",
        "Docker",
        "Apache Spark",
    ]
)


st.divider()


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button(
    "🔍 Analyze My Career",
    type="primary"
):

    if not skills:

        st.warning(
            "Please select at least one skill "
            "before analyzing your profile."
        )

    else:

        with st.spinner(
            "Running the ML recommendation engine..."
        ):

            recommendations = recommend_jobs(
                target_role=target_role,
                user_skills=skills,
                top_k=10,
                save_output=False
            )


        # -------------------------------------------------
        # PROFILE SUMMARY
        # -------------------------------------------------

        st.success(
            "Career analysis completed successfully!"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Target Role",
                target_role
            )

        with col2:
            st.metric(
                "Current Skills",
                len(skills)
            )

        with col3:
            st.metric(
                "Jobs Analyzed",
                len(recommendations)
            )


        st.divider()


        # -------------------------------------------------
        # JOB RECOMMENDATIONS
        # -------------------------------------------------

        st.header("🎯 Recommended Jobs")

        st.caption(
            "Ranked using your hybrid ML recommendation engine."
        )


        for _, job in recommendations.iterrows():

            score = job[
                "ml_recommendation_score"
            ]

            with st.container():

                st.subheader(
                    f"#{int(job['ml_recommendation_rank'])} "
                    f"{job['title']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(
                        f"🏢 **Company:** "
                        f"{job['company']}"
                    )

                with col2:
                    st.write(
                        f"📍 **Location:** "
                        f"{job['location']}"
                    )

                with col3:
                    st.write(
                        f"🎯 **Match:** "
                        f"{score:.2f}%"
                    )


                st.write(
                    f"**Matched skills:** "
                    f"{job['matched_skills'] or 'None detected'}"
                )


                st.write(
                    f"**Missing detected skills:** "
                    f"{job['missing_skills'] or 'None detected'}"
                )


                with st.expander(
                    "Why was this job recommended?"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**Content similarity:** "
                            f"{job['content_similarity']:.2f}%"
                        )

                        st.write(
                            f"**Role similarity:** "
                            f"{job['role_similarity']:.2f}%"
                        )

                    with col2:

                        st.write(
                            f"**Target-role skill match:** "
                            f"{job['skill_match']:.2f}%"
                        )

                        st.write(
                            f"**Job-specific skill fit:** "
                            f"{job['job_skill_fit']:.2f}%"
                        )


                    st.write(
                        "**Job description snippet:**"
                    )

                    st.write(
                        job["description"]
                    )


                if (
                    "redirect_url" in job
                    and pd.notna(job["redirect_url"])
                ):
                    st.link_button(
                        "View Job",
                        job["redirect_url"]
                    )


                st.divider()