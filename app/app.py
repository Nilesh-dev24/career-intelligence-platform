import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# IMPORT ML COMPONENTS
# ---------------------------------------------------------

from src.ml_job_recommender import recommend_jobs
from src.skill_gap_engine import (
    calculate_skill_gap,
    calculate_readiness,
    get_priority_summary
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Career Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🎯 Career Intelligence Platform")

st.write(
    "AI-powered career analysis using real job market data."
)


# ---------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------

st.header("👤 Your Career Profile")

target_role = st.selectbox(
    "Target Role",
    [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "Data Engineer",
        "AI Engineer",
        "Business Analyst"
    ]
)


skills = st.multiselect(
    "Your Skills",
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
        "Apache Spark"
    ]
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

analyze = st.button(
    "🚀 Analyze Career",
    type="primary"
)


if analyze:

    if not skills:

        st.warning(
            "Please select at least one skill before "
            "running the analysis."
        )

    else:

        # -------------------------------------------------
        # JOB RECOMMENDATIONS
        # -------------------------------------------------

        st.header("💼 Job Recommendations")

        with st.spinner("Analyzing job opportunities..."):

            recommendations = recommend_jobs(
                target_role=target_role,
                user_skills=skills,
                top_k=10,
                save_output=False
            )

        if recommendations.empty:

            st.error(
                "No job recommendations were found."
            )

        else:

            # -------------------------------------------------
            # PROFILE SUMMARY
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Target Role",
                    target_role
                )

            with col2:
                st.metric(
                    "Your Skills",
                    len(skills)
                )

            with col3:
                st.metric(
                    "Recommendations",
                    len(recommendations)
                )


            st.divider()


            # -------------------------------------------------
            # DISPLAY RECOMMENDATIONS
            # -------------------------------------------------

            for index, (_, job) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                score = job["ml_recommendation_score"]

                st.subheader(
                    f"{index}. {job['title']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"**Company:** "
                        f"{job['company']}"
                    )

                with col2:

                    st.write(
                        f"**Location:** "
                        f"{job['location']}"
                    )

                with col3:

                    st.write(
                        f"**Match Score:** "
                        f"{score:.2f}%"
                    )


                matched_skills = job.get(
                    "matched_skills",
                    ""
                )

                missing_skills = job.get(
                    "missing_skills",
                    ""
                )


                if pd.notna(matched_skills) and matched_skills:

                    st.write(
                        f"✅ **Matched skills:** "
                        f"{matched_skills}"
                    )

                else:

                    st.write(
                        "✅ **Matched skills:** None detected"
                    )


                if pd.notna(missing_skills) and missing_skills:

                    st.write(
                        f"⚠️ **Skills not detected in job "
                        f"data:** {missing_skills}"
                    )


                with st.expander(
                    "Why was this job recommended?"
                ):

                    st.write(
                        f"**Content similarity:** "
                        f"{job['content_similarity']:.2f}"
                    )

                    st.write(
                        f"**Role similarity:** "
                        f"{job['role_similarity']:.2f}"
                    )

                    st.write(
                        f"**Target-role skill match:** "
                        f"{job['skill_match']:.2f}"
                    )

                    st.write(
                        f"**Job-specific skill fit:** "
                        f"{job['job_skill_fit']:.2f}"
                    )


                    description = job.get(
                        "description",
                        ""
                    )

                    if pd.notna(description) and description:

                        st.write("**Job Description**")

                        st.write(description)


                job_url = job.get(
                    "redirect_url",
                    ""
                )

                if pd.notna(job_url) and job_url:

                    st.link_button(
                        "View Job",
                        job_url
                    )


                st.divider()


        # -------------------------------------------------
        # SKILL GAP ANALYSIS
        # -------------------------------------------------

        st.header("📊 Skill Gap Analysis")

        with st.spinner("Analyzing your skill gaps..."):

            skill_gap = calculate_skill_gap(
                role=target_role,
                user_skills=skills,
                minimum_percentage=5
            )


        if skill_gap is None or skill_gap.empty:

            st.warning(
                f"No skill-demand data was found for "
                f"{target_role}."
            )

        else:

            # -------------------------------------------------
            # CAREER READINESS
            # -------------------------------------------------

            readiness = calculate_readiness(
                skill_gap
            )

            priority_summary = get_priority_summary(
                skill_gap
            )


            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Career Readiness",
                    f"{readiness['readiness_percentage']:.1f}%"
                )

            with col2:

                st.metric(
                    "Skills You Have",
                    readiness["matched_skills"]
                )

            with col3:

                st.metric(
                    "Skills Analyzed",
                    readiness["total_skills"]
                )

            with col4:

                st.metric(
                    "High Priority Gaps",
                    priority_summary["high_priority"]
                )


            st.divider()


            # -------------------------------------------------
            # SKILL GAP TABLE
            # -------------------------------------------------

            display_columns = [
                "skill",
                "job_count",
                "percentage",
                "status",
                "priority"
            ]

            display_data = skill_gap[
                display_columns
            ].copy()

            display_data = display_data.rename(
                columns={
                    "skill": "Skill",
                    "job_count": "Jobs",
                    "percentage": "Demand %",
                    "status": "Status",
                    "priority": "Priority"
                }
            )


            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


            # -------------------------------------------------
            # PRIORITIZED GAPS
            # -------------------------------------------------

            missing = skill_gap[
                skill_gap["status"] == "Missing"
            ].copy()


            st.subheader(
                "🎯 Prioritized Skill Gaps"
            )


            if missing.empty:

                st.success(
                    "Excellent! No major skill gaps detected."
                )

            else:

                for index, (_, row) in enumerate(
                    missing.iterrows(),
                    start=1
                ):

                    st.write(
                        f"**{index}. {row['skill']}** — "
                        f"{row['priority']} priority "
                        f"({row['percentage']:.2f}% "
                        f"of jobs)"
                    )


st.caption(
    "Career Intelligence Platform • "
    "Powered by job market data and machine learning"
)