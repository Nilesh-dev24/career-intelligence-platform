import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Career Intelligence Platform",
    page_icon="🎯",
    layout="wide"
)


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

from src.career_roadmap import build_career_roadmap


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

if "skill_gap" not in st.session_state:
    st.session_state.skill_gap = pd.DataFrame()

if "roadmap" not in st.session_state:
    st.session_state.roadmap = pd.DataFrame()

if "target_role" not in st.session_state:
    st.session_state.target_role = ""

if "skills" not in st.session_state:
    st.session_state.skills = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🎯 Career Intelligence")

    st.write(
        "AI-powered career analysis "
        "using real job market data."
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "💼 Job Recommendations",
            "📊 Skill Gap",
            "🗺️ Career Roadmap",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.caption(
        "Career Intelligence Platform"
    )

    st.caption(
        "Powered by Machine Learning"
    )


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.title("🎯 Career Intelligence Platform")

st.write(
    "AI-powered career analysis using real job market data."
)


# =========================================================
# ABOUT PAGE
# =========================================================

if page == "ℹ️ About":

    st.header("ℹ️ About the Platform")

    st.write(
        """
        The Career Intelligence Platform is a machine-learning
        powered career analysis application designed to help
        users understand their career fit and identify the
        skills they should learn next.
        """
    )

    st.divider()

    st.subheader("🚀 What the platform does")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **💼 Job Recommendations**

            Recommends relevant jobs using a hybrid ML
            recommendation system based on job content,
            target role similarity, and skill matching.

            **📊 Skill Gap Analysis**

            Compares your selected skills with skills
            frequently detected in jobs for your target role.
            """
        )

    with col2:

        st.markdown(
            """
            **🗺️ Career Roadmap**

            Creates a dependency-aware learning sequence
            for missing skills.

            **📈 Career Dashboard**

            Provides an overview of career readiness,
            recommended jobs, priority gaps, and learning needs.
            """
        )

    st.divider()

    st.subheader("🧠 Machine Learning Components")

    st.write(
        """
        • TF-IDF based job representation

        • Cosine similarity

        • Target-role similarity

        • Skill-demand analysis

        • Personalized skill-gap detection

        • Dependency-aware career roadmap
        """
    )

    st.divider()

    st.subheader("📊 Data Source")

    st.write(
        """
        Job-market information is collected from the
        Adzuna Jobs API and processed through the platform's
        data pipeline.
        """
    )

    st.info(
        "The platform's skill detection is based on the "
        "job-description data available to the system."
    )


# =========================================================
# PROFILE PAGE / INPUT
# =========================================================

elif page in [
    "🏠 Dashboard",
    "💼 Job Recommendations",
    "📊 Skill Gap",
    "🗺️ Career Roadmap"
]:

    # -----------------------------------------------------
    # USER PROFILE
    # -----------------------------------------------------

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
        ],
        index=0
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


    analyze = st.button(
        "🚀 Analyze Career",
        type="primary"
    )


    # -----------------------------------------------------
    # RUN ANALYSIS
    # -----------------------------------------------------

    if analyze:

        if not skills:

            st.warning(
                "Please select at least one skill before "
                "running the analysis."
            )

            st.stop()


        with st.spinner(
            "Analyzing your career profile..."
        ):

            # ---------------------------------------------
            # JOB RECOMMENDATIONS
            # ---------------------------------------------

            recommendations = recommend_jobs(
                target_role=target_role,
                user_skills=skills,
                top_k=10,
                save_output=False
            )


            # ---------------------------------------------
            # SKILL GAP
            # ---------------------------------------------

            skill_gap = calculate_skill_gap(
                role=target_role,
                user_skills=skills,
                minimum_percentage=5
            )


            # ---------------------------------------------
            # ROADMAP
            # ---------------------------------------------

            if (
                skill_gap is not None
                and not skill_gap.empty
            ):

                roadmap = build_career_roadmap(
                    user_skills=skills,
                    skill_gap_df=skill_gap
                )

            else:

                roadmap = pd.DataFrame()


        # ---------------------------------------------
        # SAVE RESULTS IN SESSION
        # ---------------------------------------------

        st.session_state.analysis_complete = True

        st.session_state.recommendations = (
            recommendations
        )

        if skill_gap is not None:

            st.session_state.skill_gap = skill_gap

        else:

            st.session_state.skill_gap = pd.DataFrame()


        st.session_state.roadmap = roadmap

        st.session_state.target_role = target_role

        st.session_state.skills = skills


        st.success(
            "Career analysis completed successfully!"
        )


    # -----------------------------------------------------
    # LOAD STORED RESULTS
    # -----------------------------------------------------

    if st.session_state.analysis_complete:

        recommendations = (
            st.session_state.recommendations
        )

        skill_gap = (
            st.session_state.skill_gap
        )

        roadmap = (
            st.session_state.roadmap
        )

        target_role = (
            st.session_state.target_role
        )

        skills = (
            st.session_state.skills
        )


        # =================================================
        # DASHBOARD
        # =================================================

        if page == "🏠 Dashboard":

            st.header("📊 Career Dashboard")

            # ---------------------------------------------
            # READINESS
            # ---------------------------------------------

            if (
                skill_gap is not None
                and not skill_gap.empty
            ):

                readiness = calculate_readiness(
                    skill_gap
                )

                priority_summary = (
                    get_priority_summary(
                        skill_gap
                    )
                )

            else:

                readiness = {
                    "total_skills": 0,
                    "matched_skills": 0,
                    "readiness_percentage": 0.0
                }

                priority_summary = {
                    "high_priority": 0,
                    "medium_priority": 0,
                    "low_priority": 0
                }


            total_jobs = len(
                recommendations
            )

            total_to_learn = len(
                roadmap
            )

            high_priority_gaps = (
                priority_summary["high_priority"]
            )


            # ---------------------------------------------
            # DASHBOARD METRICS
            # ---------------------------------------------

            col1, col2, col3, col4, col5 = (
                st.columns(5)
            )


            with col1:

                st.metric(
                    "Career Readiness",
                    f"{readiness['readiness_percentage']:.1f}%"
                )


            with col2:

                st.metric(
                    "Your Skills",
                    len(skills)
                )


            with col3:

                st.metric(
                    "Recommended Jobs",
                    total_jobs
                )


            with col4:

                st.metric(
                    "High-Priority Gaps",
                    high_priority_gaps
                )


            with col5:

                st.metric(
                    "Skills to Learn",
                    total_to_learn
                )


            st.divider()


            # ---------------------------------------------
            # PROFILE OVERVIEW
            # ---------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "🎯 Career Profile"
                )

                st.write(
                    f"**Target Role:** "
                    f"{target_role}"
                )

                st.write(
                    f"**Skills Selected:** "
                    f"{len(skills)}"
                )

                st.write(
                    "**Your Skills:** "
                    + ", ".join(skills)
                )


            with col2:

                st.subheader(
                    "🚀 Recommended Next Step"
                )

                if (
                    roadmap is not None
                    and not roadmap.empty
                ):

                    first_skill = (
                        roadmap.iloc[0]["skill"]
                    )

                    first_phase = (
                        roadmap.iloc[0]["phase"]
                    )

                    st.write(
                        f"**Start with:** "
                        f"{first_skill}"
                    )

                    st.write(
                        f"**Phase:** "
                        f"{first_phase}"
                    )

                    st.write(
                        f"**Skills remaining:** "
                        f"{len(roadmap)}"
                    )

                else:

                    st.success(
                        "🎉 No additional skills "
                        "were identified."
                    )


            st.divider()


            # ---------------------------------------------
            # TOP SKILL GAPS
            # ---------------------------------------------

            st.subheader(
                "🎯 Top Skill Gaps"
            )


            if (
                skill_gap is not None
                and not skill_gap.empty
            ):

                missing = skill_gap[
                    skill_gap["status"] == "Missing"
                ].copy()


                if missing.empty:

                    st.success(
                        "No major skill gaps detected."
                    )

                else:

                    top_missing = missing.head(5)


                    for index, (_, row) in enumerate(
                        top_missing.iterrows(),
                        start=1
                    ):

                        st.write(
                            f"**{index}. "
                            f"{row['skill']}** — "
                            f"{row['priority']} priority — "
                            f"{row['percentage']:.2f}% "
                            f"job demand"
                        )


            st.divider()


            # ---------------------------------------------
            # TOP JOBS
            # ---------------------------------------------

            st.subheader(
                "💼 Top Job Matches"
            )


            if recommendations.empty:

                st.warning(
                    "No job recommendations found."
                )

            else:

                dashboard_jobs = (
                    recommendations.head(5)
                )


                for index, (_, job) in enumerate(
                    dashboard_jobs.iterrows(),
                    start=1
                ):

                    score = (
                        job[
                            "ml_recommendation_score"
                        ]
                    )

                    st.write(
                        f"**{index}. "
                        f"{job['title']}** — "
                        f"{job['company']} — "
                        f"Match: {score:.2f}%"
                    )


        # =================================================
        # JOB RECOMMENDATIONS
        # =================================================

        elif page == "💼 Job Recommendations":

            st.header(
                "💼 Job Recommendations"
            )


            if recommendations.empty:

                st.error(
                    "No job recommendations were found."
                )

            else:

                st.write(
                    f"Showing the top "
                    f"{len(recommendations)} "
                    f"ML-ranked job recommendations."
                )


                st.divider()


                for index, (_, job) in enumerate(
                    recommendations.iterrows(),
                    start=1
                ):

                    score = (
                        job[
                            "ml_recommendation_score"
                        ]
                    )


                    st.subheader(
                        f"{index}. {job['title']}"
                    )


                    col1, col2, col3 = (
                        st.columns(3)
                    )


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


                    if (
                        pd.notna(matched_skills)
                        and matched_skills
                    ):

                        st.write(
                            f"✅ **Matched skills:** "
                            f"{matched_skills}"
                        )

                    else:

                        st.write(
                            "✅ **Matched skills:** "
                            "None detected"
                        )


                    if (
                        pd.notna(missing_skills)
                        and missing_skills
                    ):

                        st.write(
                            f"⚠️ **Skills not detected "
                            f"in job data:** "
                            f"{missing_skills}"
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


                        if (
                            pd.notna(description)
                            and description
                        ):

                            st.write(
                                "**Job Description**"
                            )

                            st.write(
                                description
                            )


                    job_url = job.get(
                        "redirect_url",
                        ""
                    )


                    if (
                        pd.notna(job_url)
                        and job_url
                    ):

                        st.link_button(
                            "View Job",
                            job_url
                        )


                    st.divider()


        # =================================================
        # SKILL GAP
        # =================================================

        elif page == "📊 Skill Gap":

            st.header(
                "📊 Skill Gap Analysis"
            )


            if (
                skill_gap is None
                or skill_gap.empty
            ):

                st.warning(
                    f"No skill-demand data was found "
                    f"for {target_role}."
                )

            else:

                readiness = calculate_readiness(
                    skill_gap
                )

                priority_summary = (
                    get_priority_summary(
                        skill_gap
                    )
                )


                # -----------------------------------------
                # READINESS METRICS
                # -----------------------------------------

                col1, col2, col3, col4 = (
                    st.columns(4)
                )


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
                        priority_summary[
                            "high_priority"
                        ]
                    )


                st.divider()


                # -----------------------------------------
                # SKILL GAP TABLE
                # -----------------------------------------

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


                display_data = (
                    display_data.rename(
                        columns={
                            "skill": "Skill",
                            "job_count": "Jobs",
                            "percentage": "Demand %",
                            "status": "Status",
                            "priority": "Priority"
                        }
                    )
                )


                st.dataframe(
                    display_data,
                    width="stretch",
                    hide_index=True
                )


                st.divider()


                # -----------------------------------------
                # MISSING SKILLS
                # -----------------------------------------

                st.subheader(
                    "🎯 Prioritized Skill Gaps"
                )


                missing = skill_gap[
                    skill_gap["status"] == "Missing"
                ].copy()


                if missing.empty:

                    st.success(
                        "Excellent! No major skill gaps "
                        "detected."
                    )

                else:

                    for index, (_, row) in enumerate(
                        missing.iterrows(),
                        start=1
                    ):

                        st.write(
                            f"**{index}. "
                            f"{row['skill']}** — "
                            f"{row['priority']} priority "
                            f"({row['percentage']:.2f}% "
                            f"of jobs)"
                        )


        # =================================================
        # CAREER ROADMAP
        # =================================================

        elif page == "🗺️ Career Roadmap":

            st.header(
                "🗺️ Personalized Career Roadmap"
            )


            if roadmap.empty:

                st.success(
                    "🎉 You already have all the skills "
                    "identified for this career path!"
                )

            else:

                # -----------------------------------------
                # ROADMAP SUMMARY
                # -----------------------------------------

                total_to_learn = len(
                    roadmap
                )


                high_priority = len(
                    roadmap[
                        roadmap["priority"] == "High"
                    ]
                )


                first_phase = (
                    roadmap.iloc[0]["phase"]
                )


                first_skill = (
                    roadmap.iloc[0]["skill"]
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )


                with col1:

                    st.metric(
                        "Skills to Learn",
                        total_to_learn
                    )


                with col2:

                    st.metric(
                        "High Priority",
                        high_priority
                    )


                with col3:

                    st.metric(
                        "Starting Phase",
                        first_phase
                    )


                with col4:

                    st.metric(
                        "Start With",
                        first_skill
                    )


                st.divider()


                # -----------------------------------------
                # ROADMAP PHASES
                # -----------------------------------------

                phases = (
                    roadmap["phase"]
                    .drop_duplicates()
                    .tolist()
                )


                for phase in phases:

                    st.subheader(
                        f"📍 {phase}"
                    )


                    phase_data = roadmap[
                        roadmap["phase"] == phase
                    ]


                    for _, row in (
                        phase_data.iterrows()
                    ):

                        order = (
                            row["learning_order"]
                        )

                        skill = row["skill"]

                        priority = row["priority"]

                        demand = (
                            row[
                                "job_demand_percentage"
                            ]
                        )

                        dependencies = (
                            row["dependencies"]
                        )


                        with st.container(
                            border=True
                        ):

                            col1, col2 = (
                                st.columns([4, 1])
                            )


                            with col1:

                                st.write(
                                    f"### "
                                    f"{int(order)}. "
                                    f"{skill}"
                                )


                            with col2:

                                st.write(
                                    f"**{priority}**"
                                )


                            st.write(
                                f"📈 **Job demand:** "
                                f"{demand:.2f}%"
                            )


                            if dependencies:

                                st.write(
                                    f"🔗 **Prerequisites:** "
                                    f"{dependencies}"
                                )

                            else:

                                st.write(
                                    "🔗 **Prerequisites:** "
                                    "None"
                                )


                    st.write("")


    else:

        # -----------------------------------------------------
        # NO ANALYSIS YET
        # -----------------------------------------------------

        st.info(
            "👆 Select your target role and skills, "
            "then click **Analyze Career** to generate "
            "your personalized career intelligence."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Career Intelligence Platform • "
    "Powered by job market data and machine learning"
)