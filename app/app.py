# =========================================================
# CAREER INTELLIGENCE PLATFORM
# Main Streamlit Application
# =========================================================

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORT ML COMPONENTS
# =========================================================

from src.ml_job_recommender import recommend_jobs

from src.skill_gap_engine import (
    calculate_skill_gap,
    calculate_readiness,
    get_priority_summary,
)

from src.career_roadmap import build_career_roadmap


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Career Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎯 Career Intelligence")

    st.caption(
        "AI-powered career analysis"
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "💼 Job Recommendations",
            "📊 Skill Gap",
            "🗺️ Career Roadmap",
            "ℹ️ About",
        ],
    )

    st.divider()

    st.caption(
        "Career Intelligence Platform"
    )

    st.caption(
        "Powered by Machine Learning"
    )


# =========================================================
# MAIN HEADER
# =========================================================

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
        The **Career Intelligence Platform** is a machine-learning
        powered career analysis application designed to help users
        understand their career readiness and discover relevant
        opportunities.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **💼 Job Recommendations**

            Recommends relevant jobs using a hybrid ML
            recommendation system based on job content,
            target-role similarity, and skill matching.

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

        • Skill matching

        • Hybrid recommendation scoring

        • Explainable job recommendations

        • Data-driven skill gap analysis

        • Dependency-aware career roadmap
        """
    )

    st.divider()

    st.info(
        "Job market data is sourced from the Adzuna API."
    )

    st.stop()


# =========================================================
# PROFILE INPUT
# =========================================================

elif page in [
    "🏠 Dashboard",
    "💼 Job Recommendations",
    "📊 Skill Gap",
    "🗺️ Career Roadmap",
]:

    st.header("👤 Your Career Profile")

    col1, col2 = st.columns([1, 2])

    with col1:

        target_role = st.selectbox(
            "Target Role",
            [
                "Data Scientist",
                "Machine Learning Engineer",
                "Data Analyst",
                "Data Engineer",
                "AI Engineer",
                "Business Analyst",
            ],
            index=0,
        )

    with col2:

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
                "Docker",
                "Power BI",
                "Tableau",
                "Excel",
                "Data Visualization",
            ],
        )

    st.divider()

    analyze = st.button(
        "🚀 Analyze My Career",
        type="primary",
        width="stretch",
    )

    # =====================================================
    # RUN ANALYSIS
    # =====================================================

    if analyze:

        if not skills:

            st.warning(
                "Please select at least one skill before analyzing."
            )

            st.stop()

        with st.spinner(
            "Analyzing your career profile..."
        ):

            # -------------------------------------------------
            # JOB RECOMMENDATIONS
            # -------------------------------------------------

            recommendations = recommend_jobs(
                target_role=target_role,
                user_skills=skills,
                top_k=10,
                save_output=False,
            )

            # -------------------------------------------------
            # SKILL GAP
            # -------------------------------------------------

            skill_gap = calculate_skill_gap(
                role=target_role,
                user_skills=skills,
                minimum_percentage=5,
            )

            # -------------------------------------------------
            # CAREER ROADMAP
            # -------------------------------------------------

            if (
                skill_gap is not None
                and not skill_gap.empty
            ):

                roadmap = build_career_roadmap(
                    user_skills=skills,
                    skill_gap_df=skill_gap,
                )

            else:

                roadmap = pd.DataFrame()

        # -----------------------------------------------------
        # SAVE RESULTS IN SESSION
        # -----------------------------------------------------

        st.session_state.analysis_complete = True

        st.session_state.recommendations = (
            recommendations
        )

        st.session_state.skill_gap = skill_gap

        st.session_state.roadmap = roadmap

        st.session_state.target_role = target_role

        st.session_state.skills = skills

        st.success(
            "Career analysis completed successfully."
        )


# =========================================================
# LOAD SESSION RESULTS
# =========================================================

recommendations = st.session_state.recommendations

skill_gap = st.session_state.skill_gap

roadmap = st.session_state.roadmap

analysis_complete = st.session_state.analysis_complete


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header("📈 Career Dashboard")

    if not analysis_complete:

        st.info(
            "Select your target role and skills above, "
            "then click **Analyze My Career** to generate "
            "your personalized dashboard."
        )

        st.stop()

    # -----------------------------------------------------
    # READINESS
    # -----------------------------------------------------

    readiness = calculate_readiness(
        skill_gap
    )

    priority_summary = get_priority_summary(
        skill_gap
    )

    total_skills = readiness.get(
        "total_skills",
        0,
    )

    matched_skills = readiness.get(
        "matched_skills",
        0,
    )

    readiness_percentage = readiness.get(
        "readiness_percentage",
        0,
    )

    high_priority = priority_summary.get(
        "high",
        0,
    )

    missing_count = (
        total_skills - matched_skills
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Career Readiness",
            f"{readiness_percentage:.1f}%",
        )

    with col2:

        st.metric(
            "Your Skills",
            len(st.session_state.skills),
        )

    with col3:

        st.metric(
            "Recommended Jobs",
            len(recommendations),
        )

    with col4:

        st.metric(
            "High-Priority Gaps",
            high_priority,
        )

    with col5:

        st.metric(
            "Skills to Learn",
            missing_count,
        )

    st.divider()

    # -----------------------------------------------------
    # PROFILE OVERVIEW
    # -----------------------------------------------------

    st.subheader("👤 Profile Overview")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:

        st.markdown("**Target Role**")

        st.info(
            st.session_state.target_role
        )

    with profile_col2:

        st.markdown("**Your Skills**")

        st.info(
            ", ".join(
                st.session_state.skills
            )
        )

    st.divider()

    # -----------------------------------------------------
    # TOP SKILL GAPS
    # -----------------------------------------------------

    st.subheader("⚠️ Top Skill Gaps")

    if skill_gap.empty:

        st.success(
            "No significant skill gaps were detected."
        )

    else:

        missing = skill_gap[
            skill_gap["status"] == "Missing"
        ].copy()

        if missing.empty:

            st.success(
                "Excellent! No missing skills were detected "
                "for the selected threshold."
            )

        else:

            missing = missing.sort_values(
                "percentage",
                ascending=False,
            )

            display_columns = [
                "skill",
                "percentage",
                "priority",
            ]

            st.dataframe(
                missing[
                    display_columns
                ].head(5),
                width="stretch",
                hide_index=True,
            )

    st.divider()

    # -----------------------------------------------------
    # TOP JOB MATCHES
    # -----------------------------------------------------

    st.subheader("💼 Top Job Matches")

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
            start=1,
        ):

            score = job[
                "ml_recommendation_score"
            ]

            st.write(
                f"**{index}. {job['title']}** — "
                f"{job['company']} — "
                f"Match: {score:.2f}%"
            )


# =========================================================
# JOB RECOMMENDATIONS
# =========================================================

elif page == "💼 Job Recommendations":

    st.header("💼 Job Recommendations")

    if not analysis_complete:

        st.info(
            "Run your career analysis first to generate "
            "personalized job recommendations."
        )

        st.stop()

    if recommendations.empty:

        st.error(
            "No job recommendations were found. "
            "Try changing your target role or skills."
        )

    else:

        st.write(
            f"Showing the top **{len(recommendations)}** "
            f"ML-ranked job recommendations."
        )

        st.caption(
            "Recommendations are ranked using job-content "
            "similarity, target-role similarity, and skill matching."
        )

        st.divider()

        # -------------------------------------------------
        # JOB CARDS
        # -------------------------------------------------

        for index, (_, job) in enumerate(
            recommendations.iterrows(),
            start=1,
        ):

            score = job[
                "ml_recommendation_score"
            ]

            # -------------------------------------------------
            # JOB CARD
            # -------------------------------------------------

            with st.container(border=True):

                # ---------------------------------------------
                # JOB HEADER
                # ---------------------------------------------

                header_col1, header_col2 = st.columns(
                    [4, 1]
                )

                with header_col1:

                    st.subheader(
                        f"{index}. {job['title']}"
                    )

                    st.caption(
                        f"{job['company']} • "
                        f"{job['location']}"
                    )

                with header_col2:

                    st.metric(
                        "Match",
                        f"{score:.1f}%",
                    )

                st.divider()

                # ---------------------------------------------
                # BASIC INFORMATION
                # ---------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.markdown(
                        "**🏢 Company**"
                    )

                    st.write(
                        job["company"]
                    )

                with col2:

                    st.markdown(
                        "**📍 Location**"
                    )

                    st.write(
                        job["location"]
                    )

                with col3:

                    st.markdown(
                        "**🎯 ML Match Score**"
                    )

                    st.write(
                        f"{score:.2f}%"
                    )

                # ---------------------------------------------
                # SKILL FIT
                # ---------------------------------------------

                st.markdown(
                    "### 🧠 Skill Fit"
                )

                matched_skills = job.get(
                    "matched_skills",
                    "",
                )

                missing_skills = job.get(
                    "missing_skills",
                    "",
                )

                if (
                    pd.notna(matched_skills)
                    and str(matched_skills).strip()
                ):

                    st.success(
                        f"Matched skills: "
                        f"{matched_skills}"
                    )

                else:

                    st.info(
                        "No matching skills were detected "
                        "in the available job data."
                    )

                if (
                    pd.notna(missing_skills)
                    and str(missing_skills).strip()
                ):

                    st.warning(
                        f"Skills not detected in available "
                        f"job data: {missing_skills}"
                    )

                # ---------------------------------------------
                # EXPLAINABILITY
                # ---------------------------------------------

                with st.expander(
                    "🔍 Why was this job recommended?"
                ):

                    metric_col1, metric_col2 = (
                        st.columns(2)
                    )

                    with metric_col1:

                        st.write(
                            "**Content similarity**"
                        )

                        content_similarity = float(
                            job[
                                "content_similarity"
                            ]
                        )

                        st.progress(
                            min(
                                max(
                                    content_similarity,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                        st.caption(
                            f"Score: "
                            f"{content_similarity:.2f}"
                        )

                        st.write(
                            "**Role similarity**"
                        )

                        role_similarity = float(
                            job[
                                "role_similarity"
                            ]
                        )

                        st.progress(
                            min(
                                max(
                                    role_similarity,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                        st.caption(
                            f"Score: "
                            f"{role_similarity:.2f}"
                        )

                    with metric_col2:

                        st.write(
                            "**Target-role skill match**"
                        )

                        skill_match = float(
                            job[
                                "skill_match"
                            ]
                        )

                        st.progress(
                            min(
                                max(
                                    skill_match,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                        st.caption(
                            f"Score: "
                            f"{skill_match:.2f}"
                        )

                        st.write(
                            "**Job-specific skill fit**"
                        )

                        job_skill_fit = float(
                            job[
                                "job_skill_fit"
                            ]
                        )

                        st.progress(
                            min(
                                max(
                                    job_skill_fit,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                        st.caption(
                            f"Score: "
                            f"{job_skill_fit:.2f}"
                        )

                    st.divider()

                    st.caption(
                        "The recommendation score combines "
                        "job-content similarity, target-role "
                        "similarity, and skill matching."
                    )

                # ---------------------------------------------
                # JOB DESCRIPTION
                # ---------------------------------------------

                description = job.get(
                    "description",
                    "",
                )

                if (
                    pd.notna(description)
                    and str(description).strip()
                ):

                    with st.expander(
                        "📄 Job Description"
                    ):

                        st.write(
                            description
                        )

                # ---------------------------------------------
                # JOB LINK
                # ---------------------------------------------

                redirect_url = job.get(
                    "redirect_url",
                    "",
                )

                if (
                    pd.notna(redirect_url)
                    and str(redirect_url).strip()
                ):

                    st.link_button(
                        "🔗 View Job",
                        str(redirect_url),
                        width="stretch",
                    )

            st.write("")


# =========================================================
# SKILL GAP
# =========================================================

elif page == "📊 Skill Gap":

    st.header("📊 Skill Gap Analysis")

    if not analysis_complete:

        st.info(
            "Run your career analysis first to generate "
            "your personalized skill gap."
        )

        st.stop()

    if skill_gap.empty:

        st.warning(
            "No skill gap data is available."
        )

    else:

        readiness = calculate_readiness(
            skill_gap
        )

        priority_summary = get_priority_summary(
            skill_gap
        )

        # -------------------------------------------------
        # READINESS METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Career Readiness",
                f"{readiness['readiness_percentage']:.1f}%",
            )

        with col2:

            st.metric(
                "Skills Analyzed",
                readiness["total_skills"],
            )

        with col3:

            st.metric(
                "Skills Matched",
                readiness["matched_skills"],
            )

        with col4:

            st.metric(
                "Missing Skills",
                (
                    readiness["total_skills"]
                    - readiness["matched_skills"]
                ),
            )

        st.divider()

        # -------------------------------------------------
        # PRIORITY SUMMARY
        # -------------------------------------------------

        st.subheader(
            "🎯 Priority Summary"
        )

        priority_col1, priority_col2, priority_col3 = (
            st.columns(3)
        )

        with priority_col1:

            st.metric(
                "🔴 High Priority",
                priority_summary.get("high", priority_summary.get("High", 0)),
            )

        with priority_col2:

            st.metric(
                "🟠 Medium Priority",
                priority_summary.get("medium", priority_summary.get("Medium", 0)),
            )

        with priority_col3:

            st.metric(
                "🟢 Low Priority",
                priority_summary.get("low", priority_summary.get("Low", 0)),
            )

        st.divider()

        # -------------------------------------------------
        # COMPLETE SKILL GAP TABLE
        # -------------------------------------------------

        st.subheader(
            "📋 Detailed Skill Gap"
        )

        display_columns = [
            "skill",
            "percentage",
            "status",
            "priority",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in skill_gap.columns
        ]

        st.dataframe(
            skill_gap[
                available_columns
            ],
            width="stretch",
            hide_index=True,
        )

        st.divider()

        # -------------------------------------------------
        # PRIORITIZED MISSING SKILLS
        # -------------------------------------------------

        st.subheader(
            "🚀 Skills You Should Learn"
        )

        missing = skill_gap[
            skill_gap["status"] == "Missing"
        ].copy()

        if missing.empty:

            st.success(
                "You currently have all skills "
                "identified as important for this role."
            )

        else:

            missing = missing.sort_values(
                "percentage",
                ascending=False,
            )

            for index, (_, row) in enumerate(
                missing.iterrows(),
                start=1,
            ):

                skill = row["skill"]

                demand = row[
                    "percentage"
                ]

                priority = row[
                    "priority"
                ]

                with st.container(border=True):

                    col1, col2, col3 = st.columns(
                        [3, 2, 1]
                    )

                    with col1:

                        st.markdown(
                            f"**{index}. {skill}**"
                        )

                    with col2:

                        st.write(
                            f"Job demand: "
                            f"{demand:.2f}%"
                        )

                    with col3:

                        st.write(
                            f"**{priority}**"
                        )


# =========================================================
# CAREER ROADMAP
# =========================================================

elif page == "🗺️ Career Roadmap":

    st.header("🗺️ Personalized Career Roadmap")

    if not analysis_complete:

        st.info(
            "Run your career analysis first to generate "
            "your personalized roadmap."
        )

        st.stop()

    if roadmap.empty:

        st.warning(
            "No roadmap could be generated."
        )

    else:

        st.write(
            "A dependency-aware learning sequence "
            "based on your current skills and target role."
        )

        st.divider()

        # -------------------------------------------------
        # ROADMAP OVERVIEW
        # -------------------------------------------------

        st.subheader(
            "📚 Learning Plan"
        )

        st.write(
            f"**Target Role:** "
            f"{st.session_state.target_role}"
        )

        st.write(
            f"**Skills You Currently Have:** "
            f"{', '.join(st.session_state.skills)}"
        )

        st.divider()

        # -------------------------------------------------
        # PHASES
        # -------------------------------------------------

        phases = roadmap[
            "phase"
        ].dropna().unique()

        for phase in phases:

            phase_data = roadmap[
                roadmap["phase"] == phase
            ].copy()

            if phase_data.empty:
                continue

            phase_order = phase_data[
                "learning_order"
            ].min()

            st.subheader(
                f"Phase {int(phase_order)} — {phase}"
            )

            for _, row in phase_data.iterrows():

                skill = row["skill"]

                priority = row[
                    "priority"
                ]

                demand = row[
                    "job_demand_percentage"
                ]

                dependencies = row.get(
                    "dependencies",
                    "",
                )

                with st.container(border=True):

                    col1, col2, col3 = st.columns(
                        [3, 1, 2]
                    )

                    with col1:

                        st.markdown(
                            f"**{skill}**"
                        )

                    with col2:

                        st.write(
                            f"{priority}"
                        )

                    with col3:

                        st.write(
                            f"Demand: {demand:.2f}%"
                        )

                    if (
                        pd.notna(dependencies)
                        and str(dependencies).strip()
                    ):

                        st.caption(
                            f"Prerequisites: "
                            f"{dependencies}"
                        )

            st.write("")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎯 Career Intelligence Platform • "
    "Machine Learning powered career analysis"
)