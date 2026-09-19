from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
REPO_BASE = "https://github.com/Abdelrahim-maker/microbiome-multiomics-workflows/blob/main"
PAGES_BASE = "https://abdelrahim-maker.github.io/microbiome-multiomics-workflows"
CV_URL = "https://raw.githubusercontent.com/Abdelrahim-maker/microbiome-multiomics-workflows/main/docs/assets/Afaf_Abdelrahim_CV_2026.pdf"


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, sep="\t")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_models = load_table("adjusted_feature_models.tsv")
    community_tests = load_table("adjusted_community_tests.tsv")
    concordance = load_table("direct_inferred_gene_concordance.tsv")
    return feature_models, community_tests, concordance


def style_page() -> None:
    st.set_page_config(
        page_title="Microbiome Multi-Omics Explorer",
        page_icon="🧪",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 225, 183, 0.34), transparent 26%),
                radial-gradient(circle at top right, rgba(187, 225, 255, 0.26), transparent 24%),
                linear-gradient(180deg, #fffaf3 0%, #f8fbfe 100%);
        }
        .hero {
            padding: 1.45rem 1.5rem;
            border: 1px solid rgba(24, 70, 92, 0.12);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255, 246, 229, 0.96) 0%, rgba(244, 250, 255, 0.98) 100%);
            box-shadow: 0 14px 30px rgba(19, 32, 43, 0.08);
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.05;
        }
        .hero p {
            margin: 0.65rem 0 0;
            max-width: 72ch;
            font-size: 1.02rem;
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: 0.75rem;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            background: #143849;
            color: #fff;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1rem;
        }
        .button-row a {
            text-decoration: none !important;
        }
        .app-button {
            display: inline-block;
            padding: 0.72rem 1rem;
            border-radius: 14px;
            background: #13202b;
            color: #fff !important;
            font-weight: 600;
            border: 1px solid rgba(19, 32, 43, 0.12);
            box-shadow: 0 8px 18px rgba(19, 32, 43, 0.10);
        }
        .app-button.secondary {
            background: rgba(255, 255, 255, 0.85);
            color: #143849 !important;
        }
        .caption-card {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: #fbfcfe;
            border: 1px solid rgba(24, 70, 92, 0.12);
            min-height: 100%;
        }
        .fit-card {
            padding: 1rem;
            border-radius: 16px;
            background: #fffdf7;
            border: 1px solid rgba(204, 90, 45, 0.18);
            min-height: 100%;
            box-shadow: 0 8px 18px rgba(19, 32, 43, 0.05);
        }
        .fit-card h3 {
            margin-top: 0;
            margin-bottom: 0.45rem;
            font-size: 1.05rem;
        }
        .spotlight {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(204, 90, 45, 0.10), rgba(31, 122, 140, 0.08));
            border: 1px solid rgba(31, 122, 140, 0.12);
            margin: 0.4rem 0 1rem;
        }
        .spotlight h3 {
            margin: 0 0 0.35rem;
        }
        .mini-note {
            color: #43606c;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_links() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Application links")
    st.sidebar.markdown(f"- [GitHub repository]({REPO_BASE.rsplit('/blob/main', 1)[0]})")
    st.sidebar.markdown(f"- [Portfolio website]({PAGES_BASE})")
    st.sidebar.markdown(f"- [CV PDF]({CV_URL})")
    st.sidebar.markdown(f"- [Internship portfolio summary]({PAGES_BASE}/projects/results-highlights.html)")
    st.sidebar.markdown(f"- [Scripts library]({PAGES_BASE}/projects/scripts-library.html)")
    st.sidebar.info("For recruiters: start with the Figures or Internship fit tabs, then use Code for traceability.")


def filter_feature_models(feature_models: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter results")
    study = st.sidebar.selectbox("Study", sorted(feature_models["study"].unique()))
    level = st.sidebar.selectbox(
        "Feature level", sorted(feature_models["level"].unique())
    )
    subset = feature_models[
        (feature_models["study"] == study) & (feature_models["level"] == level)
    ].copy()

    coefficient = st.sidebar.selectbox(
        "Coefficient", sorted(subset["coefficient"].dropna().unique())
    )
    q_max = st.sidebar.slider("Maximum q-value", 0.0, 0.25, 0.05, 0.005)
    top_n = st.sidebar.slider("Rows to show", 5, 50, 15, 5)

    filtered = subset[(subset["coefficient"] == coefficient) & (subset["q"] <= q_max)].copy()
    filtered["abs_beta"] = filtered["beta"].abs()
    filtered = filtered.sort_values(["q", "abs_beta"], ascending=[True, False]).head(top_n)
    return filtered


def render_summary(filtered: pd.DataFrame) -> None:
    left, middle, right = st.columns(3)
    left.metric("Significant features shown", int(len(filtered)))
    if filtered.empty:
        middle.metric("Lowest q-value", "N/A")
        right.metric("Largest |beta|", "N/A")
        return
    middle.metric("Lowest q-value", f"{filtered['q'].min():.3g}")
    right.metric("Largest |beta|", f"{filtered['beta'].abs().max():.3f}")
    st.caption("These summary values update with your current filters so non-specialists can review the strongest signal quickly.")


def render_top_actions() -> None:
    st.markdown(
        f"""
        <div class="button-row">
            <a class="app-button" href="{PAGES_BASE}" target="_blank">Open portfolio website</a>
            <a class="app-button secondary" href="{CV_URL}" target="_blank">Open CV</a>
            <a class="app-button secondary" href="{PAGES_BASE}/projects/results-highlights.html" target="_blank">See results highlights</a>
            <a class="app-button secondary" href="{REPO_BASE.rsplit('/blob/main', 1)[0]}" target="_blank">Browse GitHub repository</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recruiter_brief() -> None:
    st.subheader("What this demo proves")
    first, second, third = st.columns(3)
    with first:
        st.markdown(
            """
            <div class="fit-card">
                <h3>Reproducible analytics</h3>
                Scripted workflows convert raw microbiome outputs into repeatable tables, figures, and validation checks.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with second:
        st.markdown(
            """
            <div class="fit-card">
                <h3>Statistical judgment</h3>
                The app surfaces effect sizes, uncertainty intervals, FDR-adjusted signals, and community-level tests.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with third:
        st.markdown(
            """
            <div class="fit-card">
                <h3>Decision-ready communication</h3>
                Outputs are translated into reviewable summaries, figure galleries, and direct links back to the code.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_spotlight(filtered: pd.DataFrame) -> None:
    if filtered.empty:
        return
    lead_row = filtered.sort_values(["q", "abs_beta"], ascending=[True, False]).iloc[0]
    direction = "higher" if lead_row["beta"] > 0 else "lower"
    st.markdown(
        f"""
        <div class="spotlight">
            <h3>Current spotlight: {lead_row['feature']}</h3>
            <p>
                Under the selected filters, <strong>{lead_row['feature']}</strong> shows one of the strongest signals with
                a beta of <strong>{lead_row['beta']:.3f}</strong>, indicating {direction} relative abundance/expression for
                the selected comparison. The adjusted q-value is <strong>{lead_row['q']:.3g}</strong>.
            </p>
            <p class="mini-note">
                This section helps a reviewer understand one concrete result before diving into the full table.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_to_use() -> None:
    st.subheader("Best way to review this app")
    st.markdown(
        """
        1. Start with the sidebar filters to choose a study and feature level.
        2. Use Feature models to inspect strongest signals and download a filtered table.
        3. Use Community tests to see broader treatment effects beyond single features.
        4. Use Figures and Code for fast visual review and traceability back to the repository.
        """
    )


def render_internship_fit() -> None:
    st.subheader("Why this is internship-ready")
    st.markdown(
        """
        - It demonstrates end-to-end ownership: data preparation, statistical analysis, visualization, and reporting.
        - It is reviewable by both technical and non-technical audiences.
        - It shows that the analysis is not just code-complete, but presentation-ready for collaboration and decision support.
        """
    )
    st.markdown(
        """
        **Best review path for a hiring team**

        1. Review the hero section and summary metrics for context.
        2. Open the Figures tab for visual quality and communication.
        3. Open the Code tab to confirm reproducibility and engineering depth.
        4. Use the portfolio and results links for the broader application narrative.
        """
    )


def render_feature_table(filtered: pd.DataFrame) -> None:
    st.subheader("Feature model results")
    if filtered.empty:
        st.info("No rows match the current filters. Increase the q-value threshold or change the coefficient.")
        return

    display = filtered[
        ["feature", "coefficient", "beta", "SE", "CI_low", "CI_high", "p", "q", "method"]
    ].rename(
        columns={
            "SE": "standard_error",
            "CI_low": "ci_low",
            "CI_high": "ci_high",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered results as CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name="filtered_feature_models.csv",
        mime="text/csv",
    )


def render_community_tests(community_tests: pd.DataFrame, study: str) -> None:
    st.subheader("Community-level tests")
    subset = community_tests[community_tests["study"] == study].copy()
    subset = subset.sort_values(["q", "partial_R2"], ascending=[True, False])
    st.dataframe(subset, use_container_width=True, hide_index=True)


def render_concordance(concordance: pd.DataFrame, study: str) -> None:
    st.subheader("Direct vs inferred gene concordance")
    subset = concordance[concordance["study"] == study].copy()
    subset = subset.sort_values("spearman_r", ascending=False).head(20)
    st.dataframe(subset, use_container_width=True, hide_index=True)


def render_gallery() -> None:
    st.subheader("Figure gallery")
    items = [
        ("GC gene effects", "GC_gene_effects.png", "Feature-level effects across GC samples."),
        (
            "GC OM by shrub",
            "GC_OM_vs_noOM_separate_shrub_groups.png",
            "Ordination split by shrub status for a treatment-focused view.",
        ),
        (
            "GC carbon category",
            "GC_carbon_category_with_stats.png",
            "Factorial contrast figure from the manuscript story workflow.",
        ),
    ]
    columns = st.columns(len(items))
    for column, (title, filename, caption) in zip(columns, items):
        with column:
            st.image(str(ASSET_DIR / filename), caption=title, use_container_width=True)
            st.markdown(f"<div class='caption-card'>{caption}</div>", unsafe_allow_html=True)


def render_repo_links() -> None:
    st.subheader("Code entry points")
    links = {
        "Prepare pipeline": f"{REPO_BASE}/scripts/prepare.py",
        "Core analysis": f"{REPO_BASE}/scripts/analyze.py",
        "Ordination detail": f"{REPO_BASE}/scripts/ordination_OM_by_shrub.py",
        "Story workflow": f"{REPO_BASE}/manuscript_story/scripts/analyze_story.py",
        "Portfolio scripts page": f"{PAGES_BASE}/projects/scripts-library.html",
        "Results highlights page": f"{PAGES_BASE}/projects/results-highlights.html",
    }
    for label, url in links.items():
        st.markdown(f"- [{label}]({url})")


def main() -> None:
    style_page()
    feature_models, community_tests, concordance = load_data()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Internship Demo</div>
            <h1>Microbiome Multi-Omics Results Explorer</h1>
            <p>
                Interactive Streamlit demo built from stable outputs in the GC and OSS workflows.
                It is designed to show recruiters and internship teams that the project has usable
                analytics, not just static figures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered = filter_feature_models(feature_models)
    render_sidebar_links()
    render_top_actions()
    selected_study = filtered["study"].iloc[0] if not filtered.empty else st.session_state.get("Study", "GC")
    render_recruiter_brief()
    render_summary(filtered)
    render_feature_spotlight(filtered)

    overview, community, concordance_tab, gallery, code, fit = st.tabs(
        ["Feature models", "Community tests", "Concordance", "Figures", "Code", "Internship fit"]
    )

    with overview:
        render_how_to_use()
        render_feature_table(filtered)
    with community:
        render_community_tests(community_tests, selected_study)
    with concordance_tab:
        render_concordance(concordance, selected_study)
    with gallery:
        render_gallery()
    with code:
        render_repo_links()
    with fit:
        render_internship_fit()


if __name__ == "__main__":
    main()