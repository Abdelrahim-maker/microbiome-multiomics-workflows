from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ASSET_DIR = APP_DIR / "assets"
REPO_BASE = "https://github.com/Abdelrahim-maker/microbiome-multiomics-workflows/blob/main"


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
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(24, 70, 92, 0.14);
            border-radius: 18px;
            background: linear-gradient(135deg, #fff7e8 0%, #f6fbff 100%);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0.5rem 0 0;
            max-width: 70ch;
        }
        .caption-card {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: #fbfcfe;
            border: 1px solid rgba(24, 70, 92, 0.12);
            min-height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        "Full scripts library": f"{REPO_BASE}/docs/projects/scripts-library.html",
    }
    for label, url in links.items():
        st.markdown(f"- [{label}]({url})")


def main() -> None:
    style_page()
    feature_models, community_tests, concordance = load_data()

    st.markdown(
        """
        <div class="hero">
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
    selected_study = filtered["study"].iloc[0] if not filtered.empty else st.session_state.get("Study", "GC")
    render_summary(filtered)

    overview, community, concordance_tab, gallery, code = st.tabs(
        ["Feature models", "Community tests", "Concordance", "Figures", "Code"]
    )

    with overview:
        render_feature_table(filtered)
    with community:
        render_community_tests(community_tests, selected_study)
    with concordance_tab:
        render_concordance(concordance, selected_study)
    with gallery:
        render_gallery()
    with code:
        render_repo_links()


if __name__ == "__main__":
    main()