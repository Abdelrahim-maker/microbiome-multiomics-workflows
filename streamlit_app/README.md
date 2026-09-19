# Streamlit Demo

This app turns selected outputs from the microbiome multi-omics workflow into a lightweight interactive demo for internship applications.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select this repository and branch `main`.
4. Set the entry point to `streamlit_app/app.py`.

## Data bundled with the app

- `adjusted_feature_models.tsv`
- `adjusted_community_tests.tsv`
- `direct_inferred_gene_concordance.tsv`
- Selected preview figures for the gallery