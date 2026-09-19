"""Additional treatment-linked network and MAG/function analyses.

This script complements the main OLS + permutation workflow by constructing:
- gene co-occurrence networks within carbon-function groups,
- pathway-level co-occurrence networks,
- MAG-to-function response summaries for shrub and OM contrasts,
- multifunctional MAG summaries with treatment-linked enrichment.

The implementation uses robust, transparent pairwise correlation and linear-model
coefficients already present in the project inputs, without requiring external
MaAsLin2 or ALDEx2 dependencies.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent


def read_table(name, *, index_col=None, sep='\t'):
    return pd.read_csv(OUT / 'tables' / name if (OUT / 'tables' / name).exists() else OUT / 'inputs' / name,
                       sep=sep,
                       index_col=index_col)


def save(df, name):
    df.to_csv(OUT / 'tables' / name, sep='\t', index=False)


def log2p(x):
    return np.log2(np.asarray(x, dtype=float) + 0.5)


def summarize_group_corr(df, group_key, factor_name, study, level_name, min_r=0.6, min_n=6):
    records = []
    for mechanism, genes in df.groupby(group_key):
        genes = list(genes)
        if len(genes) < 2:
            continue
        X = df.loc[genes].astype(float)
        if X.shape[1] < min_n:
            continue
        mat = X.corr(method='spearman')
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                a, b = genes[i], genes[j]
                r, p = spearmanr(X.loc[a].to_numpy(), X.loc[b].to_numpy())
                if np.isnan(r):
                    continue
                if abs(r) >= min_r and p < 0.05:
                    records.append({
                        'study': study,
                        'level': level_name,
                        'group': mechanism,
                        'factor': factor_name,
                        'gene_a': a,
                        'gene_b': b,
                        'rho': float(r),
                        'p': float(p),
                        'sign': 'positive' if r > 0 else 'negative',
                    })
    return pd.DataFrame(records)


def summarize_pathway_corr(df, table, study, min_r=0.65, min_n=6):
    records = []
    for pathway, genes in table.groupby('pathway'):
        ids = genes['image_gene'].tolist()
        common = [g for g in ids if g in df.index]
        if len(common) < 2:
            continue
        X = df.loc[common].astype(float)
        if X.shape[1] < min_n:
            continue
        mat = X.corr(method='spearman')
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                a, b = common[i], common[j]
                r, p = spearmanr(X.loc[a].to_numpy(), X.loc[b].to_numpy())
                if np.isnan(r):
                    continue
                if abs(r) >= min_r and p < 0.05:
                    records.append({
                        'study': study,
                        'pathway': pathway,
                        'gene_a': a,
                        'gene_b': b,
                        'rho': float(r),
                        'p': float(p),
                        'sign': 'positive' if r > 0 else 'negative',
                    })
    return pd.DataFrame(records)


def fit_treatment_effects(df, meta, formula_terms):
    """Return per-feature treatment coefficients for the specified design terms."""
    rows = []
    for feature in df.index:
        y = np.asarray(df.loc[feature].astype(float), dtype=float)
        X = pd.DataFrame(index=meta.index)
        for term in formula_terms:
            if term == 'Intercept':
                X['Intercept'] = 1.0
            else:
                X[term] = meta[term].astype(float)
        X = X.fillna(0)
        Xmat = np.asarray(X)
        coefs = np.linalg.pinv(Xmat.T @ Xmat) @ Xmat.T @ y
        for j, term in enumerate(X.columns):
            if term == 'Intercept':
                continue
            rows.append({
                'feature': feature,
                'coefficient': term,
                'beta': float(coefs[j]),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Load metadata and feature taxonomy
    feature_labels = pd.read_csv(OUT / 'tables' / 'feature_labels.tsv', sep='\t')
    direct = pd.read_csv(OUT / 'tables' / 'direct_read_MAG_gene_contributions.tsv', sep='\t')
    gene_mech = feature_labels[['image_gene', 'mechanism']].drop_duplicates().rename(columns={'image_gene': 'gene'})
    direct = direct.merge(gene_mech, on='gene', how='left')
    direct['function_group'] = direct['mechanism']
    direct.loc[direct['gene'].str.contains('glg', case=False, na=False), 'function_group'] = 'Glycogen'
    direct.loc[direct['gene'].str.contains('pha', case=False, na=False), 'function_group'] = 'PHA'
    direct.loc[direct['gene'].str.contains(r'^(ots|tre)', case=True, na=False), 'function_group'] = 'Trehalose'

    # Build product analyses for GC and OSS
    all_gene_edges = []
    all_pathway_edges = []
    mag_function_responses = []
    enriched_mags = []

    for study in ['GC', 'OSS']:
        meta = pd.read_csv(OUT / 'inputs' / f'{study}_metadata.tsv', sep='\t', index_col=0)
        if study == 'GC':
            trait_cols = ['shrub', 'organic_matter', 'watering', 'phase']
            formula_terms = ['Intercept', 'shrub', 'organic_matter', 'watering', 'phase']
        else:
            trait_cols = ['shrub', 'fertilizer', 'context']
            formula_terms = ['Intercept', 'shrub', 'fertilizer', 'context']

        # Gene co-occurrence network for carbon genes
        df = pd.read_csv(OUT / 'inputs' / f'{study}_direct_gene_TPM.tsv', sep='\t', index_col=0)
        gene_map = feature_labels[['image_gene', 'mechanism']].drop_duplicates().set_index('image_gene')['mechanism']
        keep = [g for g in df.index if g in gene_map.index]
        X = df.loc[keep].astype(float)
        X = X.loc[:, meta.index]
        if X.shape[0] > 1:
            # Compute network within each mechanism block
            for mechanism in sorted(gene_map.loc[keep].unique()):
                genes = [g for g in keep if gene_map.get(g) == mechanism]
                if len(genes) < 2:
                    continue
                sub = X.loc[genes, :]
                if sub.shape[1] < 6:
                    continue
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        a, b = genes[i], genes[j]
                        r, p = spearmanr(np.log2(sub.loc[a].to_numpy() + 0.5), np.log2(sub.loc[b].to_numpy() + 0.5))
                        if np.isnan(r):
                            continue
                        if abs(r) >= 0.6 and p < 0.05:
                            all_gene_edges.append({
                                'study': study,
                                'mechanism': mechanism,
                                'gene_a': a,
                                'gene_b': b,
                                'rho': float(r),
                                'p': float(p),
                                'sign': 'positive' if r > 0 else 'negative'
                            })

        # Pathway network using gene-to-pathway mapping
        path_map = feature_labels[['image_gene', 'pathway']].drop_duplicates().set_index('image_gene')['pathway']
        path_rows = []
        for pathway in sorted(path_map.loc[keep].unique()):
            genes = [g for g in keep if path_map.get(g) == pathway]
            if len(genes) < 2:
                continue
            sub = X.loc[genes, :]
            if sub.shape[1] < 6:
                continue
            for i in range(len(genes)):
                for j in range(i + 1, len(genes)):
                    a, b = genes[i], genes[j]
                    r, p = spearmanr(np.log2(sub.loc[a].to_numpy() + 0.5), np.log2(sub.loc[b].to_numpy() + 0.5))
                    if np.isnan(r):
                        continue
                    if abs(r) >= 0.65 and p < 0.05:
                        path_rows.append({
                            'study': study,
                            'pathway': pathway,
                            'gene_a': a,
                            'gene_b': b,
                            'rho': float(r),
                            'p': float(p),
                            'sign': 'positive' if r > 0 else 'negative',
                        })
        all_pathway_edges.extend(path_rows)

        # MAG-function response summaries
        eff = pd.read_csv(OUT / 'tables' / 'adjusted_feature_models.tsv', sep='\t')
        eff = eff[(eff['study'] == study) & (eff['analysis'] == 'main') & (eff['level'] == 'gene') & eff['feature'].notna()].copy()
        eff = eff[eff['coefficient'].isin(['C(shrub)[T.Shrub]', 'C(organic_matter)[T.noOM]'])].copy() if study == 'GC' else eff[eff['coefficient'].isin(['C(shrub)[T.Shrub]', 'C(fertilizer)[T.1x]'])].copy()
        eff = eff.merge(feature_labels[['image_gene', 'mechanism']].drop_duplicates(), left_on='feature', right_on='image_gene', how='left')
        eff['response_function_group'] = eff['mechanism']
        eff.loc[eff['feature'].str.contains('glg', case=False, na=False), 'response_function_group'] = 'Glycogen'
        eff.loc[eff['feature'].str.contains('pha', case=False, na=False), 'response_function_group'] = 'PHA'
        eff.loc[eff['feature'].str.contains(r'^(ots|tre)', case=True, na=False), 'response_function_group'] = 'Trehalose'

        direct_study = direct[direct['study'] == study].copy()
        merged = direct_study.merge(eff[['study','feature','coefficient','beta','q','response_function_group']],
                                   left_on=['study', 'gene'], right_on=['study', 'feature'], how='left')
        merged = merged[merged['coefficient'].notna()].copy()
        if not merged.empty:
            def safe_weighted_beta(d):
                w = d['mean_direct_TPM'].to_numpy(dtype=float)
                if np.sum(w) <= 0:
                    return np.nan
                return float(np.average(d['beta'].to_numpy(dtype=float), weights=w))

            w = merged.groupby(['study', 'MAG', 'response_function_group', 'coefficient'], as_index=False).apply(
                lambda d: pd.Series({'weighted_beta': safe_weighted_beta(d)})
            ).reset_index()
            w = w.rename(columns={'response_function_group': 'function_group'})
            w = w.dropna(subset=['weighted_beta'])
            if not w.empty:
                mag_function_responses.append(w)

                # top enriched MAGs by mean weighted response per function group
                top = w.sort_values(['function_group', 'weighted_beta'], ascending=[True, False]).groupby('function_group').head(5)
                enriched_mags.append(top)

    if all_gene_edges:
        all_gene_edges = pd.DataFrame(all_gene_edges)
        save(all_gene_edges, 'gene_cooccurrence_network.tsv')
    else:
        pd.DataFrame(columns=['study','mechanism','gene_a','gene_b','rho','p','sign']).to_csv(OUT / 'tables' / 'gene_cooccurrence_network.tsv', sep='\t', index=False)

    if all_pathway_edges:
        all_pathway_edges = pd.DataFrame(all_pathway_edges)
        save(all_pathway_edges, 'pathway_cooccurrence_network.tsv')
    else:
        pd.DataFrame(columns=['study','pathway','gene_a','gene_b','rho','p','sign']).to_csv(OUT / 'tables' / 'pathway_cooccurrence_network.tsv', sep='\t', index=False)

    if mag_function_responses:
        mag_function = pd.concat(mag_function_responses, ignore_index=True)
        save(mag_function, 'MAG_function_response.tsv')
    else:
        pd.DataFrame(columns=['study','MAG','function_group','coefficient','weighted_beta']).to_csv(OUT / 'tables' / 'MAG_function_response.tsv', sep='\t', index=False)

    if enriched_mags:
        enriched = pd.concat(enriched_mags, ignore_index=True)
        save(enriched, 'functionally_enriched_MAGs.tsv')
    else:
        pd.DataFrame(columns=['study','MAG','function_group','coefficient','weighted_beta']).to_csv(OUT / 'tables' / 'functionally_enriched_MAGs.tsv', sep='\t', index=False)

    # Convenience plots
    if (OUT / 'tables' / 'gene_cooccurrence_network.tsv').exists():
        gnet = pd.read_csv(OUT / 'tables' / 'gene_cooccurrence_network.tsv', sep='\t')
        if not gnet.empty:
            for study, sub in gnet.groupby('study'):
                d = sub.groupby(['mechanism', 'sign']).size().reset_index(name='count')
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=d, x='mechanism', y='count', hue='sign', ax=ax)
                ax.set_title(f'{study}: gene co-occurrence edges by mechanism')
                ax.tick_params(axis='x', rotation=45)
                fig.tight_layout()
                fig.savefig(OUT / 'figures' / f'{study}_gene_network_summary.png', dpi=180)
                plt.close(fig)

    if (OUT / 'tables' / 'MAG_function_response.tsv').exists():
        magfun = pd.read_csv(OUT / 'tables' / 'MAG_function_response.tsv', sep='\t')
        if not magfun.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=magfun, x='function_group', y='weighted_beta', hue='study', ax=ax)
            ax.set_title('MAG-to-function response strength')
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            fig.savefig(OUT / 'figures' / 'MAG_function_response_boxplot.png', dpi=180)
            plt.close(fig)

    print('Network and MAG-function analyses complete')
