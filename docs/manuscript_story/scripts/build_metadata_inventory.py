#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def assign_category(name, study):
    n = name.lower()
    if n in {'sample_id','treatment','block','shrub','watering','fertilizer','sample_type','season','context','phase','cropxom','typexseason','candidate_plot','final.height'}:
        return 'Design / grouping'
    if any(k in n for k in ['rna', 'antioxidant', 'exopolysaccharide', 'nutrient', 'osmolytes', 'phytohormone', 'sum']):
        return 'RNA / physiology'
    if any(k in n for k in ['pern', 'perc', 'c_mg', 'n_mg', 'n/c', 'pH', 'clay', 'sand', 'silt', 'height', 'biomass', 'grain', 'panicles', 'millet']):
        return 'Plant / soil chemistry'
    if any(k in n for k in ['al', 'ca', 'cu', 'fe', 'k', 'mg', 'mn', 'na', 'p', 'zn']):
        return 'Soil chemistry'
    if any(k in n for k in ['tplfa', 'g.pos', 'g.neg', 'actino', 'gen.bact', 'tbact', 's/m', 'cy/pr', 'amf', 'sap.fun', 'tfungi', 'f/b']):
        return 'Microbial community / PLFA'
    if any(k in n for k in ['ap', 'glu', 'nag']):
        return 'Enzyme / substrate chemistry'
    return 'Other'


def prepare_df(df, study):
    df = df.copy()
    if 'sample_id' in df.columns:
        df = df.set_index('sample_id')
    sample_order = df.index.tolist()
    columns = [c for c in df.columns if c not in {'sample_id','candidate_plot','treatment','block','shrub','watering','fertilizer','sample_type','season','context','phase','cropxom','typexseason'}]
    columns = [c for c in columns if c and not c.startswith('Unnamed')]
    # Keep design columns in a separate short list to display as essential metadata
    design_cols = [c for c in df.columns if c in {'treatment','block','shrub','watering','fertilizer','sample_type','season','context','phase','cropxom','typexseason','candidate_plot'}]
    cols = design_cols + columns
    df = df[cols].copy()
    for c in df.columns:
        if c in {'sample_id','candidate_plot'}:
            continue
        df[c] = df[c].apply(lambda x: 1 if pd.notna(x) and str(x).strip() not in {'', 'NA', 'nan', 'NaN'} else 0)
    avail = df.copy()
    # normalize for plotting: rows as variables, columns as samples
    plot_df = avail.T
    plot_df = plot_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    plot_df['category'] = [assign_category(c, study) for c in plot_df.index]
    plot_df = plot_df.sort_values(['category', 'category'], kind='mergesort')
    return plot_df


def save_metadata_inventory(df, study, outpath):
    # preserve the original values as 0/1 matrix for plotting
    plot_df = prepare_df(df, study)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # left: availability matrix
    heat = plot_df.drop(columns=['category']).copy()
    heat = heat.reindex(sorted(heat.index, key=lambda x: (x.lower())))
    arr = heat.to_numpy(dtype=float)
    # Create category row labels for the left heatmap using actual variable names
    row_labels = [str(x) for x in heat.index]
    y_labels = [x for x in row_labels]

    im = axes[0].imshow(arr, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title(f'{study} metadata availability', fontsize=14)
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Variables')
    axes[0].set_yticks(range(len(y_labels)))
    axes[0].set_yticklabels(y_labels, fontsize=7)
    axes[0].set_xticks(range(len(heat.columns)))
    axes[0].set_xticklabels([str(x) for x in heat.columns], rotation=90, fontsize=7)
    for i in range(len(y_labels)):
        axes[0].axhline(i - 0.5, color='white', linewidth=0.2)

    # category summary bar
    category_counts = plot_df['category'].value_counts().sort_index()
    axes[1].bar(category_counts.index, category_counts.values, color='#5C7CFA')
    axes[1].set_title(f'{study} variable counts by category', fontsize=12)
    axes[1].set_ylabel('Number of variables')
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    outdir = Path('/fs/ess/PAS1212/Afaf/000_combined_OSS_GC/fnal results/manuscript_story/figures')
    outdir.mkdir(parents=True, exist_ok=True)

    for study, path in {
        'GC': '/fs/ess/PAS1212/Afaf/000_combined_OSS_GC/fnal results/inputs/GC_metadata.tsv',
        'OSS': '/fs/ess/PAS1212/Afaf/000_combined_OSS_GC/fnal results/inputs/OSS_metadata.tsv',
    }.items():
        df = pd.read_csv(path, sep='\t')
        save_metadata_inventory(df, study, outdir / f'{study}_metadata_inventory.png')
        print(f'Wrote {study}_metadata_inventory.png')
