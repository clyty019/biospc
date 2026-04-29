# Bio-SPC

**Bio-SPC** — A Python library for identifying cliff points along time/pseudotime trajectories based on single-cell RNA-seq data.

## Overview

Bio-SPC leverages a 1D ResNet architecture to perform feature extraction and dimensionality reduction on single-cell gene expression matrices. By adopting a sliding-window scanning strategy across the time/pseudotime axis, it integrates three quantitative metrics, including Bhattacharyya distance, Wasserstein distance, and Isolation Forest anomaly score. These indicators are adaptively fused into a unified CPI (Cliff Point Index) curve via the entropy weight method. This framework enables the automatic detection of critical cliff points during biological processes including development, senescence, and disease progression. Corresponding key cells can be further extracted for downstream analytical workflows.


## Install

```bash
pip install biospc
```

## Quick Start

```python
import scanpy as sc
import pandas as pd
from biospc import bio_spc_pipeline

# 1. Loading data
adata = sc.read("your_data.h5ad")
metadata = pd.read_csv("your_metadata.csv", index_col=0)
adata.obs['Pseudotime'] = metadata['Pseudotime']

# 2. Running Bio-SPC Pipeline
peak_report = bio_spc_pipeline(
    adata=adata,
    time_col='Pseudotime',
    random_seed=42,
    do_plot=True,
)

# 3. Viewing Results
print(peak_report[['Rank', 't', 'Confidence']])
```


## Core Parameters

| Parameter | Default | Description |
|-----------|--------|------------|
| `time_col` | `'Pseudotime'` | Name of the pseudotime column |
| `out_dim` | `30` | Output dimension of the ResNet |
| `hidden_dim` | `128` | Hidden layer dimension of the ResNet |
| `epochs` | `1000` | Number of training epochs for the ResNet |
| `lr` | `1e-4` | Learning rate |
| `step_ratio` | `0.02` | Step size ratio for the sliding window |
| `safe_margin_ratio` | `0.02` | Safety margin ratio at the edges |
| `min_cells_ratio` | `0.025` | Minimum cell count ratio |
| `smooth_window` | `7` | Window size for Savitzky–Golay smoothing |
| `prominence` | `0.1` | Peak prominence |
| `distance` | `5` | Minimum distance between peaks |
| `extract_ratio` | `0.04` | Half-window ratio for cell extraction |
| `random_seed` | `42` | Global random seed |

## Output

- **Peak Report**: A DataFrame containing Rank, time/pseudotime point `t`, and Confidence.  
- **CSV File**: A list of cell barcodes corresponding to each cliff point.  
- **Visualization**: CPI curve with multi-peak annotations (optional).  

## Dependencies

- Python >= 3.8
- numpy, pandas, scanpy, anndata
- torch >= 1.10
- scikit-learn, scipy, matplotlib

## License

MIT
