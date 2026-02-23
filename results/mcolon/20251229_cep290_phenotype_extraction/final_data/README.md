# CEP290 Phenotype Clustering - Final Data

## Overview
This directory contains the final approved clustering results for all CEP290 experimental data.

## Experiments Included
Data from the following experiment IDs:
- 20250512
- 20251017_combined
- 20251106
- 20251112
- 20251113
- 20251205
- 20251212

## Files
- `embryo_data_with_labels.csv` - Full dataframe with all trajectory data and cluster assignments
- `embryo_cluster_labels.csv` - Simplified file containing only embryo IDs and their cluster labels

## Cluster Categories
The clustering identified 6 distinct phenotypic trajectories, grouped into broader categories:

### Main Categories (cluster_categories)
- **Low_to_High** - Embryos starting with low phenotype values that increase over time (clusters 0, 2)
- **High_to_Low** - Embryos starting with high phenotype values that decrease over time (clusters 1, 4)
- **Intermediate** - Embryos with intermediate, stable phenotype values (cluster 3)
- **Not Penetrant** - Embryos showing no significant phenotypic changes (cluster 5)

### Subcategories (cluster_subcategories)
More specific labels to distinguish between similar trajectory patterns:
- Low_to_High_A (cluster 0)
- Low_to_High_B (cluster 2)
- High_to_Low_A (cluster 1)
- High_to_Low_B (cluster 4)
- Intermediate (cluster 3)
- Not Penetrant (cluster 5)

## Missing Data (NaN values)
Some embryo records contain NaN values in the cluster assignment columns. These represent outliers that were removed during the clustering process to ensure robust cluster identification.

## Date
Analysis completed: 2025-12-29
