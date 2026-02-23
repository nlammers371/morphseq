"""
Schema definition for plate metadata.

This module defines required columns for the plate layout metadata table,
which contains biological and experimental annotations for each well.
"""

REQUIRED_COLUMNS_PLATE_METADATA = [
    # Core identifiers
    'experiment_id',
    'well_id',
    'well_index',

    # Biological metadata
    'genotype',
    'treatment',              # or 'chem_perturbation'
    'start_age_hpf',

    # Experimental conditions
    'temperature',          # Critical for developmental timing normalization
    'medium',
]
