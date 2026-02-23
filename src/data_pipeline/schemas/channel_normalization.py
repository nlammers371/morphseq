"""
Schema definition for channel name normalization.

This module defines the mapping from microscope-specific channel names
to standardized channel names used throughout the pipeline.
"""

# Map raw microscope channel names to standardized names
CHANNEL_NORMALIZATION_MAP = {
    # YX1 microscope channel names
    'EYES - Dia': 'BF',
    'EYES - GFP': 'GFP',
    'EYES - RFP': 'RFP',

    # Keyence microscope channel names
    'Brightfield': 'BF',
    'GFP': 'GFP',
    'RFP': 'RFP',
    'mCherry': 'RFP',

    # Add other microscope-specific mappings as needed
}

# Valid standardized channel names
VALID_CHANNEL_NAMES = [
    'BF',      # Brightfield
    'GFP',     # Green fluorescent protein
    'RFP',     # Red fluorescent protein
    'CFP',     # Cyan fluorescent protein
    'YFP',     # Yellow fluorescent protein
]

# Brightfield channel identifiers (for special processing)
BRIGHTFIELD_CHANNELS = ['BF']
