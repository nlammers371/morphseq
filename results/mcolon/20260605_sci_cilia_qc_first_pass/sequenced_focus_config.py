"""Shared sequenced-focus plotting conventions for cilia QC first pass."""

STAGE_GRIDS = {
    "b9d2": [14, 18, 30, 48],
    "cep290": [18, 24, 30, 48],
}

TARGET_HPF_WINDOWS = {
    "b9d2": [14, 18, 30, 48],
    "cep290": [18, 24, 30, 48],
}

HOMO_PHENOTYPE_ORDER = {
    "b9d2": ["CE", "HTA"],
    "cep290": ["High_to_Low", "Low_to_High"],
}

PHENOTYPE_COLORS = {
    "b9d2": {
        "CE": "#1b9e77",
        "HTA": "#d95f02",
        "wildtype": "#1f77b4",
    },
    "cep290": {
        "High_to_Low": "#E76FA2",
        "Low_to_High": "#2FB7B0",
        "Not Penetrant": "#BBBBBB",
    },
}

PHENOTYPE_ALIASES = {
    "High_to_Low": "HtL",
    "Low_to_High": "LtH",
    "Not Penetrant": "NotPen",
    "CE": "CE",
    "HTA": "HTA",
    "wildtype": "wt",
}
