#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constants.py

Constants for the symmesh package
Author: Mathias Roesler
Date: 01/25
"""

import os

# Define global constants
HOME = os.path.expanduser("~")
BASE = "Documents/phd"
CONVERSION_IDX = [  # List of node indices for each tetrahedra
    [0, 1, 2, 4],  # Tetrahedron 1
    [0, 2, 3, 4],  # Tetrahedron 2
    [1, 2, 4, 5],  # Tetrahedron 3
    [2, 3, 4, 7],  # Tetrahedron 4
    [2, 4, 5, 6],  # Tetrahedron 5
    [2, 4, 6, 7],  # Tetrahedron 6
]
QUALITY_METRIC_MAP = {  # Mapping for quality metrics
    "ar": "Aspect Ratio",
    "ja": "Jacobian",
    "sj": "Scaled Jacobian",
    "mr": "Mean Ratio",
}
RES_DICT = {  # Number of elements in the scaffolds
    "uterus_scaffold_scaled_1": 1258,
    "uterus_scaffold_scaled_2": 9984,
    "uterus_scaffold_scaled_3": 14976,
    "uterus_scaffold_scaled_4": 22464,
    "uterus_scaffold_scaled_5": 33696,
}
# Marks for facets
OUTER_MARK = 1
INNER_MARK = 2
BASE_MARK = 3
MARKERS = {
    "base": BASE_MARK,
    "lv": INNER_MARK,
    "epi": OUTER_MARK,
}
# Plot constants
LEFT = 0.22
BOTTOM = 0.17
RIGHT = 0.80
