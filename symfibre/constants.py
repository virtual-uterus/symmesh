#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constants.py

Constants for the symfibre package
Author: Mathias Roesler
Date: 01/25
"""

import os

# Define global constants
HOME = os.path.expanduser("~")
BASE = "Documents/phd"

# Marks for facets
OUTER_MARK = 1
INNER_MARK = 2
BASE_MARK = 3
MARKERS = {
    "base": BASE_MARK,
    "lv": INNER_MARK,
    "epi": OUTER_MARK,
}
