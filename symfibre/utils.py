#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Utility functions used for Python code
Author: Mathias Roesler
Date: 01/25
"""


def extract_coordinates(points, fibre_fct, sheet_fct, normal_fct):
    """Extracts the coordinates of the ldrb FiberSheetSystem at specific points

    Arguments:
    points -- np.array, list of point coordinates to evaluate
    the FiberSheetSystem functions.
    fibre_fct -- df.Function, FiberSheetSystem fibre function.
    sheet_fct -- df.Function, FiberSheetSystem sheet function.
    normal_fct -- df.Function, FiberSheetSystem normal function.

    Return:
    fibres -- np.array, list of fibre coordinates.
    sheets -- np.array, list of sheet coordinates.
    normals -- np.array, list of normal coordinates.

    Raises:

    """
    # Create empty lists
    fibres = []
    sheets = []
    normals = []

    for point in points:
        fibres.append(fibre_fct(point))
        sheets.append(sheet_fct(point))
        normals.append(normal_fct(point))

    return np.array(fibres), np.array(sheets), np.array(normals)
