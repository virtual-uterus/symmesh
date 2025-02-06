#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Utility functions used for Python code
Author: Mathias Roesler
Date: 01/25
"""

import numpy as np
import dolfin as df


def write_ortho_file(file_path, fibres, sheets, normals):
    """Writes an ortho file based on the fibres, sheets, and normals

    Arguments:
    file_path -- str, path to the ortho file.
    fibres -- np.array(N, 3), coordinates of the fibres.
    sheets -- np.array(N, 3), coordinates of the sheets.
    normals -- np.array(N, 3), coordinates of the normals.

    Return:

    Raises:
    ValueError -- if the size of the arrays does not match.
    ValueError -- if the arrays do not have three coordinates.

    """
    if not (fibres.shape == sheets.shape == normals.shape):
        raise ValueError("input arrays must have the same shape.")

    if not fibres.shape[1] == 3:
        raise ValueError("input arrays must have 3 coordinates.")

    n = fibres.shape[0]

    with open(file_path, "w") as f:
        f.write(f"{n}\n")  # Write the number of entries

        for i in range(n):
            f.write(
                f"{fibres[i, 0]:.1f} {fibres[i, 1]:.1f} {fibres[i, 2]:.1f} "
                f"{sheets[i, 0]:.1f} {sheets[i, 1]:.1f} {sheets[i, 2]:.1f} "
                f"{normals[i, 0]:.1f} {normals[i, 1]:.1f} {normals[i, 2]:.1f}\n"
            )


def extract_coordinates(points, fibre_fct, sheet_fct):
    """Extracts the coordinates of the ldrb FiberSheetSystem at specific points

    Normal vectors are calculated using the cross product between the fibre and
    sheet vectors.

    Arguments:
    points -- np.array(N, 3), list of point coordinates to evaluate
    the FiberSheetSystem functions.
    fibre_fct -- df.Function, FiberSheetSystem fibre function.
    sheet_fct -- df.Function, FiberSheetSystem sheet function.

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

    for i, point in enumerate(points):
        fibre_vec = fibre_fct(point)
        sheet_vec = sheet_fct(point)

        fibres.append(fibre_fct(point))
        normals.append(np.cross(fibre_vec, sheet_vec))
        sheets.append(np.cross(fibre_vec, normals[i]))
    return np.array(fibres), np.array(sheets), np.array(normals)


def element_centres(mesh):
    """Extracts element centres from a given mesh

    Arguments:
    mesh -- df.Mesh, mesh to extract the centres from.

    Return:
    centres -- np.array(N, 3), element centres.

    Raises:
    TypeError -- if the mesh is not a df.Mesh

    """
    try:
        assert type(mesh) is type(df.Mesh())

    except AssertionError:
        raise TypeError("mesh should be a df.Mesh")

    centres = np.zeros((mesh.num_cells(), 3))  # Initialise empty array

    for i, ele in enumerate(mesh.cells()):
        centres[i] = np.mean(mesh.coordinates()[ele], axis=0)

    return centres
