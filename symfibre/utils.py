#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Utility functions used for Python code
Author: Mathias Roesler
Date: 01/25
"""

import sys
import numpy as np
import pandas as pd

try:
    import dolfin as df
except ImportError:
    sys.stderr.write("Warning: no module named dolfin\n")


def write_ortho_file(file_path, fibres, sheets, normals):
    """Writes an ortho file based on the fibres, sheets, and normals

    Args:
    file_path -- str, path to the ortho file.
    fibres -- np.array(N, 3), coordinates of the fibres.
    sheets -- np.array(N, 3), coordinates of the sheets.
    normals -- np.array(N, 3), coordinates of the normals.

    Returns:

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
                f"{fibres[i, 0]:.6f} {fibres[i, 1]:.6f} {fibres[i, 2]:.6f} "
                f"{sheets[i, 0]:.6f} {sheets[i, 1]:.6f} {sheets[i, 2]:.6f} "
                f"{normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}\n"
            )


def extract_coordinates(points, fibre_fct, sheet_fct):
    """Extracts the coordinates of the ldrb FiberSheetSystem at specific points

    Normal vectors are calculated using the cross product between the fibre and
    sheet vectors.

    Args:
    points -- np.array(N, 3), list of point coordinates to evaluate
    the FiberSheetSystem functions.
    fibre_fct -- df.Function, FiberSheetSystem fibre function.
    sheet_fct -- df.Function, FiberSheetSystem sheet function.

    Returns:
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

        # Normalise fibre and sheet vectors
        fibre_vec /= np.linalg.norm(fibre_vec)
        sheet_vec /= np.linalg.norm(sheet_vec)

        # Compute normal vector and normalise it
        normal_vec = np.cross(fibre_vec, sheet_vec)
        normal_vec /= np.linalg.norm(normal_vec)

        # Recompute sheet vector to ensure orthogonality
        sheet_vec = np.cross(fibre_vec, normal_vec)

        fibres.append(fibre_vec)
        sheets.append(sheet_vec)
        normals.append(normal_vec)

    return np.array(fibres), np.array(sheets), np.array(normals)


def element_centres(mesh):
    """Extracts element centres from a given mesh

    Args:
    mesh -- df.Mesh, mesh to extract the centres from.

    Returns:
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


def fibres_from_ortho(ortho_file):
    """Extracts fibre vectors and angles from an ortho file

    If the ortho file contains the angle at the end of the line the angle
    is read directly otherwise it is calculated relative to the XY plane.

    Args:
    ortho_file -- str, path to the ortho file.

    Returns:
    fibres -- np.array(float), array of fibre vectors.
    angles -- np.array(float), orientation of the fibre based on the
    z-axis, in degrees.

    Raises:
    FileNotFoundError, if the ortho file is not found.

    """
    try:
        with open(ortho_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise e

    # Initialise arrays with 0s
    fibres = np.zeros((len(lines) - 1, 3), dtype=np.float64)
    angles = np.zeros((len(lines) - 1, 1), dtype=np.float64)

    for i, line in enumerate(lines):
        if i == 0:  # Skip the number of lines
            continue

        split_line = line.strip().split()  # Split line into individual values

        fibres[i - 1] = np.array(  # Get the fibre values
            [float(ele) for ele in split_line[0:3]]
        )

        if len(split_line) == 10:
            # If the angles are at the end of the line
            angles[i - 1] = split_line[-1]

        else:
            # Calculate angles relative to the XY plane
            angles[i - 1] = np.degrees(
                np.arccos(
                    np.dot(
                        fibres[i - 1],
                        [0, 0, 1],
                    )
                )
            )

        if angles[i - 1] > 90:
            angles[i - 1] -= 180

    return fibres, abs(angles)


def fibres_from_scaffold_ortho(ortho_file, outer_points_path, mesh_points):
    """Extracts fibre vectors and angles from the scaffold ortho file

    Args:
    ortho_file -- str, path to the ortho file.
    outer_points_path -- str, path to the outer points, only used for
    scaffolds because they do not have centerlines, default value "".
    mesh_points -- np.array(float), coordinates of the mesh points.

    Returns:
    fibres -- np.array(float), array of fibre vectors.
    angles -- np.array(float), orientation of the fibre based on the
    z-axis, in degrees.

    Raises:
    FileNotFoundError, if the ortho file is not found.
    FileNotFoundError, if the outer points file is not found.
    ValueError, if the number of mesh points is different from the number of
    fibres.

    """
    try:
        with open(ortho_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise e

    try:
        data_frame = pd.read_csv(outer_points_path)
    except FileNotFoundError as e:
        raise e
    # Get the vtk IDs of the outer points
    outer_points_ids = np.array(data_frame[data_frame.columns[0]])

    if len(lines) - 1 != len(mesh_points):
        raise ValueError("number of fibres and mesh points should be equal")

    # Initialise arrays with 0s
    fibres = np.zeros((len(lines) - 1, 3), dtype=np.float64)
    angles = np.zeros((len(lines) - 1, 1), dtype=np.float64)

    for i, line in enumerate(lines):
        if i == 0:  # Skip the number of lines
            continue

        split_line = line.strip().split()  # Split line into individual values

        fibres[i - 1] = np.array(  # Get the fibre values
            [float(ele) for ele in split_line[0:3]]
        )

        if i - 1 in outer_points_ids:
            # Outer layer point are 0 degrees relative to the centerline
            angles[i - 1] = 0
        elif mesh_points[i - 1, 2] > 1.99 or mesh_points[i - 1, 2] < 0.25:
            # For points near the ovaries or cervix estimate XY plane angle
            angles[i - 1] = np.degrees(
                np.arccos(
                    np.dot(
                        fibres[i - 1],
                        [0, 0, 1],
                    )
                )
            )
        else:
            # Other angles are circumferential and 90 degrees
            angles[i - 1] = 90

        if angles[i - 1] > 90:
            angles[i - 1] -= 180

    return fibres, abs(angles)
