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

from symmesh.constants import CONVERSION_IDX

try:
    import dolfin as df
except ImportError:
    sys.stderr.write("Warning: no module named dolfin\n")


def get_range(num_range):
    """Converts the input range into a list of numbers

    Arguments:
    num_range -- str, range of number from the input argument.

    Return:
    num_list -- list[int], list of numbers extracted from the range.

    """
    if len(num_range) == 1:
        split = num_range[0].split("-")
        if len(split) == 1:
            # Single number
            num_list = int(num_range[0])
        else:
            # Range
            num_list = [i for i in range(int(split[0]), int(split[1]) + 1)]
    else:
        # Convert to list to int
        num_list = [int(i) for i in num_range]

    return num_list


def get_indices(normal_vector, elements, plane_distance, centre_norm):
    """Finds the indices of the elements to annotate with the thickness.

    Arguments:
    normal_vector -- ndarray, normal vector for the plane.
    elements -- ndarray, list of elements in the mesh to sort.
    plane_distance -- int, distance between origin and cut planes.
    centre_norm -- float, norm of the normal vector.

    Return:
    idx_list -- np.array[int], list of indices of the correct elements.

    """
    ele_dot_prod = np.dot(normal_vector, elements)

    # Get points between the two planes
    dist_to_first_plane = (ele_dot_prod + plane_distance) / centre_norm
    dist_to_second_plane = (ele_dot_prod - plane_distance) / centre_norm

    idx_list = [
        idx
        for idx, (a, b) in enumerate(
            zip(
                dist_to_first_plane > 0.0,
                dist_to_second_plane < 0.0,
            )
        )
        if a and b
    ]

    return np.array(idx_list, dtype=int)


def convert_connections(cube_node_list):
    """Converts the connections of the cubic element to six tetrahedra
    connections

    Arguments:
    cube_node_list -- [list[int]], list of nodes for the cubic element.

    Return:
    tet_node_list -- list[list[int]], list of the six node lists for the
            tetrahedral elements.

    """
    tet_node_list = []
    for node in cube_node_list:
        for idx_list in CONVERSION_IDX:
            tet_list = [node[idx] for idx in idx_list]
            tet_node_list.append(tet_list)

    return tet_node_list


def print_quality(quality_array, metric_name):
    """Prints statistical information about the quality metric

    Arguments:
    quality_array -- np.array, quality data for mesh nodes.
    metric_name -- str, name of the quality metric.

    Return:

    """
    print("{} quality data:".format(metric_name))
    print(
        "Mean: {:.4f} \u00b1 {:.4f}".format(
            np.mean(quality_array),
            np.std(quality_array),
        )
    )
    print(
        "Min-Max: [{:.4f} - {:.4f}]".format(
            np.min(quality_array),
            np.max(quality_array),
        )
    )
    print("10th percentile: {:.4f}".format(np.percentile(quality_array, 10)))
    print("Median: {:.2f}".format(np.median(quality_array)))
    print("90th percentile: {:.4f}".format(np.percentile(quality_array, 90)))


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


def write_exelem_vol(file_path, elements, thickness=True):
    """Writes out the data from a volumetric mesh to a exnode file

    Args:
    file_path -- str, path to the file to save to.
    elements -- ndarray, list of nodes associated with each tetrahedra,
            size = Nx4.
    thickness -- bool, flag used if thickness has been provided to the
            exnode file, default True.

    Returns:

    """
    try:
        assert elements.shape[1] == 4

    except AssertionError:
        sys.stderr.write("Error: elements should contain 4 nodes\n")

    with open(file_path, "w") as f:
        # Write the exnode file header
        f.write("Group name: mesh\n")
        f.write("Region: /uterus\n")
        f.write("Shape.  Dimension=3 simplex(2;3)*simplex*simplex\n")
        f.write("#Scale factor sets=0\n")
        f.write("#Nodes=4\n")

        if thickness:
            f.write("#Fields=2\n")

        else:
            # If no thickness is provided there is only one field
            f.write("#Fields=1\n")

        f.write(
            "1) coordinates, coordinate, rectangular cartesian, #Components=3\n",
        )
        f.write(
            " x. l.simplex(2;3)*l.simplex*l.simplex, no modify, standard node based.\n"
        )
        f.write("  #Nodes=4\n")
        f.write("  1. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  2. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  3. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  4. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write(
            " y. l.simplex(2;3)*l.simplex*l.simplex, no modify, standard node based.\n"
        )
        f.write("  #Nodes=4\n")
        f.write("  1. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  2. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  3. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  4. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write(
            " z. l.simplex(2;3)*l.simplex*l.simplex, no modify, standard node based.\n"
        )
        f.write("  #Nodes=4\n")
        f.write("  1. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  2. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  3. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  4. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")

        if thickness:
            f.write(
                "2) thickness, field, rectangular cartesian, #Components=1\n",
            )
            f.write(" thickness. constant, no modify, standard node based.\n")
            f.write("  #Nodes=1\n")
            f.write("  1. #Values=1\n")
            f.write("	Value indices: 1\n")
            f.write("	Scale factor indices: 0\n")

        for i, nodes in enumerate(elements):
            f.write("Element: {} 0 0\n".format(i + 1))
            f.write(" Nodes: \n")
            f.write(
                "  {} {} {} {}\n".format(
                    nodes[0] + 1, nodes[1] + 1, nodes[2] + 1, nodes[3] + 1
                )
            )


def write_exelem_surf(file_path, elements, thickness=True):
    """Writes out the data from a surface mesh to a exnode file

    Args:
    file_path -- str, path to the file to save to.
    elements -- ndarray, list of nodes associated with each triangle,
            size = Nx3.
    thickness -- bool, flag used if thickness has been provided to the
            exnode file, default True.

    Returns:

    """
    try:
        assert elements.shape[1] == 3

    except AssertionError:
        sys.stderr.write("Error: elements should contain 3 nodes\n")

    with open(file_path, "w") as f:
        # Write the exnode file header
        f.write("Group name: mesh\n")
        f.write("Region: /uterus\n")
        f.write("Shape.  Dimension=2 simplex(2)*simplex\n")
        f.write("#Scale factor sets=0\n")
        f.write("#Nodes=3\n")

        if thickness:
            f.write("#Fields=2\n")

        else:
            # If no thickness is provided there is only one field
            f.write("#Fields=1\n")

        f.write(
            "1) coordinates, coordinate, rectangular cartesian, #Components=3\n",
        )
        f.write(
            " x. l.simplex(2)*l.simplex, no modify, standard node based.\n",
        )
        f.write("  #Nodes=3\n")
        f.write("  1. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  2. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  3. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write(
            " y. l.simplex(2)*l.simplex, no modify, standard node based.\n",
        )
        f.write("  #Nodes=3\n")
        f.write("  1. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  2. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  3. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write(
            " z. l.simplex(2)*l.simplex, no modify, standard node based.\n",
        )
        f.write("  #Nodes=3\n")
        f.write("  1. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  2. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")
        f.write("  3. #Values=1\n")
        f.write("	Value indices: 1\n")
        f.write("	Scale factor indices: 0\n")

        if thickness:
            f.write(
                "2) thickness, field, rectangular cartesian, #Components=1\n",
            )
            f.write(" thickness. constant, no modify, standard node based.\n")
            f.write("  #Nodes=1\n")
            f.write("  1. #Values=1\n")
            f.write("	Value indices: 1\n")
            f.write("	Scale factor indices: 0\n")

        for i, nodes in enumerate(elements):
            f.write("Element: {} 0 0\n".format(i + 1))
            f.write(" Nodes: \n")
            f.write(f"  {nodes[0] + 1} {nodes[1] + 1} {nodes[2] + 1}\n")


def write_exnode(file_path, nodes, thickness=None):
    """Writes out the nodes from a mesh to a exnode file,
    and adds the thickness field if provided

    Args:
    file_path -- str, path to the file to save to.
    nodes -- ndarray, list of coordinates for each node.
            size = Nx3
    thickness -- ndarray, list of thickness value for each node.
            size = Nx1, default value None.

    Returns:

    """
    try:
        # Check for number of coordinates
        assert nodes.shape[1] == 3

    except AssertionError:
        sys.stderr.write("Error: nodes should have three coordinates\n")
        exit()

    if type(thickness) is not type(None):
        try:
            # Check that thickness and nodes have the same dimension
            assert nodes.shape[0] == thickness.shape[0]

        except AssertionError:
            sys.stderr.write(
                "Error: nodes and thickness should have the same number of elements\n"
            )
            exit()

    with open(file_path, "w") as f:
        # Write exnode file header
        f.write("Group name: mesh\n")
        f.write("Region: /uterus\n")

        if type(thickness) is not type(None):
            f.write("#Fields=2\n")

        else:
            # If no thickness is provided there is only one field
            f.write("#Fields=1\n")

        f.write(
            "1) coordinates, coordinate, rectangular cartesian, #Components=3\n",
        )
        f.write(" x. Value index=1, #Derivatives=0\n")
        f.write(" y. Value index=2, #Derivatives=0\n")
        f.write(" z. Value index=3, #Derivatives=0\n")

        if type(thickness) is not type(None):
            f.write(
                "2) thickness, field, rectangular cartesian, #Components=1\n",
            )
            f.write(" thickness. Value index=4, #Derivatives=0\n")

        for i in range(len(nodes)):
            f.write("Node: {}\n".format(i + 1))
            f.write(
                " {} {} {}\n".format(
                    nodes[i][0],
                    nodes[i][1],
                    nodes[i][2],
                )
            )

            if type(thickness) is not type(None):
                f.write(" {}\n".format(thickness[i]))
