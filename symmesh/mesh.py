#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh.py

Functions for extracting mesh information
Author: Mathias Roesler
Date: 11/24
"""

import meshio
import itertools
import numpy as np
import pyvista as pv

import symmesh.paraview_fct as pf

from symmesh import utils, plots
from symmesh.constants import QUALITY_METRIC_MAP, RES_DICT


def neighbour_distance(mesh_path):
    """Finds the Euclidean distance between neighbouring elements in
    a volumetric mesh

    Arguments:
    mesh_path -- str, path to the mesh to load.

    Return:
    distances -- np.array(float), array of mean distance between each cell and
    its neighbours.

    Raises:
    FileNotFoundError -- if the mesh is not found.

    """
    try:
        mesh = pv.read(mesh_path)

    except FileNotFoundError as e:
        raise e

    # Calculate centroids of all cells
    centroids = mesh.cell_centers().points

    # Dictionary to store distances between neighbouring cells
    neighbour_distances = np.zeros(mesh.n_cells)

    # Iterate through all cells in the mesh
    for i in range(mesh.n_cells):
        # Get indices of neighbouring cells using faces
        neighbours = mesh.cell_neighbors(i, "faces")

        # Calculate distances to neighbours
        distances = []
        for neighbour in neighbours:
            dist = np.linalg.norm(centroids[i] - centroids[neighbour])
            distances.append(dist)

        # Store distances
        neighbour_distances[i] = np.mean(distances)

    return np.array(neighbour_distances)


def edge_lengths(mesh_path):
    """Finds the average edge length of the elements in a volumetric mesh

    Arguments:
    mesh_path -- str, path to the mesh to load.

    Return:
    edge_len -- np.array(float), array with the edge length of the elements.

    Raises:
    FileNotFoundError -- if the mesh is not found.

    """
    edge_len = []
    try:
        mesh = meshio.read(mesh_path)

        elements = mesh.cells_dict["tetra"]  # Assume tetrahedral mesh
        points = mesh.points

        for ele in elements:
            coordinates = points[ele]
            edges = list(itertools.combinations(coordinates, 2))

            for edge in edges:
                edge_len.append(np.linalg.norm(edge[0] - edge[1]))

    except FileNotFoundError as e:
        raise e

    return edge_len


def distance_information(file_path, mesh_name, sim_range, extension):
    """Extracts the average distance between elements and the average
    edge length in a mesh and prints it

    Arguments:
    file_path -- str, path to the mesh.
    mesh_name -- str, name of the mesh used for printing.
    sim_range -- list[int], range for several mesh with the same base name.
    extension -- str, mesh extension.

    Returns:

    Raises:
    FileNotFoundError -- if the mesh is not found.

    """
    if sim_range is not None:
        sim_numbers = utils.get_range(sim_range)

        for sim_nb in sim_numbers:
            mesh_path = "{}_{}.{}".format(
                file_path,
                sim_nb,
                extension,
            )

            try:
                distance = neighbour_distance(mesh_path)
                edge_len = edge_lengths(mesh_path)
            except FileNotFoundError as e:
                raise e

            print(
                "{}_{}: mean distance {:.4f} \u00b1 {:.4f}".format(
                    mesh_name,
                    sim_nb,
                    np.mean(distance),
                    np.std(distance),
                )
            )
            print(
                "{}_{}: mean edge length {:.4f} \u00b1 {:.4f}".format(
                    mesh_name,
                    sim_nb,
                    np.mean(edge_len),
                    np.std(edge_len),
                )
            )

    else:
        mesh_path = "{}.{}".format(file_path, extension)
        try:
            distance = neighbour_distance(mesh_path)
            edge_len = edge_lengths(mesh_path)
        except FileNotFoundError as e:
            raise e

        print(
            "{}: mean distance {:.4f} \u00b1 {:.4f}".format(
                mesh_name,
                np.mean(distance),
                np.std(distance),
            )
        )
        print(
            "{}: mean edge length {:.4f} \u00b1 {:.4f}".format(
                mesh_name,
                np.mean(edge_len),
                np.std(edge_len),
            )
        )


def quality_information(file_path, mesh_name, metric, sim_range, extension):
    """Extracts and plots the quality information of a mesh based on the metric

    Arguments:
    file_path -- str, path to the mesh.
    mesh_name -- str, name of the mesh used for plotting labels.
    metric -- str, quality metric used {Aspect ratio, Jacobian}.
    sim_range -- list[int], range for several mesh with the same base name.
    extension -- str, mesh extension.

    Returns:

    Raises:
    ValueError -- if the extension is not vtk or vtu.
    FileNotFoundError -- if the file is not found.
    RuntimeError -- if the an error occurs while opening the file.
    ValueError -- if the quality array is not found.

    """
    if sim_range is not None:
        sim_numbers = utils.get_range(sim_range)

        quality_dict = {}
        density_data = np.zeros(len(sim_numbers))

        for i, sim_nb in enumerate(sim_numbers):
            mesh_path = "{}_{}.{}".format(
                file_path,
                sim_nb,
                extension,
            )

            try:
                quality_dict[sim_nb] = pf.paraview_quality(
                    mesh_path, QUALITY_METRIC_MAP[metric]
                )
            except Exception as e:
                raise e

            density_data[i] = RES_DICT[f"{mesh_name}_{sim_nb}"]

        plots.plot_multi_mesh_quality(
            quality_dict, density_data, QUALITY_METRIC_MAP[metric]
        )

    else:
        # Plot for a single mesh
        mesh_path = "{}.{}".format(file_path, extension)
        try:
            quality_data = pf.paraview_quality(
                mesh_path,
                QUALITY_METRIC_MAP[metric],
            )

        except Exception as e:
            raise e
        utils.print_quality(quality_data, metric)
        plots.plot_single_mesh_quality(
            quality_data, QUALITY_METRIC_MAP[metric], mesh_name
        )
