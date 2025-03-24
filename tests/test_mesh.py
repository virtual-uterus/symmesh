#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mesh.py

Unit tests for the mesh functions in mesh.py.
Author: Mathias Roesler
Date: 03/25

This file contains test cases for the functions:
- neighbour_distance
- edge_lengths
- quality_information
- distance_information

The tests cover various scenarios including valid inputs, invalid inputs,
and edge cases.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from symmesh.mesh import (
    neighbour_distance,
    edge_lengths,
    quality_information,
    distance_information,
)


#  Mock mesh fixture
@pytest.fixture
def mock_pyvista_mesh():
    """Mock a PyVista mesh object with cell centroids and neighbors."""
    mock_mesh = MagicMock()
    mock_mesh.n_cells = 5
    mock_mesh.cell_centers.return_value.points = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    )
    mock_mesh.cell_neighbors.side_effect = (
        lambda i, _: [i - 1, i + 1] if 0 < i < 4 else [i +
                                                       1] if i == 0 else [i - 1]
    )
    return mock_mesh


#  Mock meshio fixture
@pytest.fixture
def mock_meshio_mesh():
    """Mock a MeshIO mesh object with tetrahedral elements."""
    mock_mesh = MagicMock()
    mock_mesh.cells_dict = {"tetra": np.array([[0, 1, 2, 3], [1, 2, 3, 4]])}
    mock_mesh.points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    return mock_mesh


#  Test `neighbour_distance`
@patch("pyvista.read")
def test_neighbour_distance(mock_read, mock_pyvista_mesh):
    mock_read.return_value = mock_pyvista_mesh
    distances = neighbour_distance("dummy.vtk")

    assert isinstance(distances, np.ndarray)
    assert distances.shape == (5,)
    assert np.all(distances >= 0)  # Distances should be non-negative


# Test `edge_lengths`
@patch("meshio.read")
def test_edge_lengths(mock_read, mock_meshio_mesh):
    mock_read.return_value = mock_meshio_mesh
    edge_lengths_result = edge_lengths("dummy.vtu")

    assert isinstance(edge_lengths_result, list)
    assert all(isinstance(edge, float) for edge in edge_lengths_result)
    # Lengths must be positive
    assert all(edge > 0 for edge in edge_lengths_result)


#  Test `distance_information`
@patch("symmesh.mesh.neighbour_distance")
@patch("symmesh.mesh.edge_lengths")
def test_distance_information(
    mock_edge_lengths,
    mock_neighbour_distance,
    capsys,
):
    mock_neighbour_distance.return_value = np.array([1.0, 2.0, 3.0])
    mock_edge_lengths.return_value = [1.5, 2.5, 3.5]

    distance_information("path/to/mesh", "test_mesh", None, "vtu")

    captured = capsys.readouterr()
    assert "mean distance" in captured.out
    assert "mean edge length" in captured.out


# Test `quality_information`
@pytest.mark.parametrize("metric", ["ar", "sj", "mr", "ja"])
@patch("symmesh.paraview_fct.paraview_quality")
@patch("symmesh.utils.print_quality")
@patch("symmesh.plots.plot_single_mesh_quality")
def test_quality_information(
    mock_plot,
    mock_print,
    mock_paraview_quality,
    metric,
):
    mock_paraview_quality.return_value = np.array([0.8, 0.9, 1.0])

    quality_information("path/to/mesh", "test_mesh", metric, None, "vtu")

    mock_paraview_quality.assert_called_once()
    mock_print.assert_called_once()
    mock_plot.assert_called_once()
