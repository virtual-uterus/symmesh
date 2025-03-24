#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paraview_fct.py

Functions based off Paraview's API
Author: Mathias Roesler
Date: 11/24
"""

import os

import numpy as np
import paraview.simple as ps
import paraview.servermanager as psm


def fetch_quality_data(quality, mesh_quality, view):
    """Fetches the quality data based on the selected quality

    Args:
    quality -- str, quality metric to fecth.
    mesh_quality -- psm.MeshQuality, Paraview extracted mesh quality object.
    view -- psm.RenderView, Paraview rendered view of the mesh.

    Returns:
    quality_data -- np.array, quality value for the cells in the mesh.

    Raises:
    ValueError -- if the quality array is not found.

    """
    # Properties modified on mesh_quality
    mesh_quality.TetQualityMeasure = quality

    # Update the view to ensure updated data information
    view.Update()

    fetched_data = psm.Fetch(mesh_quality)
    cell_data = fetched_data.GetCellData()
    quality_array = cell_data.GetArray("Quality")

    if quality_array:
        quality_data = [
            quality_array.GetValue(i)
            for i in range(
                quality_array.GetNumberOfTuples(),
            )
        ]

    else:
        raise ValueError("quality array not found")

    return np.array(quality_data)


def paraview_quality(mesh_path, metric):
    """Inspects the quality of the mesh for a given metric

    Args:
    mesh_path -- str, path to the mesh vtu file.
    metric -- str, quality metric used {Aspect Ratio, Mean Ratio, Jacobian}.

    Returns:
    quality_data -- np.array, quality data for the given metric.

    Raises:
    ValueError -- if the extension is not vtk or vtu.
    FileNotFoundError -- if the file is not found.
    RuntimeError -- if the an error occurs while opening the file.
    ValueError -- if the quality array is not found.

    """
    extension = os.path.splitext(mesh_path)[1]

    try:
        if extension == ".vtk":
            mesh = ps.LegacyVTKReader(
                registrationName="mesh.vtk",
                FileNames=[mesh_path],
            )

        elif extension == ".vtu":
            # Create a new 'XML Unstructured Grid Reader'
            mesh = ps.XMLUnstructuredGridReader(
                registrationName="mesh.vtu",
                FileName=[mesh_path],
            )

        else:
            raise ValueError(
                "unrecognised extension {}\n".format(extension),
            )
    except FileNotFoundError as e:
        raise e
    except ValueError as e:
        raise e
    except Exception as e:
        raise e

    # Get active view
    view = ps.GetActiveViewOrCreate("RenderView")

    # Update the view to ensure updated data information
    view.Update()

    # Create a new 'Mesh Quality'
    mesh_quality = ps.MeshQuality(
        registrationName="quality",
        Input=mesh,
    )
    # Update the view to ensure updated data information
    view.Update()

    try:
        return fetch_quality_data(metric, mesh_quality, view)
    except ValueError as e:
        raise e
