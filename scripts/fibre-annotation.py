#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fibre-annotation.py

Annotates a mesh with the fibre information to be viewed in Paraview
Author: Mathias Roesler
Date: 02/25
"""

from symfibre.utils import (
    fibres_from_ortho,
)

import os
import meshio
import argparse

import numpy as np

from symfibre.constants import HOME, BASE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotates a mesh to visualise fibres in Paraview"
    )
    parser.add_argument(
        "dir_path",
        type=str,
        metavar="dir-path",
        help="path from BASE to the data",
    )
    parser.add_argument(
        "mesh_name",
        type=str,
        metavar="mesh-name",
        help="name of the mesh to load without extension",
    )
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        default="vtk",
        help="mesh extension",
    )

    args = parser.parse_args()  # Parse input arguments

    # Create mesh and data path
    mesh_path = os.path.join(HOME, BASE, args.dir_path, args.mesh_name)
    data_path = os.path.join(HOME, BASE, args.dir_path)

    mesh = meshio.read(mesh_path + "." + args.extension)
    ortho_file = os.path.join(mesh_path + ".ortho")

    fibres, angles = fibres_from_ortho(ortho_file)
    mesh.point_data["fibres"] = fibres
    mesh.point_data["angles"] = angles

    if "cell_scalars" in mesh.cell_data.keys():
        # Convert cell_scalars to float to avoid error in Paraview
        float_cell_scalars = mesh.cell_data["cell_scalars"][0].astype(
            np.float64,
        )
        mesh.cell_data["cell_scalars"][0] = float_cell_scalars

    mesh.write(mesh_path + "_annotated." + args.extension)
