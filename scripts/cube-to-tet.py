#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cube-to-tet.py

Script to convert a cubic mesh to a tetrahedral one
Author: Mathias Roesler
Date: 03/25
"""

import os
import sys
import meshio
import argparse

from symmesh.utils import convert_connections
from symmesh.constants import BASE, HOME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a cubic vtk mesh to a tetrahedral one"
    )
    # Parse input arguments
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
        type=str,
        help="mesh extension, vtk or vtu",
        default="vtk",
    )

    args = parser.parse_args()

    # Create file path
    dir_path = os.path.join(
        HOME,
        BASE,
        args.dir_path,
    )
    mesh_path = os.path.join(dir_path, args.mesh_name)

    # Check extension
    try:
        assert args.extension == "vtk" or args.extension == "vtu"

    except AssertionError:
        sys.stderr.write("Error: incorrect extension\n")
        sys.stderr.write(f"Got {args.extension} instead of vtk or vtu")
        exit()

    hex_mesh_path = os.path.join(mesh_path + "." + args.extension)
    tet_mesh_path = os.path.join(mesh_path + "_tet." + args.extension)

    # Read the whole vtk file
    hex_mesh = meshio.read(hex_mesh_path)

    points = hex_mesh.points
    nodes = hex_mesh.cells_dict["hexahedron"]
    cells = [("tetra", convert_connections(nodes))]

    # Create tet mesh and save
    tet_mesh = meshio.Mesh(points, cells)
    tet_mesh.write(tet_mesh_path)
