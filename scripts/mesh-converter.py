#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh-converter.py

Script that converts a vtu mesh to cmgui format
Author: Mathias Roesler
Date: 06/23
"""

import argparse
import os
import sys

import meshio

import symmesh.utils as utils
from symmesh.constants import BASE, HOME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts an annotated vtu or vtk mesh to a cmgui format"
    )

    # Parse input arguments
    parser.add_argument(
        "mesh_name",
        type=str,
        metavar="mesh-name",
        help="name of the mesh to convert",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="mesh/",
        help="path from BASE to the mesh, default mesh/",
    )
    parser.add_argument(
        "-e",
        "--extension",
        choices={"vtu", "vtk"},
        help="mesh extesion, default value vtk",
        default="vtk",
    )

    args = parser.parse_args()

    # Set arguments
    mesh_path = os.path.join(HOME, BASE, args.mesh_dir)
    mesh_file = mesh_path + "/" + args.mesh_name

    # Read the mesh file
    print("Loading mesh {}".format(mesh_file + "." + args.extension))
    mesh = meshio.read(mesh_file + "." + args.extension)

    # Extract information
    nodes = mesh.points

    try:
        thickness = mesh.point_data["thickness"]
        thickness_flag = True

    except KeyError:
        sys.stderr.write("Warning: no thickness data found\n")
        thickness = None
        thickness_flag = False

    if thickness is not None and len(thickness.shape) == 2:
        # Using a vtu format that needs to be reshaped
        thickness = thickness.reshape(thickness.shape[0])

    try:
        elements = mesh.cells_dict["tetra"]
        vol_flag = True

    except KeyError:
        elements = mesh.cells_dict["triangle"]
        vol_flag = False

    # Write EX files
    try:
        print("Writing exnode file")
        utils.write_exnode(mesh_file + ".exnode", nodes, thickness)

        print("Writing exelem file")
        if vol_flag:
            utils.write_exelem_vol(mesh_file + ".exelem",
                                   elements, thickness_flag)

        else:
            utils.write_exelem_surf(
                mesh_file + ".exelem", elements, thickness_flag)
    except ValueError as e:
        sys.stderr.write("Error: {}\n".format(e))
        exit()
