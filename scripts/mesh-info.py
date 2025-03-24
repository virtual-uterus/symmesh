#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh-info.py

Script that presents information about meshes
Author: Mathias Roesler
Date: 11/24
"""

import os
import sys
import argparse


from symmesh import mesh
from symmesh.constants import HOME, BASE


def add_shared_arguments(parser):
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
        "--range",
        "-r",
        nargs="+",
        help="range of mesh numbers for comparison",
    )
    parser.add_argument(
        "--extension",
        type=str,
        help="mesh extension",
        default="vtk",
    )


if __name__ == "__main__":
    # Main argument parser setup
    parser = argparse.ArgumentParser(
        description="Extracts information from meshes",
    )
    subparsers = parser.add_subparsers(
        title="subcommands", description="Available commands", dest="command"
    )

    # Subcommand: quality
    quality_parser = subparsers.add_parser(
        "quality",
        help="Extracts the quality information based on the parameter",
    )
    add_shared_arguments(quality_parser)
    quality_parser.add_argument(
        "metric",
        type=str,
        choices={"ar", "mr", "ja", "sj"},
        help="metrics (aspect ratio, mean ratio, Jacobian, scaled Jacobian)",
    )
    quality_parser.set_defaults(func=mesh.quality_information)

    # Subcommand: resolution
    distance_parser = subparsers.add_parser(
        "distance",
        help="Extracts the average distance between elements a mesh",
    )
    add_shared_arguments(distance_parser)
    distance_parser.set_defaults(func=mesh.distance_information)

    # Parse the arguments and call the appropriate function
    args = parser.parse_args()

    # Create file path
    file_path = os.path.join(
        HOME,
        BASE,
        args.dir_path,
        args.mesh_name,
    )

    try:
        if args.command == "quality":
            args.func(
                file_path,
                args.mesh_name,
                args.metric,
                args.range,
                args.extension,
            )
        elif args.command == "distance":
            args.func(
                file_path,
                args.mesh_name,
                args.range,
                args.extension,
            )
        else:
            parser.print_help()
    except Exception as e:
        sys.stderr.write("Error: {}\n".format(e))
        exit()
