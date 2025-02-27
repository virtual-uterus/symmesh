#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
point-generation.py

Generate the list of points and element coordinates
to interrogate the structure tensor at to create realistic fibres.
Author: Mathias Roesler
Date: 02/25
"""

from symfibre.utils import (
    element_centres,
)

import os
import sys
import argparse

import dolfin as df
import pandas as pd

from symfibre.constants import HOME, BASE

desc = "Writes .csv files for points and elements for fibre generation"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
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
        default="xml",
        help="mesh extension",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        default=1,
        help="scaling factor used to scale the mesh coordinates",
    )
    parser.add_argument(
        "--translation",
        "-t",
        type=int,
        nargs="+",
        help="translation coefficients for the mesh coordinates",
    )

    args = parser.parse_args()  # Parse input arguments

    if args.translation is None:
        # Default to 0 if not translation is given
        args.translation = [0, 0, 0]

    try:
        assert len(args.translation) == 3
    except AssertionError:
        sys.stderr.write("Error: translation should have three coordinates\n")
        exit()

    # Create mesh and data path
    mesh_path = os.path.join(HOME, BASE, args.dir_path, args.mesh_name)
    data_path = os.path.join(HOME, BASE, args.dir_path)

    mesh = df.Mesh(mesh_path + "." + args.extension)

    # Save element centres
    centres = element_centres(mesh)

    rescaled_centres = (centres + args.translation) * args.scale
    rescaled_points = (mesh.coordinates + args.translation) * args.scale
    pd.DataFrame(rescaled_centres).to_csv(  # Save to csv file
        os.path.join(data_path, args.mesh_name + "_elements.csv"),
        index=False,
        header=False,
    )
    pd.DataFrame(rescaled_points).to_csv(  # Save to csv file
        os.path.join(data_path, args.mesh_name + "_points.csv"),
        index=False,
        header=False,
    )
