#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fibre-generation.py

Generate idealised fibres for a uterine mesh
Author: Mathias Roesler
Date: 01/25
"""

from symfibre.facet_fct import InnerLayer, OuterLayer, BaseLayer

import os
import ldrb
import argparse

import dolfin as df
import pandas as pd

from symfibre.constants import HOME, BASE
from symfibre.constants import (
    INNER_MARK,
    OUTER_MARK,
    BASE_MARK,
    MARKERS,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates idealised fibres for a uterine mesh",
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
        default="xml",
        help="mesh extension",
    )

    args = parser.parse_args()

    # Create mesh path
    mesh_path = os.path.join(
        HOME, BASE, args.dir_path, args.mesh_name + "." + args.extension
    )

    # Create path to outer data (currently hacky)
    data_path = os.path.join(
        HOME, BASE, "symfibre/data", args.mesh_name + "_outer_points.csv"
    )

    outer_data = pd.read_csv(data_path)

    # Extract the last three columns (assuming they are x, y, z coordinates)
    outer_coords = outer_data.iloc[:, -3:].to_numpy()

    mesh = df.Mesh(mesh_path)

    # Create facet function
    ffun = df.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    # Create instances to annotate facets (order is important)
    inner = InnerLayer()
    outer = OuterLayer(outer_coords)
    base = BaseLayer()

    inner.mark(ffun, INNER_MARK)
    outer.mark(ffun, OUTER_MARK)
    base.mark(ffun, BASE_MARK)

    df.File("../data/" + args.mesh_name + "_markers" + ".pvd") << ffun

    # Decide on the angles you want to use
    angles = dict(
        alpha_endo_lv=-45,  # Fiber angle on the LV endocardium
        alpha_epi_lv=-90,  # Fiber angle on the LV epicardium
        beta_endo_lv=0,  # Sheet angle on the LV endocardium
        beta_epi_lv=0,  # Sheet angle on the LV epicardium
    )

    fibre, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh,
        fiber_space="Quadrature_2",
        markers=MARKERS,
        ffun=ffun,
        **angles,
    )
    ldrb.fiber_to_xdmf(fibre, "../data/fibre")
