#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh-annotation.py

Script that adds the thickness data to a vtu/vtk mesh
Author: Mathias Roesler
Date: 06/23
"""

import argparse
import os

import meshio
import numpy as np
import scipy.io

import symmesh.utils as utils
from symmesh.constants import BASE, HOME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotates a vtu or vtk mesh with the thickness values "
        "of each slice"
    )

    # Parse input arguments
    parser.add_argument(
        "base_name", type=str, metavar="base-name", help="name of the dataset"
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="mesh/",
        help="path from BASE to the mesh, default mesh/",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="uterine-microCT/data",
        help="path from BASE to thickness data, default uterine-microCT/data",
    )
    parser.add_argument(
        "--horn",
        type=str,
        choices={"left", "right", "both"},
        help="horn to process",
        default="both",
    )
    parser.add_argument(
        "--normal-slices",
        type=int,
        help="number of slices in bottom and top section with normal planes",
        default=20,
    )
    parser.add_argument(
        "--plane-distance",
        type=int,
        help="distance between the origin and the cut planes",
        default=10,
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        help="factor used for upsampling",
        default=4,
    )
    parser.add_argument(
        "-s",
        "--switch",
        action="store_true",
        help="switches the labels of the left and right horn, default False",
    )
    parser.add_argument(
        "--not-d",
        action="store_true",
        help="flag used if the dataset is not downsampled, default False",
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
    mesh_directory = os.path.join(HOME, BASE, args.mesh_dir)
    thickness_directory = os.path.join(
        HOME, BASE, args.data_dir, args.base_name)
    mesh_name = os.path.join(
        mesh_directory,
        args.base_name + "_volumetric_mesh",
    )

    if not args.not_d:
        # If the dataset is downsampled
        thickness_directory = os.path.join(thickness_directory, "downsampled")
        param_file = os.path.join(
            thickness_directory, args.base_name + "_downsampled.toml"
        )

    else:
        param_file = os.path.join(
            thickness_directory,
            args.base_name + ".toml",
        )

    # Load parameters
    params = utils.parse_TOML(param_file)

    # Add the muscle segmentation to the load directory
    thickness_directory = os.path.join(
        thickness_directory,
        "muscle_segmentation",
    )

    # Convert both to left and right
    if args.horn == "both":
        horns = ["left", "right"]

    else:
        horns = [args.horn]

    # Get the centreline
    centreline_dict = scipy.io.loadmat(thickness_directory + "/centreline.mat")
    centreline = np.transpose(centreline_dict["centreline"])

    # Read the mesh file
    print("Loading mesh {}".format(mesh_name + "." + args.extension))
    mesh = meshio.read(mesh_name + "." + args.extension)
    nb_points = len(mesh.points)

    # Swap because of image coordinates
    mesh.points[:, 0] = -mesh.points[:, 0]
    mesh.points[:, 1] = -mesh.points[:, 1]

    # Read thickness data
    thickness_data = np.load(
        thickness_directory + "/muscle_thickness.pkl", allow_pickle=True
    )

    # Thickness data dictionary to be added to the mesh
    point_data_dict = dict()
    point_data_array = np.zeros((nb_points, 1))
    point_data_name = "thickness"

    plane_distance = args.plane_distance

    for i, horn in enumerate(horns):
        print("Annotating {} horn".format(horn))
        if args.switch:
            # If the centrepoints need to be switched
            i = i - 1

        thickness = thickness_data[horn]

        if horns[i] == "left":
            centrepoints = centreline[0: len(thickness), 0:2]
        else:
            centrepoints = centreline[0: len(thickness), 4:6]

        centrepoints_diff = np.diff(centrepoints, axis=0)
        z_components = np.ones((len(centrepoints_diff), 1))

        # Create normalised centre vectors
        centre_vectors = np.append(centrepoints_diff, z_components, axis=1)
        centre_norms = np.repeat(
            np.linalg.norm(centre_vectors, axis=1), 3
        )  # Repeat to be able to reshape
        centre_norms = np.reshape(
            centre_norms, centre_vectors.shape
        )  # Reshape for division
        centre_vectors_norm = centre_vectors / centre_norms

        max_slices = len(thickness) - args.normal_slices

        for j in range(len(thickness)):
            if centreline[j, 0] and centreline[j, 4] > 0:
                # Only get x_split if there are two horns otherwise
                # use previous value
                x_split = (centreline[j, 0] + centreline[j, 4]) / 2

            if j >= len(centre_vectors_norm):
                centre_vector = np.array([0, 0, 1])
                centre_norm = 1
            else:
                centre_vector = centre_vectors_norm[j]
                centre_norm = centre_norms[j][0]

            # Get the x limited indices
            if horns[i] == "left":
                x_idx = np.where(mesh.points[:, 0] < x_split)[0]
            else:
                x_idx = np.where(mesh.points[:, 0] >= x_split)[0]

            elements = mesh.points[x_idx] - np.append(
                centrepoints[j], j
            )  # Reduced set of points and recentre

            idx_list = utils.get_indices(
                centre_vector,
                np.transpose(elements),
                plane_distance,
                centre_norm,
            )

            if j <= args.normal_slices or j >= max_slices:
                extra_idx = utils.get_indices(
                    np.array([0, 0, 1]),
                    np.transpose(elements),
                    plane_distance,
                    1,
                )
                idx_list = np.append(idx_list, extra_idx)
            point_data_array[x_idx[idx_list]] = round(thickness[j], 3)

    # Add the data dictionary to the mesh
    point_data_dict[point_data_name] = point_data_array
    mesh.point_data = point_data_dict

    # Flip mesh back to original position
    mesh.points[:, 0] = -mesh.points[:, 0]
    mesh.points[:, 1] = -mesh.points[:, 1]

    # Scale and translate to fit the fibre analysis
    mesh.points[:, 0] += params["ylim"][0] - 1
    mesh.points[:, 1] += params["xlim"][0] - 2
    mesh.points *= args.upsample_factor

    # Save new mesh
    print("Saving mesh {}".format(mesh_name + "_annotated." + args.extension))
    mesh.write(mesh_name + "_annotated." + args.extension)
