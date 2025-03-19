#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py

Plot functions used in the symmesh package
Author: Mathias Roesler
Date: 03/25
"""

import matplotlib.pyplot as plt
import numpy as np

from .constants import LEFT, RIGHT, BOTTOM


def plot_single_mesh_quality(quality_data, metric, mesh_name):
    """Plots the quality data for a mesh as a boxplot

    Arguments:
    quality_data -- np.array, quality data.
    metric -- str, name of the quality metric.
    mesh_name -- str, name of the mesh for title.

    Return:

    Raises:

    """
    # Create figure and plot
    fig, ax = plt.subplots(dpi=300)

    ax.hist(quality_data, 200, color="k")

    if metric == "Scaled Jacobian" or metric == "Mean Ratio":
        plt.xlim([0, 1])

    else:
        # Display other metrics on log scales
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(f"{metric}")
    plt.ylabel("Number of elements")

    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM)
    plt.show()


def plot_multi_mesh_quality(quality_dict, density_data, metric):
    """Plots the quality data for multiple meshes as single points with
    error bars

    Arguments:
    quality_dict -- dict, dictionnary with the quality data as value
    and mesh number as key.
    density_data -- np.array, number of elements in each mesh.
    metric -- str, name of the quality metric.

    Return:

    Raises:

    """
    # Create figure and plot
    fig, ax = plt.subplots(dpi=300)
    data = np.zeros(len(quality_dict.keys()))  # Empty list for data
    yerr = np.zeros(len(quality_dict.keys()))  # Empty list for error
    cpt = 0  # Loop counter

    for sim_nb, quality_data in quality_dict.items():
        data[cpt] = np.mean(quality_data)
        yerr[cpt] = np.std(quality_data)

        cpt += 1

    ax.errorbar(
        density_data,
        data,
        yerr=yerr,
        fmt=".-",
        color="black",
        capsize=5,
        label="Mean ± Std",
    )

    plt.xlabel("Number of elements")
    plt.ylabel("{}".format(metric))

    if metric == "Scaled Jacobian" or metric == "Mean Ratio":
        plt.ylim([0, 1])

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM)
    plt.show()
