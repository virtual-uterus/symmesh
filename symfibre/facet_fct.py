#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facet_fct.py

Function for annotating uterine mesh facets.
Author: Mathias Roesler
Date: 01/25
"""

import dolfin as df
import numpy as np


class InnerLayer(df.SubDomain):
    def inside(self, coord, on_boundary):
        """Identifies inner layer elements

        Arguments:
        coord -- np.array, 3D array containing coordinates of the element.
        on_boundary -- bool, flag indicating wether the point is on the
        boundary

        Return:
        on_boundary -- bool, true if conditions are satisfied

        """
        return on_boundary
        # return coord[1] <= 0.039 and coord[1] >= -0.039 and on_boundary


class OuterLayer(df.SubDomain):
    def __init__(self, outer_coords):
        super().__init__()
        self.outer_coords = outer_coords

    def inside(self, coord, on_boundary):
        """Identifies outer layer elements

        Arguments:
        coord -- np.array, 3D array containing coordinates of the element.
        on_boundary -- bool, flag indicating wether the point is on the
        boundary

        Return:
        on_boundary -- bool, true if conditions are satisfied

        """
        if on_boundary:
            norms = np.linalg.norm(coord - self.outer_coords, axis=1)
            return min(norms) < 0.025
        return False


class BaseLayer(df.SubDomain):
    def inside(self, coord, on_boundary):
        """Identifies base layer elements

        Arguments:
        coord -- np.array, 3D array containing coordinates of the element.
        on_boundary -- bool, flag indicating wether the point is on the
        boundary

        Return:
        on_boundary -- bool, true if conditions are satisfied

        """
        if coord[2] >= 0.98 and on_boundary:
            return True
        return False
