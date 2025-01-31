#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
facet_fct.py

Function for annotating uterine mesh facets.
Author: Mathias Roesler
Date: 01/25
"""

import dolfin as df


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
        return coord[2] <= 0.5 and coord[2] >= 0.0 and on_boundary


class OuterLayer(df.SubDomain):
    def inside(self, coord, on_boundary):
        """Identifies outer layer elements

        Arguments:
        coord -- np.array, 3D array containing coordinates of the element.
        on_boundary -- bool, flag indicating wether the point is on the
        boundary

        Return:
        on_boundary -- bool, true if conditions are satisfied

        """
        return coord[2] >= 0.5 and on_boundary


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
        return coord[2] <= 0.0 and on_boundary
