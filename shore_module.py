"""
Shoreline Processing Module
===========================

Handle shoreline geometry for detecting oil-coastline interactions
during surface transport simulations.

Functions
---------
get_shore_polygon : Load shoreline from BNA file
inside_polygon : Check if point is inside shoreline

"""

from __future__ import annotations

import csv
from shapely.geometry import Polygon
from shapely.ops import unary_union

def get_shore_polygon(shore_file):
    dataset_x = []
    dataset_y = []
    len_row = []

    with open(shore_file, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if len(row) == 3:
                len_row.append(len(row))
                coord_x = []
                coord_y = []

                if len(len_row) < 2:
                    pass
                elif len_row[-1] != len_row[-2]:
                    dataset_x.append(coord_x)
                    dataset_y.append(coord_y)
                    coord_x.clear()
                    coord_y.clear()

            elif len(row) == 2:
                len_row.append(len(row))
                coord_x.append(row[0])
                coord_y.append(row[1])

    polygon_array = []

    for i, j in zip(dataset_x, dataset_y):
        transit = []
        for lon, lat in zip(i, j):
            lon, lat = float(lon), float(lat)
            transit.append((lon, lat))
        polygon_array.append(Polygon(transit))
        transit.clear()
    mergedPoly = unary_union(polygon_array)

    # return polygon_array
    return mergedPoly

