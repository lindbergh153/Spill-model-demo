"""
Shoreline Processing Module
===========================

Handle shoreline geometry for detecting oil-coastline interactions
during surface transport simulations.

Functions
---------
get_shore_polygon : Load shoreline from BNA file
rasterize_shoreline : Rasterize polygon to boolean grid for O(1) lookup
is_onshore : Fast point-in-land query via raster lookup

"""

from __future__ import annotations

import csv
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


# Keywords in BNA polygon names that indicate map boundaries (not land)
_BOUNDARY_KEYWORDS = {'map', 'bound', 'spill', 'water', 'extent', 'domain'}


def get_shore_polygon(shore_file):
    """
    Load **land** polygons from a BNA file and merge into a single geometry.

    Filters out map-boundary / spillable-area polygons using two criteria:

    1. ``num_points < 0``  (GNOME negative-count convention)
    2. Polygon name contains boundary-related keywords
       (e.g. "Map Bounds", "SpillableArea")

    Parameters
    ----------
    shore_file : str or None
        Path to BNA shoreline file.

    Returns
    -------
    mergedPoly : shapely Polygon/MultiPolygon or None
    """
    if type(shore_file) is str:
        dataset_x = []
        dataset_y = []
        coord_x = []
        coord_y = []
        skip_current = False
        skipped_names = []

        with open(shore_file, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if len(row) == 3:
                    # ---- header row of a new polygon ----
                    # Save the *previous* polygon (if it was land)
                    if coord_x and not skip_current:
                        dataset_x.append(list(coord_x))
                        dataset_y.append(list(coord_y))

                    # Start fresh coordinate lists
                    coord_x = []
                    coord_y = []

                    # --- Criterion 1: negative point count ---
                    try:
                        num_pts = int(row[2])
                    except ValueError:
                        num_pts = 1
                    skip_by_count = (num_pts < 0)

                    # --- Criterion 2: name contains boundary keywords ---
                    poly_name = row[0].strip().strip('"').lower()
                    skip_by_name = any(kw in poly_name for kw in _BOUNDARY_KEYWORDS)

                    skip_current = skip_by_count or skip_by_name
                    if skip_current:
                        skipped_names.append(row[0].strip().strip('"'))

                elif len(row) == 2 and not skip_current:
                    coord_x.append(row[0])
                    coord_y.append(row[1])

        # Append the last polygon (if land)
        if coord_x and not skip_current:
            dataset_x.append(list(coord_x))
            dataset_y.append(list(coord_y))

        if skipped_names:
            print(f'Skipped {len(skipped_names)} boundary polygon(s): '
                  f'{skipped_names}')

        # Build Shapely polygons
        polygon_array = []
        for xs, ys in zip(dataset_x, dataset_y):
            coords = [(float(lon), float(lat)) for lon, lat in zip(xs, ys)]
            if len(coords) >= 3:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty:
                    polygon_array.append(poly)

        print(f'Number of land polygons: {len(polygon_array)}')

        if not polygon_array:
            print('Warning: No valid land polygons found!')
            return None

        mergedPoly = unary_union(polygon_array)
        return mergedPoly
    else:
        print('Warning: No shoreline data!')
        return None


def rasterize_shoreline(shore_poly, bounds, resolution=0.001):
    """
    Rasterize shoreline polygon(s) to a boolean array for fast stranding checks.

    Parameters
    ----------
    shore_poly : shapely Polygon / MultiPolygon or None
        Merged shoreline geometry returned by ``get_shore_polygon``.
    bounds : tuple of float
        Domain extent as ``(W_bound, E_bound, N_bound, S_bound)`` in degrees.
    resolution : float, optional
        Grid cell size in degrees (default 0.001 ≈ 100 m).

    Returns
    -------
    shore_mask : ndarray of uint8 or None
        2-D boolean raster (1 = land, 0 = water).  Shape ``(height, width)``.
    raster_info : dict or None
        Metadata dict.
    """
    if shore_poly is None:
        return None, None

    W, E, N, S = bounds

    # Grid dimensions
    width = int(np.ceil((E - W) / resolution))
    height = int(np.ceil((N - S) / resolution))

    print(f'Rasterizing shoreline: {width} x {height} pixels, '
          f'resolution={resolution}°')
    print(f'  Bounds: W={W}, E={E}, N={N}, S={S}')

    # Vectorised contains test (shapely >= 1.8)
    from shapely.vectorized import contains

    lon_coords = np.linspace(W + resolution / 2, E - resolution / 2, width)
    lat_coords = np.linspace(N - resolution / 2, S + resolution / 2, height)
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    shore_mask = contains(shore_poly, lon_grid, lat_grid).astype(np.uint8)

    raster_info = {
        'W': W, 'E': E, 'N': N, 'S': S,
        'resolution': resolution,
        'width': width, 'height': height,
    }

    land_pixels = int(np.sum(shore_mask))
    land_ratio = land_pixels / shore_mask.size * 100
    print(f'Rasterization complete. Land pixels: {land_pixels}/{shore_mask.size} '
          f'({land_ratio:.2f}%)')

    # Sanity check: if land coverage is very high, the map boundary is likely
    # still included — warn the user
    if land_ratio > 50:
        print(f'  *** WARNING: land coverage = {land_ratio:.1f}% — '
              f'this is unusually high! ***\n'
              f'  The map-boundary polygon may have been included as land.\n'
              f'  Check the BNA file and verify that get_shore_polygon()\n'
              f'  correctly filters non-land polygons.')

    return shore_mask, raster_info


def is_onshore(lon, lat, shore_mask, raster_info):
    """
    O(1) land/water query via raster look-up.

    Parameters
    ----------
    lon, lat : float
        Geographic coordinates (degrees).
    shore_mask : ndarray of uint8
        Raster produced by ``rasterize_shoreline``.
    raster_info : dict
        Metadata produced by ``rasterize_shoreline``.

    Returns
    -------
    bool
        ``True`` if the point falls on a land pixel.
    """
    if shore_mask is None or raster_info is None:
        return False

    col = int((lon - raster_info['W']) / raster_info['resolution'])
    row = int((raster_info['N'] - lat) / raster_info['resolution'])

    if 0 <= row < shore_mask.shape[0] and 0 <= col < shore_mask.shape[1]:
        return shore_mask[row, col] == 1

    return False


def debug_point(lon, lat, shore_mask, raster_info):
    """
    Print diagnostic information for a single coordinate.
    Call this with the oil release point to verify correctness.

    Usage
    -----
    >>> debug_point(-144.0, 48.5, shore_mask, raster_info)
    """
    if shore_mask is None:
        print('shore_mask is None — rasterization was not performed')
        return

    col = int((lon - raster_info['W']) / raster_info['resolution'])
    row = int((raster_info['N'] - lat) / raster_info['resolution'])
    in_bounds = (0 <= row < shore_mask.shape[0] and
                 0 <= col < shore_mask.shape[1])
    val = shore_mask[row, col] if in_bounds else -1

    print(f'debug_point(lon={lon}, lat={lat})')
    print(f'  Raster col={col}, row={row}  (grid {shore_mask.shape[1]}x{shore_mask.shape[0]})')
    print(f'  In bounds: {in_bounds}')
    print(f'  Pixel value: {val}  (1=land, 0=water)')
    print(f'  is_onshore = {val == 1}')

    # Check a small neighbourhood
    if in_bounds:
        r0 = max(row - 2, 0)
        r1 = min(row + 3, shore_mask.shape[0])
        c0 = max(col - 2, 0)
        c1 = min(col + 3, shore_mask.shape[1])
        patch = shore_mask[r0:r1, c0:c1]
        print(f'  5x5 neighbourhood (1=land):\n{patch}')
