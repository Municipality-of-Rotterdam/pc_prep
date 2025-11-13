import argparse
import math
import re

import geopandas as gpd
from shapely.geometry import box, Polygon

TILE_SIZE = 50  # meters, Cyclomedia NL tiles (EPSG:28992)


def parse_ix_iy(filename: str) -> tuple[int, int]:
    """
    Parse tile indices (ix, iy) from a Cyclomedia-style tile filename.

    Expected pattern:
        "..._<ix>_<iy>..."
    where <ix> and <iy> are 4-digit integers, e.g. "filtered_1647_8639.laz".

    The first occurrence of "<4 digits>_<4 digits>" is used.

    Parameters
    ----------
    filename : str
        The filename or path containing the tile indices.

    Returns
    -------
    tuple[int, int]
        (ix, iy) tile indices as integers.

    Raises
    ------
    ValueError
        If `filename` is not a string or does not contain a valid index pattern.
    """
    if not isinstance(filename, str):
        raise ValueError("FileName is not a string.")

    match = re.match(r".*(\d{4}_\d{4}).*", filename)
    if not match:
        raise ValueError(f"Could not parse tile indices from FileName '{filename}'.")

    ix, iy = list(map(int, match.group(1).split("_")))[0:2]
    return ix, iy


def ix_iy_from_lower_left(geom: Polygon) -> tuple[int, int]:
    """
    Compute tile indices from a geometry's lower-left corner.

    The tile index grid is assumed to be a 50 m lattice (RD New / EPSG:28992).
    The tile indices are simply floor(minx / TILE_SIZE), floor(miny / TILE_SIZE).

    Parameters
    ----------
    geom : shapely.geometry.Polygon
        Polygon geometry representing a tile footprint.

    Returns
    -------
    tuple[int, int]
        (ix, iy) tile indices inferred from the polygon's lower-left corner.
    """
    minx, miny, _, _ = geom.bounds
    ix = math.floor(minx / TILE_SIZE)
    iy = math.floor(miny / TILE_SIZE)
    return ix, iy


def snap_bbox_to_index_ranges(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box into inclusive index ranges for tile lower-left corners.

    Given world coordinates (in meters, EPSG:28992), compute:
    - ix_min, iy_min : lower bounds, floored
    - ix_max, iy_max : upper bounds, ceiled, minus 1 so that we stay inside

    This defines the full set of tile indices that *should* exist to cover
    the bounding box.

    Parameters
    ----------
    minx, miny, maxx, maxy : float
        Bounding box coordinates in tile CRS (meters, EPSG:28992).

    Returns
    -------
    tuple[int, int, int, int]
        (
            ix_min,
            iy_min,
            ix_max,
            iy_max,
        )
        inclusive index ranges for ix and iy.
    """
    ix_min = int(math.floor(minx / TILE_SIZE))
    iy_min = int(math.floor(miny / TILE_SIZE))
    ix_max = int(math.ceil(maxx / TILE_SIZE) - 1)
    iy_max = int(math.ceil(maxy / TILE_SIZE) - 1)
    return ix_min, iy_min, ix_max, iy_max


def main(in_path: str, out_path: str) -> None:
    """
    Generate a fully filled Cyclomedia-style tile grid and write it to a GPKG.

    What this script does:
    1. Read an existing (possibly sparse) Cyclomedia-like tile grid from `in_path`.
    2. Infer tile indices and alignment from the first feature, and validate that
       the filename-based indices match the geometry-based indices.
    3. Compute the overall bounding box of all features and snap it to the 50 m
       Cyclomedia RD New lattice.
    4. Create a *complete* grid of 50 m × 50 m tiles that fills this bounding box.
    5. Name each new tile using the Cyclomedia naming scheme:
           filtered_{ix}_{iy}.laz
    6. Write the new grid to `out_path` as a GeoPackage with columns:
       - "FileName"
       - "geometry"

    Parameters
    ----------
    in_path : str
        Input .gpkg path containing at least:
        - geometry (tile polygons)
        - "FileName" column with names like "filtered_<ix>_<iy>.laz".
    out_path : str
        Output .gpkg path for the filled grid.

    Raises
    ------
    SystemExit
        If the input is empty, missing CRS, not in EPSG:28992 (or convertible),
        or missing the required "FileName" column.
        Also raised if the first feature fails the filename/geometry consistency check.
    """
    gdf = gpd.read_file(in_path)
    if gdf.empty:
        raise SystemExit("Input has no features.")

    # Ensure CRS is RD New; reproject if needed.
    if gdf.crs is None:
        raise SystemExit("Input has no CRS; please define or reproject to EPSG:28992.")
    if gdf.crs.to_epsg() != 28992:
        gdf = gdf.to_crs(epsg=28992)

    # Check required columns
    if "FileName" not in gdf.columns:
        raise SystemExit("Expected a 'FileName' column in the input.")

    # Normalize multiparts/geometry collections to single polygons
    gdf = gdf.explode(index_parts=False, ignore_index=True)

    # ---- Validate first feature's filename vs geometry indices ----
    first = gdf.iloc[0]
    try:
        ix_fn, iy_fn = parse_ix_iy(first["FileName"])
    except ValueError as e:
        raise SystemExit(f"Filename check failed: {e}") from e

    ix_geom, iy_geom = ix_iy_from_lower_left(first.geometry)
    assert (ix_fn, iy_fn) == (ix_geom, iy_geom), (
        "Filename check failed for first feature: "
        f"FileName → ({ix_fn},{iy_fn}) vs geometry → ({ix_geom},{iy_geom})."
    )

    # Determine index ranges to cover the entire dataset
    minx, miny, maxx, maxy = gdf.total_bounds
    ix_min, iy_min, ix_max, iy_max = snap_bbox_to_index_ranges(minx, miny, maxx, maxy)

    # Generate all tiles within the inclusive index ranges
    new_features = []
    for ix in range(ix_min, ix_max + 1):
        x0 = ix * TILE_SIZE
        for iy in range(iy_min, iy_max + 1):
            y0 = iy * TILE_SIZE
            geom = box(x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)
            filename = f"filtered_{ix}_{iy}.laz"
            new_features.append(
                {
                    "FileName": filename,
                    "geometry": geom,
                }
            )

    if new_features:
        new_gdf = gpd.GeoDataFrame(
            new_features,
            geometry="geometry",
            crs=gdf.crs,
        )
    else:
        new_gdf = gpd.GeoDataFrame(
            columns=gdf.columns,
            geometry="geometry",
            crs=gdf.crs,
        )

    new_gdf.to_file(out_path, driver="GPKG")
    print(f"Done. Features: {len(new_gdf)}. Output: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Create a filled metadata grid based on Cyclomedia formatting."
    )
    ap.add_argument(
        "--input",
        help=(
            "Input .gpkg path (Cyclomedia tile grid to base the new grid on). "
            "A reference Cyclomedia metadata file is needed for the new file "
            "to be exactly aligned with Cyclomedia."
        ),
    )
    ap.add_argument(
        "--output",
        default="filled.gpkg",
        help="Output .gpkg path to the new Cyclomedia-formatted grid.",
    )
    args = ap.parse_args()
    main(args.input, args.output)
