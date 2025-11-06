import os
import argparse

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.errors import GEOSException

from pc_prep.logger import logger


def prepare_output_paths(
    args: argparse.Namespace,
    pc_path: str,
    pc_output_path: str,
    tree_gpkg_path: str,
    bgt_pavements_path: str,
    image_path: str,
    prompt_path: str,
) -> tuple[str, str, str, str, str, str]:
    """
    Produce relative (basename-like) versions of several absolute output paths.

    This trims known base directories (from `args`) off of full paths so that
    downstream code / metadata JSON can store compact, repo-friendly paths
    instead of long absolute paths.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments namespace. Must include:
        - pc_raw
        - pc_dir
        - bgt_dir
        - img_dir
    pc_path : str
        Absolute path to the raw point cloud tile.
    pc_output_path : str
        Absolute path to the processed point cloud output file (.laz).
    tree_gpkg_path : str
        Absolute path to the GeoPackage with tree geometries ("obsurv_trees.gpkg").
    bgt_pavements_path : str
        Absolute path to the processed BGT pavements GeoPackage.
    image_path : str
        Absolute path to the generated raster image (TIFF).
    prompt_path : str
        Absolute path to the prompt metadata file associated with the image.

    Returns
    -------
    tuple[str, str, str, str, str, str]
        (
            pc_path_basename,
            pc_output_path_basename,
            tree_gpkg_path_basename,
            bgt_pavements_path_basename,
            image_path_basename,
            prompt_path_basename,
        )

        Each value is the input path with its known base directory stripped.
        For example, `pc_output_path_basename` is `pc_output_path` with
        `args.pc_dir + "/"` removed from the start.

    Notes
    -----
    This uses simple string replacement, so it assumes the provided path
    actually starts with the specified base directory.
    """
    pc_path_basename = pc_path.replace(os.path.join(args.pc_raw, ""), "")
    pc_output_path_basename = pc_output_path.replace(os.path.join(args.pc_dir, ""), "")
    tree_gpkg_path_basename = tree_gpkg_path.replace(os.path.join(args.pc_dir, ""), "")
    bgt_pavements_path_basename = bgt_pavements_path.replace(
        os.path.join(args.bgt_dir, ""), ""
    )
    image_path_basename = image_path.replace(os.path.join(args.img_dir, ""), "")
    prompt_path_basename = prompt_path.replace(os.path.join(args.img_dir, ""), "")

    return (
        pc_path_basename,
        pc_output_path_basename,
        tree_gpkg_path_basename,
        bgt_pavements_path_basename,
        image_path_basename,
        prompt_path_basename,
    )


def setup_pc_paths(
    pc_path: str,
    pc_code: str,
    pc_out_dir: str,
    bgt_out_dir: str,
) -> tuple[str, str, str]:
    """
    Construct output paths for all per-tile products derived from a single point cloud tile.

    Output layout for a tile named `<basename>` with capture code `<pc_code>` looks like:
        <pc_out_dir>/<basename>/<pc_code>/<basename>.laz
        <pc_out_dir>/<basename>/<pc_code>/obsurv_trees.gpkg
        <bgt_out_dir>/<basename>/<pc_code>/BGT_pavements.gpkg

    This function also ensures those directories exist.

    Parameters
    ----------
    pc_path : str
        Path to the source (raw) point cloud tile.
    pc_code : str
        Point cloud collection identifier (e.g. "nl-rott-230412-7415-laz").
        Used to organize output by acquisition/area.
    pc_out_dir : str
        Base directory where processed point cloud data and tree outputs will be written.
    bgt_out_dir : str
        Base directory where processed BGT pavement data will be written.

    Returns
    -------
    tuple[str, str, str]
        (
            pc_output_path,
            tree_gpkg_path,
            bgt_outfile,
        )
        where:
        - pc_output_path : str
            Output .laz path for the processed (filtered/merged/etc.) point cloud.
        - tree_gpkg_path : str
            Output GeoPackage path for tree geometries inside the tile.
        - bgt_outfile : str
            Output GeoPackage path for extracted BGT pavements.

    Notes
    -----
    This function will `os.makedirs(..., exist_ok=True)` for all required subdirectories.
    """
    basename = os.path.splitext(os.path.basename(pc_path))[0]

    pc_output_path = os.path.join(pc_out_dir, basename, pc_code, basename + ".laz")
    bgt_output_path = os.path.join(bgt_out_dir, basename, pc_code, basename + ".laz")

    local_pc_dir = os.path.dirname(pc_output_path)
    local_bgt_dir = os.path.dirname(bgt_output_path)

    os.makedirs(local_pc_dir, exist_ok=True)
    os.makedirs(local_bgt_dir, exist_ok=True)

    tree_gpkg_path = os.path.join(local_pc_dir, "obsurv_trees.gpkg")
    bgt_outfile = os.path.join(local_bgt_dir, "BGT_pavements.gpkg")

    return pc_output_path, tree_gpkg_path, bgt_outfile


def tree_pc_ready_for_img_conversion(
    trees_in_pc: gpd.GeoDataFrame,
    pc_points: np.ndarray,
) -> bool:
    """
    Check whether a point cloud tile is suitable for rasterization / segmentation.

    Conditions checked:
    - There must be at least one tree geometry inside the tile.
    - There must be at least one (filtered) non-ground point in the tile.

    Parameters
    ----------
    trees_in_pc : geopandas.GeoDataFrame
        Trees that fall within the tile (after spatial overlay). May be empty.
    pc_points : numpy.ndarray
        Filtered LiDAR / MLS / drone points (e.g. non-ground points).
        Can be empty if nothing survived filtering.

    Returns
    -------
    bool
        True if the tile is usable for image conversion. False otherwise.

    Side effects
    ------------
    Logs reasons for rejection using `logger.error`.
    """
    if trees_in_pc.empty:
        logger.error("No trees inside pointcloud area. Skipping segmentation.")
        return False

    if len(pc_points) == 0:
        logger.error("No filtered pointcloud points found.")
        return False

    return True


def determine_tree_area_of_interest(
    bounds_gdf: gpd.GeoDataFrame,
    trees_gdf: gpd.GeoDataFrame,
    total_pc_bounds: gpd.GeoDataFrame,
    pc_code: str,
    tree_span_buffer: float,
) -> gpd.GeoDataFrame:
    """
    Determine the effective area of interest (AOI) for a point cloud tile.

    Why this matters:
    Point clouds are typically split into 50m x 50m tiles. But trees (their crowns,
    especially) can spill across tile borders. We don't want to crop a tree in half.
    So this function expands the AOI to include relevant neighboring tile regions.

    High-level steps
    ----------------
    1. Buffer each tree in the current tile by `tree_span_buffer` meters
       (approximate crown radius).
    2. Overlay these buffered trees with the global tile index (`total_pc_bounds`)
       to find which neighboring tiles intersect those crowns.
    3. Union the buffered geometry and the current tile bounds, then take the
       envelope (bounding box).
    4. Clip that envelope back to the relevant neighboring tiles. Filter out
       slivers / artifacts.

    Parameters
    ----------
    bounds_gdf : geopandas.GeoDataFrame
        Geometry representing the spatial bounds of the current tile.
    trees_gdf : geopandas.GeoDataFrame
        Trees located (at least partially) in this tile.
    total_pc_bounds : geopandas.GeoDataFrame
        Bounds for all available tiles in the dataset. Must include:
        - "Basename" and/or "FileName" columns so we can filter tiles
          that match the current capture (`pc_code`).
    pc_code : str
        Point cloud capture identifier, e.g. "nl-rott-230412-7415-laz".
        Used to select only tiles from the same acquisition batch.
    tree_span_buffer : float
        Buffer radius in meters to apply around each tree point.
        This is used as a proxy for canopy extent.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame of tile geometries (subset of `total_pc_bounds`)
        that intersect the AOI of the current tile, with very tiny pieces removed.

    Notes
    -----
    - Uses CRS EPSG:28992 (RD New) assumptions for buffering and area math.
    - Filters out polygons smaller than MIN_AREA to avoid geometric noise.
    - If unioning geometries raises a topology error, applies a tiny buffer
      to clean them and retries.
    """
    assert (
        pc_code is not None
    ), "Cannot determine tile AOI without the pointcloud capture info."

    TILE_WIDTH = 50  # meters
    MIN_AREA = 0.5  # m²; filters out tiny slivers from overlay/rounding

    # Heuristic: estimate the fraction of a 50x50 tile you'd expect to
    # overlap with neighbors if you buffer trees by tree_span_buffer.
    perc_overlap = (
        4 * tree_span_buffer * TILE_WIDTH - 4 * tree_span_buffer * tree_span_buffer
    ) / (TILE_WIDTH**2)

    logger.info(
        "Including neighbouring tiles if crowns cross tile edges. "
        "This ensures full crowns for segmentation."
    )
    logger.info(
        "Estimated fraction of tile area affected with tree_span_buffer=%s: %s",
        tree_span_buffer,
        perc_overlap,
    )

    # Limit search to tiles from the same acquisition (same pc_code).
    # For Cyclomedia MLS data, pc_code appears in 'Basename'.
    # For drone data, pc_code appears in 'FileName'.
    if "Basename" in total_pc_bounds.columns:
        total_pc_subset = total_pc_bounds[
            total_pc_bounds["Basename"].apply(lambda x: pc_code in x)
        ]
    else:
        total_pc_subset = total_pc_bounds[
            total_pc_bounds["FileName"].apply(lambda x: pc_code in x)
        ]

    # 1. Buffer trees
    buffered_trees = gpd.GeoDataFrame(
        geometry=trees_gdf.buffer(tree_span_buffer),
        crs=trees_gdf.crs,
    )

    # 2. Find overlapping neighbour tiles for those buffered crowns
    trees_in_neighbouring_tiles_gdf = gpd.overlay(
        df1=buffered_trees,
        df2=total_pc_subset,
        how="intersection",
    )

    # 3. Union current tile bounds + neighbour overlaps, then envelope
    geoms_of_interest = pd.concat(
        [trees_in_neighbouring_tiles_gdf, bounds_gdf]
    ).geometry

    try:
        # unary_union may raise topology errors if geometries are invalid
        pc_aoi = unary_union(geoms_of_interest).envelope
    except GEOSException as e:
        logger.error(
            "Topology error in unary_union. "
            "Applying tiny buffer (1e-8) to clean geometries. Full error: %s",
            e,
        )
        pc_aoi = unary_union(geoms_of_interest.buffer(1e-8))

    # 4. Intersect that AOI with the tile index again to get clipped per-tile shapes
    tiles_aoi = gpd.overlay(
        df1=gpd.GeoDataFrame(geometry=[pc_aoi], crs="EPSG:28992"),
        df2=total_pc_subset,
        how="intersection",
    )

    # Filter out geometric crumbs that appear due to numeric instability.
    tiles_aoi = tiles_aoi[tiles_aoi.geometry.area >= MIN_AREA]

    return tiles_aoi


def get_trees_from_tile(
    bounds_gdf: gpd.GeoDataFrame,
    tree_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip trees to a tile boundary.

    Parameters
    ----------
    bounds_gdf : geopandas.GeoDataFrame
        Bounds of the point cloud tile (usually a single polygon).
    tree_gdf : geopandas.GeoDataFrame
        All tree points/polygons to test against `bounds_gdf`.

    Returns
    -------
    geopandas.GeoDataFrame
        Trees that fall within (intersect) the provided tile bounds.
        If `tree_gdf` is empty, returns an empty GeoDataFrame.

    Notes
    -----
    Uses `gpd.overlay(..., how="intersection")`, so trees that only partially
    overlap will still be included.
    """
    if tree_gdf.empty:
        return gpd.GeoDataFrame([])

    logger.info("Intersecting tree geometries with tile bounds.")
    trees_in_laz = gpd.overlay(df1=tree_gdf, df2=bounds_gdf, how="intersection")
    return trees_in_laz
