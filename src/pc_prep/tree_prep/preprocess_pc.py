import geopandas as gpd
import numpy as np

from pc_prep.general_prep.utils import (
    get_pc_bounds,
    enlarge_pc_tile,
)
from pc_prep.tree_prep.utils import (
    determine_tree_area_of_interest,
    get_trees_from_tile,
)
from pc_prep.pavement_prep.utils import get_relevant_assets

from pc_prep.tree_prep.metadata_handler import (
    preprocess_tree_df,
    preprocess_pc_metadata,
)

from pc_prep.pavement_prep.csf_with_pavements import add_asset_classification_to_pc
from pc_prep.logger import logger


def preprocess_pc(
    pc_path: str,
    pc_code: str,
    pc_metadata_df_path: str,
    tree_df_path: str,
    bgt_df_path: str,
    pc_output_path: str,
    tree_gpkg_path: str,
    bgt_pavements_path: str,
    tree_span_buffer: int = 8,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Prepares and processes LiDAR point cloud data for further analysis by segmenting trees,
    merging point cloud tiles, and extracting pavement features.

    This function performs the following steps:
    1. Sets up paths for various output files based on the input parameters.
    2. Checks if the output file for tree segmentation already exists; if so, skips the processing.
    3. Loads metadata and retrieves bounds of the LiDAR data.
    4. Loads tree geometries and determines the area of interest based on tree locations.
    5. Merges point cloud tiles and applies filtering.
    6. Processes BGT data for pavement extraction.
    7. Applies the CSF filtering and segments the pavements.

    Args:
        pc_path (str): Path to the input LiDAR point cloud file.
        pc_code (str): Details on the specific recording of the pointcloud.
        pc_metadata_df_path (str): pointcloud tile metadata filepath.
        bgt_df_path (str): BGT pavements filepath.
        tree_df_path (str): .pkl trees filepath.
        tree_span_buffer (int): Buffer to use around tree point geometries to determine pc AOI.

    Returns:
        Tuple[gpd.GeoDataFrame,np.ndarray]:
            - GeoDataFrame of trees extracted from the LiDAR data.
            - NumPy array of non-ground points extracted from the point cloud.

    Raises:
        Exception: Logs any errors encountered during processing steps.
    """
    pc_metadata_gdf = preprocess_pc_metadata(mls_metadata_path=pc_metadata_df_path)
    tree_gdf = preprocess_tree_df(tree_df_path=tree_df_path)

    logger.info("Determining area of interest based on tree locations")
    laz_bounds_gdf = get_pc_bounds(pc_path=pc_path, mls_metadata_gdf=pc_metadata_gdf)
    trees_in_laz = get_trees_from_tile(bounds_gdf=laz_bounds_gdf, tree_gdf=tree_gdf)
    tiles_aoi = determine_tree_area_of_interest(
        bounds_gdf=laz_bounds_gdf,
        trees_gdf=trees_in_laz,
        total_pc_bounds=pc_metadata_gdf,
        pc_code=pc_code,
        tree_span_buffer=tree_span_buffer,
    )
    if trees_in_laz.empty:
        logger.error("No trees in pc tile. Skipping.")
        non_ground_points = np.array([])
        return (
            trees_in_laz,
            non_ground_points,
        )

    logger.info("Writing OBSURV tree geometries to gpkg.")
    trees_in_laz.to_file(tree_gpkg_path, driver="GPKG")

    logger.info("Potentially enlarging pointcloud to match area of interest.")
    valid_pc = enlarge_pc_tile(
        pc_path=pc_path, tiles_aoi=tiles_aoi, outfname=pc_output_path
    )

    if not valid_pc:
        logger.error("No valid pointcloud. Skipping.")
        non_ground_points = np.array([])
        return (
            trees_in_laz,
            non_ground_points,
        )

    logger.info("Add asset and ground classification to pointcloud.")
    get_relevant_assets(
        asset_df_path=bgt_df_path,
        bounds_gdf=laz_bounds_gdf,
        bgt_outfile=bgt_pavements_path,
    )
    non_ground_points, _, _, _ = add_asset_classification_to_pc(
        input_file=pc_output_path, asset_file=bgt_pavements_path
    )

    return (
        trees_in_laz,
        non_ground_points,
    )
