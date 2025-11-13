import os

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pdal

from pc_prep.logger import logger


def setup_img_paths(
    output_dir: str,
    pc_code: str,
    basename: str,
) -> tuple[str, str, str]:
    """
    Build output file paths for all raster/image-related products of a tile.

    The directory structure is:
        <output_dir>/<basename>/<pc_code>/
            raster.tif
            img_with_prompts.png
            point_prompt.json

    The directory is created if it doesn't exist.

    Parameters
    ----------
    output_dir : str
        Root directory where image artifacts should be written.
    pc_code : str
        Acquisition identifier / collection code for this tile
        (e.g. "nl-rott-230412-7415-laz").
    basename : str
        Base name of the tile, e.g. "filtered_1846_8712". Used to group outputs.

    Returns
    -------
    tuple[str, str, str]
        (
            image_path,
            prompt_img_path,
            prompt_path,
        )

        where:
        - image_path : str
            GeoTIFF path for the rasterized point cloud (top-down RGB).
        - prompt_img_path : str
            Debug PNG path (raster + prompt dots visualized).
        - prompt_path : str
            JSON path containing point prompt coordinates/labels for SAM-like models.
    """
    outdir = os.path.join(output_dir, basename, pc_code)
    os.makedirs(outdir, exist_ok=True)

    image_path = os.path.join(outdir, "raster.tif")
    prompt_img_path = os.path.join(outdir, "img_with_prompts.png")
    prompt_path = os.path.join(outdir, "point_prompt.json")

    return image_path, prompt_img_path, prompt_path


def enlarge_pc_tile(
    pc_path: str,
    tiles_aoi: gpd.GeoDataFrame,
    outfname: str,
) -> bool:
    """
    Expand a tile's point cloud by merging neighboring tiles and clipping
    to the local area of interest (AOI).

    High-level workflow:
    1. Identify neighboring tile point clouds from `tiles_aoi`.
       Only keep tiles whose file size is above a minimum threshold.
    2. Merge those tiles into a temporary combined point cloud.
    3. Clip that merged point cloud to just the AOI bounds using `filter_pointcloud_to_aoi`.
    4. Validate the final clipped tile based on number of points.

    Parameters
    ----------
    pc_path : str
        Path to the reference point cloud file (used primarily to locate sibling tiles
        on disk and infer directory structure).
    tiles_aoi : geopandas.GeoDataFrame
        Metadata for the tiles determined to be spatially relevant to this tile
        (e.g. output of `determine_tree_area_of_interest`). Must have a "FileName"
        column listing candidate neighbor tile filenames.
    outfname : str
        Output path for the enlarged (merged + clipped) point cloud tile (.laz).

    Returns
    -------
    bool
        True if the enlarged/clipped point cloud is valid and written to `outfname`,
        False otherwise.

    Notes
    -----
    - A temporary merged file `<outfname>_tmp.laz` is created and deleted.
    - Tiles below a minimum size threshold are ignored entirely.
    - If no valid tiles are found, returns False.
    """
    MIN_FILE_SIZE = 512  # kB; tiles smaller than this are considered junk/incomplete

    os.makedirs(os.path.dirname(outfname), exist_ok=True)
    outfname_tmp = outfname.replace(".laz", "_tmp.laz")
    pc_dir = os.path.dirname(pc_path)

    # Collect candidate tile paths, but skip tiny/broken tiles.
    pc_paths = [
        os.path.join(pc_dir, os.path.basename(tile))
        for tile in tiles_aoi["FileName"]
        if os.path.getsize(os.path.join(pc_dir, os.path.basename(tile))) > MIN_FILE_SIZE
    ]

    if len(pc_paths) == 0:
        logger.error("No valid point cloud tiles found for %s", pc_path)
        return False

    # Merge candidate tiles into a temp LAS/LAZ.
    merge_point_clouds_with_pdal_pipeline(
        file_paths=pc_paths,
        outfname=outfname_tmp,
    )

    # Clip merged file down to the AOI bounds and validate point count.
    success = filter_pointcloud_to_aoi(
        input_file=outfname_tmp,
        tiles_aoi=tiles_aoi,
        output_file=outfname,
    )
    return success


def filter_pointcloud_to_aoi(
    input_file: str,
    tiles_aoi: gpd.GeoDataFrame,
    output_file: str,
    min_points: int = 1_000_000,
) -> bool:
    """
    Spatially crop a merged point cloud to the AOI and validate the result.

    What it does:
    - Reads `input_file` (a merged tile).
    - Computes the AOI bounding box from `tiles_aoi`.
    - Streams through the LAS/LAZ in chunks, writing out only points whose (x, y)
      fall within that bounding box.
    - Drops both the merged file and the filtered output if the surviving point
      count is too low.

    Parameters
    ----------
    input_file : str
        Path to the merged temporary LAS/LAZ file.
    tiles_aoi : geopandas.GeoDataFrame
        Tiles of interest. Only their combined bounding box is used here.
    output_file : str
        Path to write the filtered (clipped) LAS/LAZ file.
    min_points : int, default 1_000_000
        Minimum number of points required for the final AOI-clipped output
        to be considered valid.

    Returns
    -------
    bool
        True if the filtered file was successfully created and contains at least
        `min_points` points. False if it's below threshold (and cleaned up).

    Notes
    -----
    - The input merged file is always removed at the end.
    - The output file is also removed if it doesn't meet `min_points`.
    """
    # Bounding box of AOI
    minx, miny, maxx, maxy = tiles_aoi.total_bounds

    with laspy.open(input_file) as las_reader:
        total_points = las_reader.header.point_count

        # Quick sanity check: avoid work if total merged set is already tiny.
        if total_points < min_points:
            logger.info(
                "Skipping %s — only %s points in merged file (min %s).",
                input_file,
                total_points,
                min_points,
            )
            os.remove(input_file)
            return False

        header = las_reader.header
        filtered_points_total = 0

        # Write filtered chunks directly into the output file.
        with laspy.open(output_file, mode="w", header=header) as las_writer:
            for points in las_reader.chunk_iterator(1_000_000):
                x = points.x
                y = points.y

                # Axis-aligned bounding box mask
                mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)

                count_chunk = np.count_nonzero(mask)
                filtered_points_total += count_chunk

                if count_chunk > 0:
                    las_writer.write_points(points[mask])

    # If the AOI-clipped file has too few points, delete it.
    if filtered_points_total < min_points:
        logger.info(
            "Skipping %s — only %s points in AOI after filtering (min %s).",
            input_file,
            filtered_points_total,
            min_points,
        )
        os.remove(output_file)
        os.remove(input_file)
        return False

    # Cleanup merged temp file if we kept the filtered data
    os.remove(input_file)
    return True


def merge_point_clouds_with_pdal_pipeline(
    file_paths: list[str],
    outfname: str,
) -> str:
    """
    Merge multiple LAS/LAZ point clouds into a single file using a PDAL pipeline.

    Parameters
    ----------
    file_paths : list[str]
        List of LAS/LAZ files to be merged.
    outfname : str
        Output path for the merged LAS/LAZ file.

    Returns
    -------
    str
        The path to the merged file (`outfname`).

    Notes
    -----
    - This builds a minimal PDAL pipeline dynamically by chaining multiple
      Readers (one per file) into a single Writer.
    """
    pipeline = pdal.Pipeline()

    # Each file becomes a reader stage in the pipeline
    for file_path in file_paths:
        pipeline |= pdal.Reader.las(filename=file_path)

    # Then write them all into a single LAS/LAZ
    pipeline |= pdal.Writer.las(filename=outfname)

    pipeline.execute()
    return outfname


def convert_df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert a tabular DataFrame with WKT geometries into a GeoDataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table. Must contain:
        - 'ID': unique identifier (tree ID, etc.)
        - 'WKT_GEOMETRIE': geometry column in WKT form.
        CRS is assumed to be EPSG:28992.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with:
        - geometry column (parsed from WKT)
        - ID column copied from input
        If the input DataFrame is empty, returns an empty GeoDataFrame.
    """
    if df.empty:
        return gpd.GeoDataFrame([])

    gdf = gpd.GeoDataFrame(
        columns=["geometry"],
        data=gpd.GeoSeries.from_wkt(df["WKT_GEOMETRIE"], crs="EPSG:28992"),
    )
    gdf["ID"] = df["ID"]

    return gdf


def get_pc_bounds(
    pc_path: str,
    mls_metadata_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Extract the polygon bounds of a specific point cloud tile.

    We match the filename of `pc_path` against the "FileName" field in
    `mls_metadata_gdf`, grab the corresponding geometry, and return it as a
    one-row GeoDataFrame.

    Parameters
    ----------
    pc_path : str
        Path to the point cloud tile on disk.
    mls_metadata_gdf : geopandas.GeoDataFrame
        Metadata index describing all tiles in the dataset. Must include:
        - "FileName" column containing filenames
        - valid geometries in EPSG:28992

    Returns
    -------
    geopandas.GeoDataFrame
        Single-row GeoDataFrame with the tile's polygon geometry, CRS EPSG:28992.

    Raises
    ------
    IndexError
        If no matching row is found for the given `pc_path`.
    """
    logger.info("Determining bounds for the point cloud.")

    match_mask = mls_metadata_gdf["FileName"].apply(
        os.path.basename
    ) == os.path.basename(pc_path)
    laz_bounds_poly = mls_metadata_gdf[match_mask].geometry.values[0]

    laz_bounds_gdf = gpd.GeoDataFrame(
        gpd.GeoSeries(data=[laz_bounds_poly], crs="EPSG:28992"),
        columns=["geometry"],
    )
    return laz_bounds_gdf


def read_laz_points(input_file: str) -> np.ndarray:
    """
    Read a LAS/LAZ file and return XYZRGB as a NumPy array.

    The RGB channels are normalized to [0, 1].

    Parameters
    ----------
    input_file : str
        Path to a LAS/LAZ file.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 6):
        [X, Y, Z, R, G, B]
        where R, G, B are floats in [0, 1].

    Notes
    -----
    - Uses `laspy.read`, so the file must be readable by laspy.
    - This does not perform any ground / non-ground filtering; it's raw data.
    """
    las = laspy.read(input_file)
    pcd = las.points

    points = np.vstack(
        (
            pcd.x,
            pcd.y,
            pcd.z,
            pcd.red / 255.0,
            pcd.green / 255.0,
            pcd.blue / 255.0,
        )
    ).T

    return points
