import os
import re

import pandas as pd
import geopandas as gpd

from pc_prep.general_prep.utils import convert_df_to_gdf
from pc_prep.logger import logger


def prepare_pc_paths(
    tree_df_path: str,
    pc_metadata_df_path: str,
    mounted_pc_path: str,
) -> list[str]:
    """
    Build a list of unique point cloud tile paths that contain at least one tree.

    Workflow:
    1. Load and preprocess:
       - tree data (pickled DataFrame with tree locations)
       - point cloud tile index (GeoPackage with tile footprints and metadata)
    2. Spatially intersect trees with tile footprints to find which tiles contain trees.
    3. Convert those tile metadata records into full absolute filepaths on disk.
    4. Sort tiles by how many trees they contain (descending), then return the paths.

    Parameters
    ----------
    tree_df_path : str
        Path to the pickled tree DataFrame (.pkl). This should include tree locations
        and any ID column used for grouping (e.g. 'ID').
    pc_metadata_df_path : str
        Path to the point cloud tile metadata (.gpkg). This should contain, at minimum,
        tile geometries and either:
        - 'Basename' + 'FileName' (Cyclomedia-style MLS capture), OR
        - just 'FileName' (e.g. drone tiles).
    mounted_pc_path : str
        Root directory under which the point cloud tiles are mounted.
        Example layout for MLS data:
        <mounted_pc_path>/
            2023/
                LAZ/
                    nl-rott-230412-7415-laz/
                        las_processor_bundled_out/
                            <FileName>.las

    Returns
    -------
    list[str]
        A list of absolute file paths to point cloud tiles, sorted so that
        tiles with the most trees appear first.

    Notes
    -----
    This function does not touch disk except for reading the inputs.
    It does not validate that the constructed filepaths actually exist.
    """
    logger.info("Reading tree df and pointcloud metadata df.")
    tree_gdf = preprocess_tree_df(tree_df_path)
    pc_tile_metadata_gdf = preprocess_pc_metadata(pc_metadata_df_path)

    logger.info("Filtering pointcloud tiles that contain trees (spatial overlay).")
    trees_in_pc_tiles = gpd.overlay(
        df1=tree_gdf,
        df2=pc_tile_metadata_gdf,
        how="intersection",
    )

    trees_in_pc_tiles = determine_full_filepath_from_laz_metadata(
        gdf=trees_in_pc_tiles,
        mount_path=mounted_pc_path,
    )

    logger.info("Sorting tiles by number of matched trees.")
    trees_per_tile = (
        trees_in_pc_tiles[["filepath", "ID"]]
        .groupby("filepath")
        .agg(len)
        .sort_values(by="ID", ascending=False)
    )

    pc_paths = list(trees_per_tile.index)
    return pc_paths


def determine_full_filepath_from_laz_metadata(
    gdf: gpd.GeoDataFrame,
    mount_path: str,
    suffix_start: str = "nl-rott-",
) -> gpd.GeoDataFrame:
    """
    Add a 'filepath' column to a GeoDataFrame of tile metadata, pointing to the
    expected absolute path of each LAS/LAZ file on disk.

    There are (at least) two supported layouts:

    1. Cyclomedia MLS-style (has a 'Basename' column):
       The 'Basename' looks like 'nl-rott-230412-7415-laz', which encodes
       capture date and area code.
       The actual tile is assumed to be stored under:
           <mount_path>/
               20<yy>/
                   LAZ/
                       <Basename>/
                           las_processor_bundled_out/
                               <FileName>
       where <yy> is taken from the yymmdd portion inside `Basename`.

    2. Drone-style (no 'Basename' column):
       We assume files are stored flat under mount_path:
           <mount_path>/<FileName>

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with metadata. Must contain at least 'FileName'.
        If it also contains 'Basename', it's treated as Cyclomedia MLS-style data.
    mount_path : str
        Root directory under which point clouds are mounted.
    suffix_start : str, default "nl-rott-"
        Prefix used to parse basename tokens like "nl-rott-230412-7415-laz".

    Returns
    -------
    geopandas.GeoDataFrame
        The same GeoDataFrame with an additional 'filepath' column
        giving the full expected path to each point cloud tile.

    Raises
    ------
    AssertionError
        If 'Basename' is present but cannot be parsed into a valid year
        (handled by `modify_string`).
    """
    if "Basename" in gdf.columns:
        # Cyclomedia MLS layout.
        # Example Basename: "nl-rott-230412-7415-laz"
        # Extract "23" (yy) to build "20<yy>" = "2023".
        str_ind = len(suffix_start)
        gdf["filepath"] = gdf.apply(
            lambda x: os.path.join(
                mount_path,
                f"20{modify_string(x['Basename'], suffix_start=suffix_start)[str_ind:str_ind+2]}",
                "LAZ",
                modify_string(x["Basename"]),
                "las_processor_bundled_out",
                x["FileName"],
            ),
            axis=1,
        )
    else:
        # Drone-style / flat layout.
        gdf["filepath"] = gdf.apply(
            lambda x: os.path.join(
                mount_path,
                x["FileName"],
            ),
            axis=1,
        )

    return gdf


def get_collection_info(pc_filename: str) -> str:
    """
    Extract the point cloud collection code embedded in a filename or path.

    The collection code looks like:
        nl-rott-230412-7415-laz
    which encodes capture date (e.g. 230412 = 2023-04-12) and area code (e.g. 7415).

    We return exactly that substring, including the 'nl-rott-' prefix and '-laz' suffix.

    Parameters
    ----------
    pc_filename : str
        The filename or path of the point cloud tile. This should contain
        something like 'nl-rott-230412-7415-laz'.

    Returns
    -------
    str
        The extracted collection code, e.g. 'nl-rott-230412-7415-laz'.

    Raises
    ------
    ValueError
        If the code cannot be found in the provided string.
    """
    prefix = "nl-rott"
    suffix = "laz"

    # Pattern: nl-rott-(anything non-greedy)-laz
    pattern = rf"{prefix}-(.*?)-{suffix}"
    match = re.search(pattern, pc_filename)

    if match:
        captured_part = match.group(1)
        return f"{prefix}-{captured_part}-{suffix}"

    raise ValueError(
        "Cannot determine collection info from input point cloud filename: "
        f"{pc_filename!r}"
    )


def modify_string(
    str_to_modify: str,
    suffix_start: str = "nl-rott-",
) -> str:
    """
    Normalize a Cyclomedia-style basename by trimming any leading junk
    before the expected code.

    The basename is expected to contain:
        {suffix_start}{yymmdd}-{areacode}-laz
    e.g.
        nl-rott-190411-7415-laz
        nl-rott-230412-7415-laz

    This function:
    - Finds the first occurrence of `suffix_start`.
    - Returns the substring starting at that point.
    - Asserts that the 'yy' portion of yymmdd is within an allowed range.

    Parameters
    ----------
    str_to_modify : str
        Raw basename string to clean.
    suffix_start : str, default "nl-rott-"
        Prefix that marks the start of the expected pattern.

    Returns
    -------
    str
        The cleaned string starting at `suffix_start`, e.g.
        "nl-rott-230412-7415-laz".

    Raises
    ------
    AssertionError
        If the string does not contain the expected pattern,
        or the 2-digit year code is not in the supported range.
    """
    MIN_YEAR = 17
    MAX_YEAR = 32

    assert suffix_start in str_to_modify, (
        "String should contain form "
        f"'{suffix_start}yymmdd-areacode-laz' but doesn't. "
        f"Got: {str_to_modify}"
    )

    # Keep from first occurrence of suffix_start onward
    modified_s = str_to_modify[str_to_modify.find(suffix_start) :]

    # Validate that the yy in yymmdd appears sane
    yy_str = modified_s[len(suffix_start) : len(suffix_start) + 2]
    assert yy_str.isdigit() and MIN_YEAR <= int(yy_str) <= MAX_YEAR, (
        "The yy value in yymmdd is invalid. "
        f"Expected a 2-digit year between {MIN_YEAR} and {MAX_YEAR}, "
        f"got {modified_s!r}"
    )

    return modified_s


def preprocess_tree_df(tree_df_path: str) -> gpd.GeoDataFrame:
    """
    Load and prepare the tree DataFrame, then convert it into a GeoDataFrame.

    Steps:
    - Read the pickled pandas DataFrame from disk.
    - Convert it to a GeoDataFrame using `convert_df_to_gdf`, which is
      expected to create geometries (e.g. points) in the correct CRS.

    Parameters
    ----------
    tree_df_path : str
        Path to the pickled pandas DataFrame of trees (.pkl).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with at least:
        - tree identifiers (e.g. 'ID')
        - point geometries representing tree locations
    """
    tree_df = pd.read_pickle(tree_df_path)
    tree_gdf = convert_df_to_gdf(df=tree_df)
    return tree_gdf


def preprocess_pc_metadata(mls_metadata_path: str) -> gpd.GeoDataFrame:
    """
    Load and normalize the point cloud tile metadata.

    Steps:
    - Read the tile metadata (GeoPackage / GeoDataFrame).
    - Reproject to EPSG:28992 if the file is still in EPSG:4326 (WGS84).
    - Rename certain columns to a standard schema, e.g. "Folder" -> "Basename"
      to match downstream assumptions in `determine_full_filepath_from_laz_metadata`.

    Parameters
    ----------
    mls_metadata_path : str
        Path to the point cloud tile metadata file (.gpkg or similar).
        For MLS data, this should describe tile footprints and fields such as
        'FileName', 'Folder', etc.

    Returns
    -------
    geopandas.GeoDataFrame
        The cleaned metadata GeoDataFrame. Guaranteed to:
        - have a projected CRS (EPSG:28992),
        - have a 'Basename' column if 'Folder' originally existed.
    """
    mls_laz_metadata = gpd.read_file(mls_metadata_path)

    # Normalize CRS: prefer RD New (EPSG:28992) for Dutch geodata.
    if mls_laz_metadata.crs == "EPSG:4326":
        mls_laz_metadata.to_crs(epsg=28992, inplace=True)

    # Standardize naming:
    mls_laz_metadata.rename(columns={"Folder": "Basename"}, inplace=True)

    return mls_laz_metadata
