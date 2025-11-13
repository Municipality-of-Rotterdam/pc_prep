import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from pc_prep.pavement_prep.config import AssetClassifier
from pc_prep.logger import logger


def convert_asset_to_single_layer_gpkg(
    asset_gdf: gpd.GeoDataFrame,
    output_filename: str,
    data_class: AssetClassifier,
) -> None:
    """
    Convert raw asset data into a normalized single-layer GeoPackage.

    The function:
    - Maps raw class labels in the "KLASSE" column to numeric class IDs.
    - Assigns a per-class instance index.
    - Filters out assets that are not part of the known/allowed classes.
    - Writes the result to a GeoPackage with layer name "asset".

    Parameters
    ----------
    asset_gdf : geopandas.GeoDataFrame
        Input asset geometries and attributes. Must contain a "KLASSE" column.
    output_filename : str
        Output path for the resulting GeoPackage (.gpkg).
    data_class : AssetClassifier
        Dataclass-like config that defines:
        - `classifications`: mapping of class names to:
            - "class_values": list of raw category strings that map to this class
            - "class_nr": numeric class ID used in the model

    Returns
    -------
    None
        Writes the processed layer to disk at `output_filename`, in layer "asset".

    Notes
    -----
    - The output layer includes:
        - "classification": numeric class ID
        - "instance": per-class instance index starting at 1
    - Geometries not in any known class are dropped.
    """
    # Initialize new columns
    asset_gdf["classification"] = 0
    asset_gdf["instance"] = 0

    # We'll collect all valid raw class values so we can filter at the end
    asset_classes: list[str] = []

    for _, classes in data_class.classifications.items():
        class_values = classes["class_values"]
        class_nr = classes["class_nr"]

        mask = asset_gdf["KLASSE"].isin(class_values)

        # Assign numeric class label and per-class instance index
        asset_gdf.loc[mask, "classification"] = class_nr
        asset_gdf.loc[mask, "instance"] = np.arange(mask.sum()) + 1

        asset_classes.extend(class_values)

    # Keep only rows that map to a known class
    converted = asset_gdf[asset_gdf["KLASSE"].isin(asset_classes)]

    converted.to_file(
        output_filename,
        layer="asset",
        driver="GPKG",
    )


def get_relevant_assets(
    asset_df_path: str,
    bounds_gdf: gpd.GeoDataFrame,
    bgt_outfile: str,
) -> None:
    """
    Extract and save relevant asset data for a single point cloud tile.

    Workflow:
    1. Compute the bounding box of the tile (`bounds_gdf`).
    2. Read the asset dataset.
    3. Spatially crop assets to that bounding box.
    4. Normalize classes / instances and save them as a single-layer GPKG.

    Parameters
    ----------
    asset_df_path : str
        Path to the (global) asset GeoPackage file.
    bounds_gdf : geopandas.GeoDataFrame
        Bounds of the current point cloud tile. Must contain at least one row;
        the first geometry is used to derive the bounding box.
    bgt_outfile : str
        Output path where the processed, single-layer asset file should be written.

    Returns
    -------
    None
        Writes a 'asset' layer to `bgt_outfile`.

    Raises
    ------
    Exception
        Any exception is not caught here. Callers should wrap if they want
        graceful degradation.
    """
    # Take the bounds of the first geometry in the tile bounds GeoDataFrame.
    bbox = bounds_gdf.iloc[0].geometry.bounds
    bgt_gdf = get_assets_from_tile(
        asset_filepath=asset_df_path,
        bounding_box=bbox,
    )

    if bgt_gdf is None or bgt_gdf.empty:
        logger.info(
            "No relevant assets found for tile. Output will be empty: %s",
            bgt_outfile,
        )
        # Even if empty, still produce a file so downstream code doesn't explode.
        empty_asset_df = gpd.GeoDataFrame(
            columns=["geometry", "classification", "instance", "KLASSE"],
            geometry=[],
            crs="EPSG:28992",
        )
        asset_dataclass = AssetClassifier()
        convert_asset_to_single_layer_gpkg(
            asset_gdf=empty_asset_df,
            output_filename=bgt_outfile,
            data_class=asset_dataclass,
        )
        return

    asset_dataclass = AssetClassifier()
    convert_asset_to_single_layer_gpkg(
        asset_gdf=bgt_gdf,
        output_filename=bgt_outfile,
        data_class=asset_dataclass,
    )


def get_assets_from_tile(
    asset_filepath: str,
    bounding_box: list[float],
) -> gpd.GeoDataFrame | None:
    """
    Load asset data and clip it to a given bounding box.

    The function:
    - Reads all assets from `asset_filepath`.
    - Builds a bounding geometry from the provided [minx, miny, maxx, maxy].
    - Intersects assets with that bounding geometry.
    - Returns only assets within this tile.

    Parameters
    ----------
    asset_filepath : str
        Path to the input asset data (GeoPackage).
    bounding_box : list[float]
        Bounding box coordinates in EPSG:28992, formatted as:
        [minx, miny, maxx, maxy].

    Returns
    -------
    geopandas.GeoDataFrame | None
        GeoDataFrame of assets intersecting the bounding box.
        Returns None if `bounding_box` is malformed (<4 values).

    Notes
    -----
    - Uses `gpd.overlay(..., how="intersection")`, so features that partially
      overlap the box are included.
    - Assumes the asset file is already in EPSG:28992 or compatible with it.
    """
    if len(bounding_box) < 4:
        logger.info(
            "The provided bounding box does not contain enough coordinates. "
            "Expected [minx, miny, maxx, maxy], got %s",
            bounding_box,
        )
        return None

    geodata = gpd.read_file(asset_filepath)

    # Construct a bbox polygon as a tiny GeoDataFrame
    # We build a LineString from (minx, miny) to (maxx, maxy) and then take
    # the envelope to get the full rectangle.
    bbox_geom = LineString(
        [bounding_box[0:2], bounding_box[2:]],
    ).envelope
    bbox_gdf = gpd.GeoDataFrame(
        data=gpd.GeoSeries(bbox_geom),
        columns=["geometry"],
        crs="EPSG:28992",
    )

    geodata_overlay = gpd.overlay(
        df1=geodata,
        df2=bbox_gdf,
        how="intersection",
    )

    return geodata_overlay
