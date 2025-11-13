"""Custom CSF- and asset-extraction utilities for point cloud preprocessing."""

import argparse
import logging

import geopandas as gpd
import laspy
import numpy as np
import pdal

# PDAL's CSF filter marks ground points with classification value 2.
GROUND_POINTS_INDEX = 2


def add_asset_classification_to_pc(
    input_file: str | list[str],
    asset_file: str,
    csf_threshold: float = 0.5,
    csf_resolution: float = 0.2,
    csf_smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a PDAL pipeline on one or more LiDAR point cloud files to:
    - classify ground vs. non-ground points using CSF (Cloth Simulation Filter),
    - attach per-point asset metadata ("AssetType" and "AssetInstance") from a geospatial layer,
    - write results back to LAS/LAZ,
    - return split point arrays and their indices.

    The function:
    1. Builds and runs a PDAL pipeline:
       - Reads the LAS/LAZ file(s).
       - Applies `pdal.Filter.csf` to classify points.
       - Adds two new dimensions: AssetType and AssetInstance.
       - Optionally overlays those dimensions with per-point attributes from a vector
         dataset (e.g. BGT) if the `asset` layer is non-empty.
       - Writes enriched output back to the same `input_file` path.
    2. Re-reads the file with `laspy` to build an (N, 6) array: X, Y, Z, R, G, B
       with RGB scaled to [0, 1].
    3. Uses the PDAL-produced "Classification" values to split points into
       ground and non-ground.

    Parameters
    ----------
    input_file : str | list[str]
        Path to a single LAS/LAZ file, or a list of LAS/LAZ files, to process.
        Note: The pipeline currently writes to `input_file`, so this is effectively
        in-place enrichment.
    asset_file : str
        Path to a geospatial dataset (e.g. GeoPackage) that contains a layer
        named "asset". The layer must have:
        - "classification" (used to populate AssetType),
        - "instance" (used to populate AssetInstance).
        If the layer is empty, overlay is skipped.
    csf_threshold : float, default 0.5
        Threshold parameter for the PDAL CSF filter.
    csf_resolution : float, default 0.2
        Resolution parameter (cloth resolution) for CSF.
    csf_smooth : bool, default False
        Whether to apply smoothing in the CSF filter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (
            non_ground_points,
            ground_points,
            non_ground_indices,
            ground_indices,
        )

        Where:
        - non_ground_points : np.ndarray
            Array of shape (N_non_ground, 6) with columns [X, Y, Z, R, G, B].
            RGB values are normalized to [0, 1].
        - ground_points : np.ndarray
            Array of shape (N_ground, 6) with columns [X, Y, Z, R, G, B].
        - non_ground_indices : np.ndarray
            Indices of the non-ground points relative to the original point set.
        - ground_indices : np.ndarray
            Indices of the ground points relative to the original point set.

    Raises
    ------
    Exception
        Any exception during PDAL execution, I/O, or classification is caught,
        logged with `logging.exception`, and the function will return
        four empty arrays instead of raising.

    Notes
    -----
    - PDAL uses the "Classification" dimension. By convention here:
        classification == 2 → ground
        classification != 2 → non-ground
      Future work aims to rename this to "GroundClassification" and
      normalize to 0/1 instead of 2/non-2.
    - We currently read the data twice: once via PDAL, and again via laspy
      to get scaled XYZRGB. This could be optimized by reusing PDAL's
      `pipeline.arrays`, but it's explicit (and safe) as written.
    """
    try:
        # Build PDAL pipeline
        pipeline = pdal.Reader(input_file)
        pipeline |= pdal.Filter.csf(
            threshold=csf_threshold,
            resolution=csf_resolution,
            smooth=csf_smooth,
        )

        # Create empty dimensions so they exist even if overlay is skipped
        pipeline |= pdal.Filter.ferry(dimensions="=>AssetType")
        pipeline |= pdal.Filter.ferry(dimensions="=>AssetInstance")

        # If assets are available, add per-point attributes via overlay
        asset_layer = gpd.read_file(asset_file, layer="asset")
        if not asset_layer.empty:
            # AssetType = classification column
            pipeline |= pdal.Filter.overlay(
                dimension="AssetType",
                datasource=asset_file,
                column="classification",
                layer="asset",
            )
            # AssetInstance = unique instance column
            pipeline |= pdal.Filter.overlay(
                dimension="AssetInstance",
                datasource=asset_file,
                column="instance",
                layer="asset",
            )

        # Write enriched LAS/LAZ back out.
        # extra_dims="all" ensures new dimensions are preserved.
        pipeline |= pdal.Writer.las(filename=input_file, extra_dims="all")

        pipeline.execute()

        # Re-open with laspy to build an XYZRGB array.
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

        # Pull PDAL's classification output from the pipeline arrays.
        # `pipeline.arrays[0]` is a structured array containing per-point dims.
        point_data = pipeline.arrays[0]
        classifications = point_data["Classification"]

        # Separate ground / non-ground indices.
        ground_indices = np.where(classifications == GROUND_POINTS_INDEX)[0]
        non_ground_indices = np.where(classifications != GROUND_POINTS_INDEX)[0]

        non_ground_points = points[non_ground_indices, :]
        ground_points = points[ground_indices, :]

        return non_ground_points, ground_points, non_ground_indices, ground_indices

    except Exception:
        logging.exception("Error processing file(s): %s", input_file)
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )


def configure_arg_parser() -> argparse.Namespace:
    """
    Create and parse CLI arguments for running CSF + asset classification.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - input_file : str
            Path to the input LAS/LAZ file to classify.
        - asset_file : str
            Path to the geospatial asset dataset.
        - csf_threshold : float
            Threshold for the CSF filter.
    """
    parser = argparse.ArgumentParser(
        description="Apply CSF-based ground classification and overlay asset info using PDAL."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input LAS/LAZ file to process.",
    )
    parser.add_argument(
        "--asset_file",
        type=str,
        required=True,
        help="Path to the geospatial dataset (must contain 'asset' layer "
        "with 'classification' and 'instance' columns).",
    )
    parser.add_argument(
        "--csf_threshold",
        type=float,
        default=0.5,
        help="Threshold value for the CSF filter (default: 0.5).",
    )
    parser.add_argument(
        "--csf_resolution",
        type=float,
        default=0.2,
        help="Resolution (cloth resolution) for CSF (default: 0.2).",
    )
    parser.add_argument(
        "--csf_smooth",
        type=lambda s: s.lower() == "true",
        default=False,
        help="If 'true', enable CSF smoothing (default: false).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    CLI entry point.

    1. Parse args from the command line.
    2. Run add_asset_classification_to_pc with those args.

    Notes
    -----
    The function modifies the LAS/LAZ file in-place, then prints nothing.
    """
    args = configure_arg_parser()
    add_asset_classification_to_pc(
        input_file=args.input_file,
        asset_file=args.asset_file,
        csf_threshold=args.csf_threshold,
        csf_resolution=args.csf_resolution,
        csf_smooth=args.csf_smooth,
    )
