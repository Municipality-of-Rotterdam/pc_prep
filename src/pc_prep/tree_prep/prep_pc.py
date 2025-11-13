import argparse
import json
import multiprocessing
import os
from functools import partial

from tqdm import tqdm

from pc_prep.logger import logger
from pc_prep.general_prep.convert_pc_to_img import (
    calculate_point_coords_and_labels,
    convert_pc_to_img,
    save_image_prompt,
)
from pc_prep.general_prep.utils import setup_img_paths
from pc_prep.tree_prep.metadata_handler import prepare_pc_paths, get_collection_info
from pc_prep.tree_prep.preprocess_pc import preprocess_pc
from pc_prep.tree_prep.utils import (
    tree_pc_ready_for_img_conversion,
    setup_pc_paths,
    prepare_output_paths,
)


def process_single_pc(
    pc_path: str, args: argparse.Namespace
) -> dict[str, dict[str, dict[str, str] | str]] | None:
    """
    Process a single point cloud tile:
    - Preprocess the point cloud and extract relevant subsets (trees, non-ground points, pavements).
    - Rasterize to an image.
    - Generate prompt metadata (coords of sampled points).
    - Produce a structured record of output paths for downstream use.

    This function is designed to run in parallel using multiprocessing.

    Parameters
    ----------
    pc_path : str
        Absolute path to the raw point cloud file to process.
    args : argparse.Namespace
        Parsed command-line arguments containing configuration and output locations.

    Returns
    -------
    dict or None
        A nested dictionary with three top-level keys:
        - "pc":  { <tile_id>: { "pc_path": <...>, "tree_geom_path": <...> } }
        - "img": { <tile_id>: { "img_path": <...>, "prompt_path": <...> } }
        - "bgt": { <tile_id>: <bgt_pavements_path> }
        The dictionary is later merged into global metadata.
        Returns None if the tile is deemed not suitable for image conversion
        (e.g. no usable trees / insufficient non-ground points).

    Notes
    -----
    Skips heavy work if all expected outputs already exist on disk and
    --overwrite is False. In that case we still return metadata paths.
    """
    logger.info("Processing file: %s", pc_path)

    logger.info("Setting up file paths for point cloud and image products.")
    pc_code = get_collection_info(pc_filename=pc_path)
    basename = os.path.splitext(os.path.basename(pc_path))[0]

    pc_output_path, tree_gpkg_path, bgt_pavements_path = setup_pc_paths(
        pc_path=pc_path,
        pc_code=pc_code,
        pc_out_dir=args.pc_dir,
        bgt_out_dir=args.bgt_dir,
    )

    image_path, prompt_img_path, prompt_path = setup_img_paths(
        output_dir=args.img_dir,
        pc_code=pc_code,
        basename=basename,
    )

    expected_outputs = [
        pc_output_path,
        tree_gpkg_path,
        bgt_pavements_path,
        image_path,
        prompt_path,
    ]

    needs_processing = args.overwrite or not all(
        os.path.exists(path) for path in expected_outputs
    )

    if needs_processing:
        trees_in_laz, non_ground_points = preprocess_pc(
            pc_path=pc_path,
            pc_code=pc_code,
            pc_metadata_df_path=args.pc_raw_metadata,
            tree_df_path=args.reference_trees_path,
            bgt_df_path=args.bgt_pavements_raw,
            pc_output_path=pc_output_path,
            tree_gpkg_path=tree_gpkg_path,
            bgt_pavements_path=bgt_pavements_path,
        )

        # If there's not enough usable data to make an image, skip this tile entirely.
        if not tree_pc_ready_for_img_conversion(
            trees_in_pc=trees_in_laz,
            pc_points=non_ground_points,
        ):
            logger.info("Tile %s not suitable for image conversion. Skipping.", pc_path)
            return None

        point_coords, _ = calculate_point_coords_and_labels(
            filtered_points=non_ground_points,
            resolution=args.resolution,
            trees_in_laz=trees_in_laz,
            prompt_path=prompt_path,
        )

        convert_pc_to_img(
            image_path=image_path,
            points=non_ground_points,
            resolution=args.resolution,
        )

        # If debug is enabled, create a visualization of sampled prompt points.
        if args.debug:
            save_image_prompt(
                image_path=image_path,
                point_coords=point_coords,
                output_path=prompt_img_path,
            )
    else:
        # overwrite=False and everything already appears generated.
        logger.info(
            "All outputs exist and overwrite is %s. Skipping heavy processing.",
            args.overwrite,
        )

    (
        pc_path_basename,
        pc_output_path_basename,
        tree_gpkg_path_basename,
        bgt_pavements_path_basename,
        image_path_basename,
        prompt_path_basename,
    ) = prepare_output_paths(
        args=args,
        pc_path=pc_path,
        pc_output_path=pc_output_path,
        tree_gpkg_path=tree_gpkg_path,
        bgt_pavements_path=bgt_pavements_path,
        image_path=image_path,
        prompt_path=prompt_path,
    )

    return {
        "pc": {
            pc_path_basename: {
                "pc_path": pc_output_path_basename,
                "tree_geom_path": tree_gpkg_path_basename,
            }
        },
        "img": {
            pc_path_basename: {
                "img_path": image_path_basename,
                "prompt_path": prompt_path_basename,
            }
        },
        "bgt": {
            pc_path_basename: bgt_pavements_path_basename,
        },
    }


def prep_pc(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, str]]:
    """
    Batch-preprocess a collection of point cloud tiles and generate metadata
    for downstream tree analysis / segmentation work.

    The workflow is:
    1. Resolve which point cloud tiles to process with `prepare_pc_paths`.
    2. Process tiles in parallel using a worker pool.
    3. Aggregate per-tile outputs into three metadata dicts:
       - point cloud outputs (pc_metadata)
       - raster image outputs (img_metadata)
       - BGT pavement outputs (bgt_metadata)
    4. Write those dicts to JSON files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Must include:
        - tree_df_path
        - reference_trees_path
        - pc_raw
        - pc_raw_metadata
        - bgt_pavements_raw
        - pc_dir, img_dir, bgt_dir
        - pc_metadata, img_metadata, bgt_metadata
        - resolution
        - num_workers (optional)
        - overwrite (bool)
        - debug (bool)

    Returns
    -------
    (pc_metadata, img_metadata, bgt_metadata) : tuple
        pc_metadata : dict[str, dict[str, str]]
            Map of tile_id -> {"pc_path": <...>, "tree_geom_path": <...>}
        img_metadata : dict[str, dict[str, str]]
            Map of tile_id -> {"img_path": <...>, "prompt_path": <...>}
        bgt_metadata : dict[str, str]
            Map of tile_id -> <bgt_pavements_path>

    Side Effects
    ------------
    - Creates parent directories for metadata JSON outputs if needed.
    - Writes the three metadata dicts to disk as JSON.
    - Logs progress and status information.
    """
    pc_paths = prepare_pc_paths(
        tree_df_path=args.tree_df_path,
        pc_metadata_df_path=args.pc_raw_metadata,
        mounted_pc_path=args.pc_raw,
    )

    # Make sure output dirs for metadata JSONs exist.
    os.makedirs(os.path.dirname(args.img_metadata), exist_ok=True)
    os.makedirs(os.path.dirname(args.pc_metadata), exist_ok=True)
    os.makedirs(os.path.dirname(args.bgt_metadata), exist_ok=True)

    pc_metadata: dict[str, dict[str, str]] = {}
    img_metadata: dict[str, dict[str, str]] = {}
    bgt_metadata: dict[str, str] = {}

    process_func = partial(process_single_pc, args=args)

    cpu_total = multiprocessing.cpu_count()
    # Default to (cpu_count - 1), but clamp to at least 1.
    default_workers = max(cpu_total - 1, 1)

    if args.num_workers is None:
        num_workers = default_workers
    elif 0 < args.num_workers <= default_workers:
        num_workers = args.num_workers
    else:
        logger.error(
            "Invalid --num_workers value: %s. Falling back to %s.",
            args.num_workers,
            default_workers,
        )
        num_workers = default_workers

    logger.info(
        "Processing %d tiles with %d parallel workers.", len(pc_paths), num_workers
    )

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_func, pc_paths),
            total=len(pc_paths),
            desc="Processing Point Clouds",
        ):
            if result is None:
                continue

            pc_metadata.update(result["pc"])
            img_metadata.update(result["img"])
            bgt_metadata.update(result["bgt"])

    # Persist aggregated metadata
    with open(args.pc_metadata, "w") as f:
        json.dump(pc_metadata, f, indent=4)
    logger.info("Wrote point cloud metadata to %s", args.pc_metadata)

    with open(args.img_metadata, "w") as f:
        json.dump(img_metadata, f, indent=4)
    logger.info("Wrote image metadata to %s", args.img_metadata)

    with open(args.bgt_metadata, "w") as f:
        json.dump(bgt_metadata, f, indent=4)
    logger.info("Wrote BGT pavements metadata to %s", args.bgt_metadata)

    return pc_metadata, img_metadata, bgt_metadata


def configure_arg_parser() -> argparse.Namespace:
    """
    Build and parse the command-line interface for the preprocessing pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes used throughout the pipeline.

    The most important arguments:

    Inputs
    ------
    --tree_df_path : str
        Path to the DataFrame (.pkl) containing the trees we want to process.
    --reference_trees_path : str
        Path to the canonical/reference tree dataset (.pkl).
    --pc_raw : str
        Directory containing the raw point cloud tiles.
    --pc_raw_metadata : str
        Path to the point cloud metadata index (.gpkg).
        Used to locate and interpret tiles.
    --bgt_pavements_raw : str
        Path to the raw BGT pavements layer (.gpkg).

    Outputs
    -------
    --pc_dir : str
        Directory where processed point cloud outputs will be written.
    --img_dir : str
        Directory where raster images (TIFFs etc.) and prompt files will be written.
    --bgt_dir : str
        Directory where processed BGT pavement geometries will be written.
    --pc_metadata : str
        Output .json file storing per-tile point cloud products.
    --img_metadata : str
        Output .json file storing per-tile image products.
    --bgt_metadata : str
        Output .json file storing per-tile BGT pavements products.

    Processing Options
    ------------------
    --resolution : float
        Pixel resolution (in meters per pixel) for the exported TIFF.
        Point cloud tiles are 50m x 50m. With resolution=0.05,
        you get ~1000 x 1000 px images, which fits SAM's 1024x1024 encoder input.
    --num_workers : int
        Number of parallel workers for multiprocessing. Defaults to (CPU cores - 1).
    --overwrite : bool
        If True, regenerate outputs even if they already exist.
    --debug : bool
        If True, write extra debug artifacts such as prompt overlay images.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess point cloud tiles (trees / pavements) and rasterize to images."
    )

    # Inputs
    parser.add_argument(
        "--tree_df_path",
        type=str,
        required=True,
        help="Path to the .pkl of trees to process.",
    )
    parser.add_argument(
        "--reference_trees_path",
        type=str,
        required=True,
        help="Path to the reference trees .pkl.",
    )
    parser.add_argument(
        "--pc_raw",
        type=str,
        required=True,
        help="Directory containing raw point cloud tiles.",
    )
    parser.add_argument(
        "--pc_raw_metadata",
        type=str,
        required=True,
        help="Path to the point cloud metadata GeoDataFrame (.gpkg).",
    )
    parser.add_argument(
        "--bgt_pavements_raw",
        type=str,
        required=True,
        help="Path to the raw BGT pavements .gpkg.",
    )

    # Output dirs / files
    parser.add_argument(
        "--pc_dir",
        type=str,
        required=True,
        help="Output directory for processed point cloud data.",
    )
    parser.add_argument(
        "--pc_metadata",
        type=str,
        required=True,
        help="Output .json path for processed point cloud metadata.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Output directory for generated image tiles and prompts.",
    )
    parser.add_argument(
        "--img_metadata",
        type=str,
        required=True,
        help="Output .json path for processed image metadata.",
    )
    parser.add_argument(
        "--bgt_dir",
        type=str,
        required=True,
        help="Output directory for processed BGT pavements geometries.",
    )
    parser.add_argument(
        "--bgt_metadata",
        type=str,
        required=True,
        help="Output .json path for processed BGT pavements metadata.",
    )

    # Runtime / behavior
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help=(
            "Raster resolution (meters per pixel) for exported TIFFs. "
            "Tiles are 50m x 50m. With 0.05 m/px you get ~1000 x 1000 px, "
            "close to SAM's 1024 x 1024 encoder size."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel workers to use. Default: CPU cores - 1.",
    )
    parser.add_argument(
        "--overwrite",
        type=lambda s: s.lower() == "true",
        default=False,
        help="If 'true', overwrite existing outputs on disk.",
    )
    parser.add_argument(
        "--debug",
        type=lambda s: s.lower() == "true",
        default=False,
        help="If 'true', write debug prompt overlay images.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for CLI usage.

    Parses CLI args and runs the preprocessing pipeline.
    """
    args = configure_arg_parser()
    prep_pc(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
