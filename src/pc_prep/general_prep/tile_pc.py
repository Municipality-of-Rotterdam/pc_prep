import os
import math
import json
from pathlib import Path
import argparse
import datetime
from multiprocessing import Pool, cpu_count

import geopandas as gpd
from shapely.geometry import Polygon, LineString
from pyproj import CRS
import pdal
import laspy
from tqdm import tqdm

from pc_prep.logger import logger


# ---------------------------
# Utilities
# ---------------------------


def get_recording_specs(
    input_file: str,
) -> tuple[float, float, float, float, datetime.date]:
    """
    Read basic metadata from a LAS/LAZ header without loading all points.

    Extracts:
    - Bounding box of the dataset (minx, miny, maxx, maxy)
    - Recording date

    This is done using only the LAS/LAZ header, which is much faster than parsing
    the full point set.

    Parameters
    ----------
    input_file : str
        Path to the LAS/LAZ file.

    Returns
    -------
    tuple[float, float, float, float, datetime.date]
        (
            minx,
            miny,
            maxx,
            maxy,
            rec_date,
        )
        where `rec_date` is the acquisition date stored in the LAS header.

    Notes
    -----
    Assumes the LAS/LAZ file contains valid header metadata for mins, maxs,
    and date. If this is missing or malformed, laspy may raise.
    """
    with laspy.open(input_file) as fh:
        hdr = fh.header
        return hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1], hdr.date


def fmt(v: float, coord_precision: int = 0) -> str:
    """
    Format a coordinate with an optional decimal precision.

    If coord_precision == 0, rounds to nearest integer and returns that.
    Otherwise returns a string with `coord_precision` decimal places,
    and trims trailing 0s and any trailing '.'.

    Parameters
    ----------
    v : float
        Coordinate value.
    coord_precision : int, default 0
        Number of decimal places to keep.

    Returns
    -------
    str
        Formatted coordinate string.
    """
    if coord_precision <= 0:
        return str(int(round(v)))
    return f"{v:.{coord_precision}f}".rstrip("0").rstrip(".")


# ---------------------------
# Grid creation (stores filenames)
# ---------------------------


def create_grid_gpkg(
    input_file: str,
    gpkg_path: str,
    tile_size: float = 50.0,
    crs: str = "EPSG:28992",
    recording_tag: str = "drone",
) -> str:
    """
    Create a tiling grid over the spatial extent of a point cloud and save it as a GeoPackage.

    Each grid cell:
    - Is a polygon tile of size `tile_size` x `tile_size` (in CRS units).
    - Gets a 'FileName' attribute derived from the upper-left (UL) corner of the tile
      and a standard naming scheme that includes the recording date and tag.

    Example file naming convention:
        nl-rott-YYYYMMDD-<recording_tag>-laz/tile_UL_<ULx>_<ULy>.laz

    Parameters
    ----------
    input_file : str
        Path to the LAS/LAZ file used to determine overall bounds and recording date.
    gpkg_path : str
        Output path for the generated grid GeoPackage.
    tile_size : float, default 50.0
        Tile size in CRS units (meters in EPSG:28992).
    crs : str, default "EPSG:28992"
        CRS string to assign to the grid geometries.
    recording_tag : str, default "drone"
        Tag describing the capture source, appended to the directory name
        (e.g. "drone", "mls", etc.).

    Returns
    -------
    str
        The path to the written GeoPackage (`gpkg_path`).

    Notes
    -----
    - The GeoPackage will contain a polygon layer with columns:
        - id
        - FileName
        - geometry
    - The parent directory of `gpkg_path` will be created if it doesn't exist.
    """
    Path(gpkg_path).parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy, rec_date = get_recording_specs(input_file)
    rec_date_str = rec_date.strftime("%Y%m%d")
    t = float(tile_size)

    cols = int(math.ceil((maxx - minx) / t))
    rows = int(math.ceil((maxy - miny) / t))

    polys = []
    recs = []
    idx = 0

    for r in range(rows):
        yb = miny + r * t
        yt = min(yb + t, maxy)
        for c in range(cols):
            xl = minx + c * t
            xr = min(xl + t, maxx)

            # Upper-left corner of the tile
            ulx, uly = xl, yt

            filename = os.path.join(
                f"nl-rott-{rec_date_str}-{recording_tag}-laz",
                f"tile_UL_{fmt(ulx)}_{fmt(uly)}.laz",
            )

            polys.append(Polygon([(xl, yb), (xr, yb), (xr, yt), (xl, yt), (xl, yb)]))
            recs.append(
                {
                    "id": idx,
                    "FileName": filename,
                }
            )
            idx += 1

    gdf = gpd.GeoDataFrame(
        recs,
        geometry=polys,
        crs=CRS.from_user_input(crs),
    )
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info("Grid written to %s", gpkg_path)

    return gpkg_path


# ---------------------------
# Worker (single tile)
# ---------------------------


def tile_worker(
    task: tuple[str, tuple[float, float, float, float], str],
) -> tuple[str | None, int, str | None]:
    """
    Crop a point cloud to a bounding box and write it to disk using PDAL.

    This worker function is intended to be run in parallel via `multiprocessing.Pool`.
    It:
    - crops `input_file` to the given bounding box,
    - writes the cropped tile to `out_path`,
    - returns metadata about success and point count.

    Parameters
    ----------
    task : tuple[str, tuple[float, float, float, float], str]
        (
            input_file,
            (minx, miny, maxx, maxy),
            out_path,
        )

    Returns
    -------
    tuple[str | None, int, str | None]
        (
            output_path,
            n_points_written,
            error_message,
        )
        where:
        - output_path is None if the tile was empty or failed.
        - n_points_written is 0 if empty or failed.
        - error_message is None on success.

    Notes
    -----
    - Empty tiles are skipped and removed from disk immediately.
    - Any raised exception is caught and returned instead of propagated.
    """
    input_file, bounds, out_path = task
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        minx, miny, maxx, maxy = bounds

        reader = pdal.Reader(input_file)
        crop = pdal.Filter.crop(bounds=f"([{minx},{maxx}],[{miny},{maxy}])")
        writer = pdal.Writer.las(filename=out_path, forward="all")
        pipe = reader | crop | writer
        pipe.execute()

        # Count points written using PDAL's arrays
        npts = 0
        try:
            if pipe.arrays:
                npts = sum(len(buf) for buf in pipe.arrays)
        except Exception:
            # If array inspection fails, we silently fall back to 0.
            pass

        # If tile is empty, delete it and report it as skipped.
        if npts == 0:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            return None, 0, None

        return out_path, npts, None

    except Exception as e:
        # Clean up partial output on failure
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return None, 0, str(e)


# ---------------------------
# Multiprocess tiler
# ---------------------------


def tile_pointcloud_using_grid(
    input_file: str,
    grid_gpkg: str,
    out_dir: str,
    recording_tag: str = "drone",
    crs: str = "EPSG:28992",
    skip_existing: bool = True,
    processes: int | None = None,
    chunksize: int = 1,
    show_progress: bool = True,
    tqdm_desc: str = "Tiling",
) -> list[str]:
    """
    Tile a point cloud using a precomputed grid, in parallel.

    Workflow:
    1. Load the grid polygons from `grid_gpkg`.
    2. Intersect with the bounding box of `input_file` so we only keep tiles that
       overlap the actual point cloud area.
    3. For each relevant grid cell:
       - Compute bounds.
       - Compute an output tile path in `out_dir` using the same naming scheme as
         `create_grid_gpkg`.
       - Optionally skip tiling if the output already exists.
    4. Use multiprocessing to crop each tile via PDAL (see `tile_worker`).
    5. Drop empty tiles and update the grid metadata accordingly.
    6. Save updated grid metadata (`tile_metadata.gpkg`) under the output directory.

    Parameters
    ----------
    input_file : str
        Path to the source LAS/LAZ file.
    grid_gpkg : str
        Path to the grid GeoPackage (e.g. from `create_grid_gpkg`).
        Must have a column "FileName".
    out_dir : str
        Output directory where tiled LAS/LAZ files will be written.
    recording_tag : str, default "drone"
        Tag describing acquisition type; used in output subdirectory naming.
    crs : str, default "EPSG:28992"
        Expected CRS of the grid. The grid will be validated against this.
    skip_existing : bool, default True
        If True, skip re-creating tiles that already exist on disk. Those tiles
        will still be included in the returned list.
    processes : int | None, default None
        Number of worker processes. Defaults to `cpu_count()` if None.
    chunksize : int, default 1
        Chunk size for `imap_unordered`. Larger values can improve throughput
        on high-latency I/O.
    show_progress : bool, default True
        Whether to wrap the pool iterator in a tqdm progress bar.
    tqdm_desc : str, default "Tiling"
        Description label for tqdm.

    Returns
    -------
    list[str]
        Sorted unique list of paths to successfully written (or reused) tiles.

    Raises
    ------
    RuntimeError
        If the grid is empty after intersecting with the point cloud bounds,
        or if the CRS does not match the expected `crs`.

    Notes
    -----
    - The function writes an updated `tile_metadata.gpkg` that contains only the
      successfully written tiles.
    - Tiles with 0 points are deleted and not included in the output list.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    grid_raw = gpd.read_file(grid_gpkg)

    # Keep only grid cells that overlap the point cloud bounds.
    minx, miny, maxx, maxy, rec_date = get_recording_specs(input_file)
    rec_date_str = rec_date.strftime("%Y%m%d")
    rec_name = f"nl-rott-{rec_date_str}-{recording_tag}-laz"

    bbox_gdf = gpd.GeoDataFrame(
        columns=["geometry"],
        data=gpd.GeoSeries(LineString([(minx, miny), (maxx, maxy)]).envelope),
        crs="EPSG:28992",
    )

    grid = gpd.overlay(df1=grid_raw, df2=bbox_gdf, how="intersection")

    if grid.empty:
        raise RuntimeError("Grid is empty after intersecting with point cloud bounds.")
    if grid.crs is None or CRS.from_user_input(grid.crs) != CRS.from_user_input(crs):
        raise RuntimeError(f"Grid CRS must be {crs}, got {grid.crs}")

    assert (
        "FileName" in grid.columns
    ), "The grid geopackage should have a column called 'FileName'."

    # Build task list for multiprocessing, and collect tiles that already exist.
    tasks: list[tuple[str, tuple[float, float, float, float], str]] = []
    written_paths: list[str] = []

    for ind, row in grid.iterrows():
        bx_minx, bx_miny, bx_maxx, bx_maxy = row.geometry.bounds

        out_path = os.path.join(
            out_dir,
            rec_name,
            os.path.basename(row["FileName"]),
        )

        if skip_existing and os.path.exists(out_path):
            # Already there — assume valid and just keep it.
            grid.loc[ind, "FileName"] = os.path.join(
                rec_name,
                os.path.basename(row["FileName"]),
            )
            written_paths.append(out_path)
            continue

        tasks.append((input_file, (bx_minx, bx_miny, bx_maxx, bx_maxy), out_path))

    # If everything existed already, no parallel work needed.
    if not tasks:
        return sorted(set(written_paths))

    nproc = processes or cpu_count() or 1
    logger.info(
        "Tiling %d cells with %d processes (chunksize=%d)",
        len(tasks),
        nproc,
        chunksize,
    )
    logger.info("Writing tiles to %s", out_dir)

    with Pool(processes=nproc) as pool:
        iterator = pool.imap_unordered(tile_worker, tasks, chunksize=chunksize)
        itr = (
            tqdm(iterator, total=len(tasks), desc=tqdm_desc)
            if show_progress
            else iterator
        )

        for output_path, npts, err in itr:
            if err:
                logger.warning("Tile failed: %s", err)
                continue
            if output_path is not None and npts > 0:
                written_paths.append(output_path)

        # Remove rows for tiles that didn't get written
        for ind, row in grid.iterrows():
            path = os.path.join(
                out_dir,
                rec_name,
                os.path.basename(row["FileName"]),
            )
            if not os.path.exists(path):
                grid.drop(index=ind, inplace=True)

    grid_outfname = os.path.join(out_dir, rec_name, "tile_metadata.gpkg")
    logger.info("Writing updated grid to %s", grid_outfname)
    grid.to_file(grid_outfname, driver="GPKG")

    return sorted(set(written_paths))


# ---------------------------
# CLI
# ---------------------------


def build_parser() -> argparse.ArgumentParser:
    """
    Build an argument parser for the tiling/grid CLI.

    Subcommands
    -----------
    grid
        Create a tile grid GeoPackage over the bounds of an input LAS/LAZ file.
    tile
        Tile a LAS/LAZ file into many smaller LAS/LAZ files using a grid.
    pc
        Do both in sequence: generate grid, then tile.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with subcommands.
    """
    p = argparse.ArgumentParser(
        description="Point cloud grid + tiler (PDAL/GeoPandas).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # grid subcommand
    pg = sub.add_parser(
        "grid",
        help="Create a grid GeoPackage over input LAS/LAZ bounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pg.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Input LAS/LAZ file used to detect bounds.",
    )
    pg.add_argument(
        "-g",
        "--gpkg-path",
        required=True,
        help="Output GPKG path.",
    )
    pg.add_argument(
        "--tile-size",
        type=float,
        default=50.0,
        help="Grid tile size in input CRS units.",
    )
    pg.add_argument(
        "--crs",
        default="EPSG:28992",
        help="CRS of grid (and expected CRS of data).",
    )
    pg.add_argument(
        "--recording-tag",
        default="drone",
        help="Recording tag for output directory name.",
    )

    # tile subcommand
    pt = sub.add_parser(
        "tile",
        help="Tile a point cloud using a grid GeoPackage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pt.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Input LAS/LAZ file (or list via PDAL).",
    )
    pt.add_argument(
        "-g",
        "--grid-gpkg",
        required=True,
        help="Grid GeoPackage path.",
    )
    pt.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Output tiles directory.",
    )
    pt.add_argument(
        "--recording-tag",
        default="drone",
        help="Recording tag for output directory name.",
    )
    pt.add_argument(
        "--crs",
        default="EPSG:28992",
        help="Expected CRS of the grid.",
    )
    pt.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing output tiles.",
    )
    pt.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker processes (default: CPU count).",
    )
    pt.add_argument(
        "--chunksize",
        type=int,
        default=6,
        help="Work chunk size for imap_unordered.",
    )

    # pc (pipeline) subcommand: grid + tile in one go
    pc = sub.add_parser(
        "pc",
        help="Create grid and tile in one step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pc.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Input LAS/LAZ file used to detect bounds.",
    )
    pc.add_argument(
        "-g",
        "--gpkg-path",
        required=True,
        help="Output GPKG path.",
    )
    pc.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Output tiles directory.",
    )
    pc.add_argument(
        "--tile-size",
        type=float,
        default=50.0,
        help="Grid tile size in input CRS units.",
    )
    pc.add_argument(
        "--crs",
        default="EPSG:28992",
        help="CRS of grid (and expected CRS of data).",
    )
    pc.add_argument(
        "--recording-tag",
        default="drone",
        help="Recording tag for output directory name.",
    )
    pc.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing output tiles.",
    )
    pc.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker processes (default: CPU count).",
    )
    pc.add_argument(
        "--chunksize",
        type=int,
        default=6,
        help="Work chunk size for imap_unordered.",
    )

    return p


def main() -> None:
    """
    CLI entry point for grid creation and tiling.

    Runs one of:
    - grid: create a tile grid GeoPackage
    - tile: tile a point cloud using an existing grid
    - pc: create grid and then tile immediately

    Prints summary info for tiling runs as JSON.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "grid":
        create_grid_gpkg(
            input_file=args.input_file,
            gpkg_path=args.gpkg_path,
            tile_size=args.tile_size,
            crs=args.crs,
            recording_tag=args.recording_tag,
        )

    elif args.cmd == "tile":
        tiles = tile_pointcloud_using_grid(
            input_file=args.input_file,
            grid_gpkg=args.grid_gpkg,
            out_dir=args.out_dir,
            recording_tag=args.recording_tag,
            crs=args.crs,
            skip_existing=not args.no_skip_existing,
            processes=args.processes,
            chunksize=args.chunksize,
        )
        logger.info(json.dumps({"count": len(tiles), "tiles": tiles}, indent=2))

    elif args.cmd == "pc":
        # Step 1: make grid
        grid_path = create_grid_gpkg(
            input_file=args.input_file,
            gpkg_path=args.gpkg_path,
            tile_size=args.tile_size,
            crs=args.crs,
            recording_tag=args.recording_tag,
        )

        # Step 2: tile using that grid
        tiles = tile_pointcloud_using_grid(
            input_file=args.input_file,
            grid_gpkg=grid_path,
            out_dir=args.out_dir,
            recording_tag=args.recording_tag,
            crs=args.crs,
            skip_existing=not args.no_skip_existing,
            processes=args.processes,
            chunksize=args.chunksize,
        )
        logger.info(json.dumps({"count": len(tiles), "tiles": tiles}, indent=2))


if __name__ == "__main__":
    main()
