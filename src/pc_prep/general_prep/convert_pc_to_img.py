import json

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import cv2
import geopandas as gpd
from numba import njit
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from pc_prep.logger import logger


def calculate_point_coords_and_labels(
    filtered_points: np.ndarray,
    resolution: float,
    trees_in_laz: gpd.GeoDataFrame,
    prompt_path: str,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Generate per-tree click prompts for downstream segmentation models.

    For each tree point geometry in `trees_in_laz`, this function:
    - Converts its (x, y) coordinates from world space into pixel space
      matching the raster that will be produced from the same point cloud.
    - Marks each tree as a positive label (1).
    - Writes both to disk in a JSON file for later use by SAM-like models.

    Pixel coordinates are defined in the same reference frame as the raster image
    we generate in `convert_pc_to_img`, i.e.:
    - x pixel = (geom.x - pc_minx) / resolution
    - y pixel = (pc_maxy - geom.y) / resolution

    Parameters
    ----------
    filtered_points : numpy.ndarray
        Array of point cloud points after filtering (e.g. non-ground points).
        Expected columns: [X, Y, Z, R, G, B] or similar. Only X and Y are used here.
    resolution : float
        Resolution of the output raster in map units per pixel (e.g. meters/pixel).
    trees_in_laz : geopandas.GeoDataFrame
        Trees that lie within this tile's bounds. Must have Point geometries.
    prompt_path : str
        Output path for a JSON file containing:
        {
            "point_coords": [[x_px, y_px], ...],
            "point_labels": [[1], [1], ...]
        }

    Returns
    -------
    tuple[list[list[int]], list[list[int]]]
        (
            point_coords,
            point_labels,
        )
        where:
        - point_coords is a list of [x_px, y_px] pixel coordinates (ints).
        - point_labels is a list of [1] for each point (all positives).

    Side Effects
    ------------
    Writes the above structure to `prompt_path` as JSON.
    """
    pc_minx = np.min(filtered_points[:, 0])
    pc_maxy = np.max(filtered_points[:, 1])

    point_coords = (
        trees_in_laz["geometry"]
        .apply(
            lambda geom: [
                int((geom.x - pc_minx) / resolution),
                int((pc_maxy - geom.y) / resolution),
            ]
        )
        .to_list()
    )

    point_labels = [[1] for _ in point_coords]

    output_dict: dict[str, list[list[int]]] = {
        "point_coords": point_coords,
        "point_labels": point_labels,
    }

    with open(prompt_path, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    logger.info("Point coords written to %s", prompt_path)

    return point_coords, point_labels


@njit  # type: ignore
def cloud_to_image(
    points: np.ndarray,
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    resolution: float,
) -> np.ndarray:
    """
    Rasterize a point cloud into a top-down RGB image.

    Each point contributes color to exactly one pixel. If multiple points land
    in the same pixel, the point with the highest Z value wins (i.e. we take
    the top-most return).

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) or (N, 6):
        - (X, Y, Z) if shape is (N, 3). Color defaults to white [255,255,255].
        - (X, Y, Z, R, G, B) if shape is (N, 6). RGB are assumed in 0-255 range.
    minx : float
        Minimum X of the point cloud extent.
    maxx : float
        Maximum X of the point cloud extent.
    miny : float
        Minimum Y of the point cloud extent.
    maxy : float
        Maximum Y of the point cloud extent.
    resolution : float
        Pixel size in the same units as X/Y (e.g. meters per pixel).

    Returns
    -------
    numpy.ndarray
        An image array of shape (H, W, 3), dtype uint8.
        Pixel (row, col) corresponds to projected (X, Y) in top-down space.

    Notes
    -----
    - The y-axis is flipped such that higher Y in world space maps to smaller
      row indices (i.e. standard north-up raster convention).
    - Empty input returns an all-zero (black) image of the computed size.
    """
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1

    image = np.zeros((height, width, 3), dtype=np.uint8)

    if len(points) == 0:
        return image

    # If RGB not supplied, assume white.
    if points.shape[1] == 3:
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
    else:
        colors = points[:, -3:].astype(np.uint8)

    x = (points[:, 0] - minx) / resolution
    y = (maxy - points[:, 1]) / resolution
    z = points[:, 2]

    pixel_x = np.floor(x).astype(np.int32)
    pixel_y = np.floor(y).astype(np.int32)

    # Keep the highest point for each pixel (Z max)
    unique_pixels: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for i in range(len(pixel_x)):
        coord = (pixel_x[i], pixel_y[i])
        if coord in unique_pixels:
            if z[i] > unique_pixels[coord][0]:
                unique_pixels[coord] = (z[i], colors[i])
        else:
            unique_pixels[coord] = (z[i], colors[i])

    for (px, py), (_, color) in unique_pixels.items():
        image[py, px] = color

    return image


def show_points(
    coords: np.ndarray,
    ax: Axes,
    marker_size: int = 550,
) -> None:
    """
    Overlay point prompts (e.g. tree click hints) on an axes.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (N, 2) containing pixel coordinates [[x_px, y_px], ...].
    ax : matplotlib.axes.Axes
        Axes to draw on (typically `plt.gca()` of the raster image figure).
    marker_size : int, default 550
        Size of the star marker for visualization.

    Returns
    -------
    None
    """
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def save_image_prompt(
    image_path: str,
    point_coords: list[list[int]],
    output_path: str,
) -> None:
    """
    Save a debug visualization: raster image + tree prompt points.

    This is mainly for inspection. It reads the raster image that was written
    by `convert_pc_to_img`, draws the list of prompt coordinates on top,
    and writes out a PNG (or whatever extension `output_path` has).

    Parameters
    ----------
    image_path : str
        Path to the raster image (TIFF or similar). Will be read with OpenCV.
    point_coords : list[list[int]]
        List of [x_px, y_px] integer pixel coordinates of tree prompts.
    output_path : str
        Path to save the annotated image (e.g. .png).

    Returns
    -------
    None
    """
    plt.figure(figsize=(20, 20))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    show_points(np.array(point_coords), plt.gca())

    plt.savefig(output_path)


def convert_pc_to_img(
    image_path: str,
    points: np.ndarray,
    resolution: float,
) -> None:
    """
    Convert a point cloud tile to a geo-referenced, north-up RGB raster.

    Steps:
    1. Compute tile bounds (minx/maxx/miny/maxy) from the supplied points.
    2. Rasterize point cloud to a top-down color image using `cloud_to_image`.
    3. Save the raster to GeoTIFF with spatial reference and affine transform.

    Parameters
    ----------
    image_path : str
        Output file path for the GeoTIFF.
    points : numpy.ndarray
        Point cloud array of shape (N, 6) or (N, 3).
        Columns are interpreted as:
        [X, Y, Z, R, G, B] with RGB in [0,255], or just [X, Y, Z] (defaults to white).
        Must all be in the same projected CRS (expected: EPSG:28992).
    resolution : float
        Pixel size in map units (e.g. meters per pixel).

    Returns
    -------
    None
        Writes a 3-band GeoTIFF to `image_path`.

    Notes
    -----
    - The GeoTIFF is written with:
        CRS = EPSG:28992 (RD New)
        transform = Affine(resolution, 0, minx, 0, -resolution, maxy)
      which aligns pixel (0,0) at (minx, maxy) and increases row index southward.
    """
    minx = np.min(points[:, 0])
    maxx = np.max(points[:, 0])
    miny = np.min(points[:, 1])
    maxy = np.max(points[:, 1])

    logger.info("Generating raster image...")
    image = cloud_to_image(points, minx, maxx, miny, maxy, resolution)
    image = np.asarray(image).astype(np.uint8)

    logger.info("Saving raster image...")
    # Affine transform: maps pixel coordinates to projected coordinates.
    transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

    # Dutch RD New CRS.
    crs = CRS.from_epsg(28992)

    # Write RGB GeoTIFF (3 bands)
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=3,
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(image[:, :, i], i + 1)
