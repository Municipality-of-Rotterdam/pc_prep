import os
import geopandas as gpd
import numpy as np
import pandas as pd
import laspy
from unittest import mock
from shapely.geometry import box, Point
from pc_prep.general_prep.utils import (
    enlarge_pc_tile,
    filter_pointcloud_to_aoi,
    convert_df_to_gdf,
    get_pc_bounds,
    read_laz_points,
)


def test_convert_df_to_gdf():
    """Test converting a DataFrame with WKT geometries to a GeoDataFrame."""
    df = pd.DataFrame({"ID": [1], "WKT_GEOMETRIE": ["POINT(0 0)"]})
    gdf = convert_df_to_gdf(df)
    expected_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)], "ID": [1]}, crs="EPSG:28992")
    assert gdf.equals(expected_gdf)

def test_get_pc_bounds():
    """Test getting the bounds of a point cloud."""
    pc_path = "test.laz"
    mls_metadata_gdf = gpd.GeoDataFrame({"FileName": ["test.laz"], "geometry": [box(0, 0, 10, 10)]}, crs="EPSG:28992")
    bounds_gdf = get_pc_bounds(pc_path, mls_metadata_gdf)
    expected_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs="EPSG:28992")
    assert bounds_gdf.equals(expected_gdf)

@mock.patch("os.makedirs")
@mock.patch("os.path.getsize", side_effect=lambda path: 600)  # Mock all files as valid (>510 bytes)
@mock.patch("pc_prep.general_prep.utils.merge_point_clouds_with_pdal_pipeline")
@mock.patch("pc_prep.general_prep.utils.filter_pointcloud_to_aoi")
def test_enlarge_pc_tile(mock_filter, mock_merge, mock_getsize, mock_makedirs):
    """Test enlarging a point cloud tile."""
    pc_path = "test.laz"
    outfname = "output.laz"
    tiles_aoi = gpd.GeoDataFrame({"FileName": ["tile1.laz", "tile2.laz"]})
    
    enlarge_pc_tile(pc_path, tiles_aoi, outfname)
    
    mock_makedirs.assert_called_once()
    mock_merge.assert_called_once()
    mock_filter.assert_called_once()
    # Assert that getsize was called twice (once for each file)
    assert mock_getsize.call_count == 2    
    
@mock.patch("os.makedirs")
@mock.patch("os.path.getsize", side_effect=lambda path: 500)  # Mock all files as empty
@mock.patch("pc_prep.general_prep.utils.logger.error")
@mock.patch("pc_prep.general_prep.utils.merge_point_clouds_with_pdal_pipeline")
@mock.patch("pc_prep.general_prep.utils.filter_pointcloud_to_aoi")
def test_enlarge_pc_tile_no_valid_files(mock_filter, mock_merge, mock_logger, mock_getsize, mock_makedirs):
    """Test enlarge_pc_tile when all input .laz files are empty or invalid."""
    pc_path = "test.laz"
    outfname = "output.laz"
    tiles_aoi = gpd.GeoDataFrame({"FileName": ["tile1.laz", "tile2.laz"]})

    result = enlarge_pc_tile(pc_path, tiles_aoi, outfname)

    # Expect no processing to occur and an error to be logged
    assert result is False
    mock_logger.assert_called_once_with("No valid point cloud tiles found for %s", pc_path)
    mock_merge.assert_not_called()
    mock_filter.assert_not_called()


@mock.patch("os.makedirs")
@mock.patch("os.path.getsize", side_effect=lambda path: 600 if "tile1" in path else 400)  # Only one valid file
@mock.patch("pc_prep.general_prep.utils.merge_point_clouds_with_pdal_pipeline")
@mock.patch("pc_prep.general_prep.utils.filter_pointcloud_to_aoi")
def test_enlarge_pc_tile_partial_valid_files(mock_filter, mock_merge, mock_getsize, mock_makedirs):
    """Test enlarge_pc_tile when some input .laz files are valid and others are empty."""
    pc_path = "test.laz"
    outfname = "output.laz"
    tiles_aoi = gpd.GeoDataFrame({"FileName": ["tile1.laz", "tile2.laz"]})
    mock_filter.return_value = True

    result = enlarge_pc_tile(pc_path, tiles_aoi, outfname)

    # Expect processing to occur since at least one valid file exists
    assert result is True
    mock_merge.assert_called_once()
    mock_filter.assert_called_once()


@mock.patch("laspy.read")
def test_read_laz_points(mock_laspy):
    """Test reading a LAS file into a NumPy array."""
    mock_las = mock.MagicMock()
    mock_las.points.x = np.array([1.0, 2.0])
    mock_las.points.y = np.array([3.0, 4.0])
    mock_las.points.z = np.array([5.0, 6.0])
    mock_las.points.red = np.array([255, 128])
    mock_las.points.green = np.array([128, 255])
    mock_las.points.blue = np.array([64, 192])
    
    mock_laspy.return_value = mock_las  # Ensure laspy.read() returns the mock
    
    points = read_laz_points("test.laz")
    
    assert points.shape == (2, 6)
    assert np.all(points[:, :3] == np.array([[1, 3, 5], [2, 4, 6]]))

    # Adjust tolerance to avoid floating-point precision issues
    assert np.allclose(points[:, 3:], np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.75]]), atol=1e-2)
    

def test_filter_pointcloud_to_aoi(mock_las_file):
    """Tests filtering a point cloud file using actual LAS data and mock AOI."""
    input_file, output_file = mock_las_file

    # Define mock AOI (bounding box from 5 to 25)
    tiles_aoi = gpd.GeoDataFrame(geometry=[box(5, 5, 25, 25)], crs="EPSG:28992")

    # Run the function
    result = filter_pointcloud_to_aoi(input_file=str(input_file), tiles_aoi=tiles_aoi, output_file=str(output_file), min_points=2)

    # Check if function returns False, as it should because of too little points in pointcloud.
    assert result

    # Verify that only the points inside the AOI were written
    filtered_las = laspy.read(output_file)
    assert len(filtered_las.x) == 2  # Only 10 & 20 remain
    assert np.all(filtered_las.x == np.array([10, 20]))
    assert np.all(filtered_las.y == np.array([10, 20]))

    # Ensure the input file was deleted
    assert not os.path.exists(input_file)
