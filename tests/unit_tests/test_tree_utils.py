import os
from unittest import mock

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
from pc_prep.tree_prep.utils import (
    tree_pc_ready_for_img_conversion,
    determine_tree_area_of_interest,
    get_trees_from_tile,
)

from pc_prep.tree_prep.utils import setup_pc_paths

def test_setup_pc_paths():
    """Test setup_pc_paths function to ensure it creates correct paths."""
    pc_path = "data/pointcloud.laz"
    pc_code = "pc123"
    pc_out_dir = "pc_out"
    bgt_out_dir = "bgt_out"
    expected_basename = "pointcloud"
    expected_pc_local_dir = os.path.join(pc_out_dir, expected_basename, pc_code)
    expected_bgt_local_dir = os.path.join(bgt_out_dir, expected_basename, pc_code)
    expected_pc_output_path = os.path.join(expected_pc_local_dir, "pointcloud.laz")
    expected_tree_gpkg_path = os.path.join(expected_pc_local_dir, "obsurv_trees.gpkg")
    expected_bgt_outfile = os.path.join(expected_bgt_local_dir, "BGT_pavements.gpkg")
    
    with mock.patch("os.makedirs") as mock_makedirs:
        result = setup_pc_paths(pc_path, pc_code, pc_out_dir, bgt_out_dir)
        
        assert result == (
            expected_pc_output_path,
            expected_tree_gpkg_path,
            expected_bgt_outfile,
        )
        
        # Ensure both directories were created
        mock_makedirs.assert_has_calls(
            [
                mock.call(expected_pc_local_dir, exist_ok=True),
                mock.call(expected_bgt_local_dir, exist_ok=True),
            ],
            any_order=True,  # This allows the calls to be in any order
        )

        # Check that makedirs was called exactly twice
        assert mock_makedirs.call_count == 2

def test_tree_pc_ready_for_img_conversion():
    """Test whether a point cloud is ready for image conversion."""
    trees_in_pc = gpd.GeoDataFrame({"geometry": [Point(1, 1)]}, crs="EPSG:28992")
    pc_points = np.array([[1, 2, 3], [4, 5, 6]])
    assert tree_pc_ready_for_img_conversion(trees_in_pc, pc_points) is True
    
    empty_trees = gpd.GeoDataFrame([], columns=["geometry"], crs="EPSG:28992")
    assert tree_pc_ready_for_img_conversion(empty_trees, pc_points) is False
    
    empty_pc_points = np.array([])
    assert tree_pc_ready_for_img_conversion(trees_in_pc, empty_pc_points) is False

def test_determine_tree_area_of_interest():
    """Test determining the area of interest of a point cloud."""
    bounds_gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (0, 50), (50, 50), (50, 0)])]}, crs="EPSG:28992")
    trees_gdf = gpd.GeoDataFrame({"geometry": [Point(10, 10), Point(40, 40)]}, crs="EPSG:28992")
    total_pc_bounds = gpd.GeoDataFrame({"FileName": ["pc_code1"], "geometry": [Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])]}, crs="EPSG:28992")
    pc_code = "pc_code1"
    tree_span_buffer = 5
    
    result = determine_tree_area_of_interest(bounds_gdf, trees_gdf, total_pc_bounds, pc_code, tree_span_buffer)
    assert isinstance(result, gpd.GeoDataFrame)
    assert not result.empty

def test_get_trees_from_tile():
    """Test finding trees within the bounds of a point cloud."""
    bounds_gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (0, 50), (50, 50), (50, 0)])]}, crs="EPSG:28992")
    tree_gdf = gpd.GeoDataFrame({"geometry": [Point(10, 10), Point(40, 40)]}, crs="EPSG:28992")
    
    result = get_trees_from_tile(bounds_gdf, tree_gdf)
    assert isinstance(result, gpd.GeoDataFrame)
    assert not result.empty
