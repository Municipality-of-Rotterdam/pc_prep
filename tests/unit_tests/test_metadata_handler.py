import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from unittest import mock
from pc_prep.tree_prep.metadata_handler import (
    prepare_pc_paths,
    get_collection_info,
    modify_string,
    preprocess_tree_df,
    preprocess_pc_metadata,
)


@mock.patch("pc_prep.tree_prep.metadata_handler.preprocess_tree_df", return_value=gpd.GeoDataFrame({"filepath": ["file1.laz"], "ID": [1], "geometry": [Point(0, 0)]}, crs="EPSG:28992"))
@mock.patch("pc_prep.tree_prep.metadata_handler.preprocess_pc_metadata", return_value=gpd.GeoDataFrame({"filepath": ["file1.laz"], "ID": [1], "geometry": [Point(0, 0)]}, crs="EPSG:28992"))
@mock.patch("pc_prep.tree_prep.metadata_handler.determine_full_filepath_from_laz_metadata", return_value=gpd.GeoDataFrame({"filepath": ["file1.laz"], "ID": [1], "geometry": [Point(0, 0)]}, crs="EPSG:28992"))
def test_prepare_pc_paths(
    mock_determine_full_filepath,
    mock_preprocess_pc_metadata,
    mock_preprocess_tree_df,
):
    """Test prepare_pc_paths function using mocks."""
    tree_df_path = "trees.pkl"
    pc_metadata_df_path = "metadata.gpkg"
    mounted_pc_path = "mount_path"
    
    result = prepare_pc_paths(tree_df_path=tree_df_path, pc_metadata_df_path=pc_metadata_df_path, mounted_pc_path=mounted_pc_path)
    
    mock_preprocess_tree_df.assert_called_once_with(tree_df_path)
    mock_preprocess_pc_metadata.assert_called_once_with(pc_metadata_df_path)
    mock_determine_full_filepath.assert_called_once()
    assert isinstance(result, list)

def test_get_collection_info():
    """Test get_collection_info function."""
    pc_filename = "GEODATA/puntenwolk/MLS/2019/LAZ/nl-rott-190420-7415-laz/las_processor_bundled_out/filtered_1896_8737.laz"
    result = get_collection_info(pc_filename)
    assert result == "nl-rott-190420-7415-laz"
    
    with pytest.raises(ValueError):
        get_collection_info("invalid-file")

def test_modify_string():
    """Test modify_string function."""
    input_str = "cop-0df8f4a1-73e7-42bc-bf91-7e6c01c3c51c-nl-rott-230423-7415-laz"
    result = modify_string(input_str)
    assert result == "nl-rott-230423-7415-laz"
    
    input_str_no_match = "random-string"
    with pytest.raises(AssertionError):
        modify_string(input_str_no_match)

@mock.patch("pandas.read_pickle", return_value=pd.DataFrame({"ID": [1], "WKT_GEOMETRIE": ["POINT(0 0)"]}))
def test_preprocess_tree_df(mock_read_pickle):
    """Test preprocess_tree_df function."""
    tree_df_path = "trees.pkl"
    result = preprocess_tree_df(tree_df_path)
    
    mock_read_pickle.assert_called_once_with(tree_df_path)
    
    expected_result = gpd.GeoDataFrame({"geometry": [Point(0, 0)], "ID": [1]}, crs="EPSG:28992")
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.equals(expected_result)
    assert result.crs == "EPSG:28992"


@mock.patch("geopandas.read_file", return_value=gpd.GeoDataFrame({"Basename": ["nl-rott-230412"], "geometry": [Point(0, 0)]}, crs="EPSG:28992"))
def test_preprocess_pc_metadata(mock_read_file):
    """Test preprocess_pc_metadata function."""
    mls_metadata_path = "metadata.gpkg"
    result = preprocess_pc_metadata(mls_metadata_path)
    
    mock_read_file.assert_called_once_with(mls_metadata_path)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:28992"
