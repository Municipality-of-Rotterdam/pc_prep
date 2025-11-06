import tempfile
import os

import geopandas as gpd
from shapely.geometry import Polygon
from unittest import mock
from pc_prep.pavement_prep.utils import (
    convert_asset_to_single_layer_gpkg,
    get_relevant_assets,
    get_assets_from_tile,
)
from pc_prep.pavement_prep.config import AssetClassifier


def test_get_assets_from_tile():
    """Test retrieving pavements from a BGT tile within a bounding box without mocking."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        bgt_filepath = os.path.join(tmpdirname, "test_bgt.gpkg")
        bounding_box = [0, 0, 10, 10]
        
        # Create mock data
        mock_data = gpd.GeoDataFrame(
            {"geometry": [Polygon([(1, 1), (1, 5), (5, 5), (5, 1)])]}, crs="EPSG:28992"
        )
        mock_data.to_file(bgt_filepath, driver="GPKG")
        
        result = get_assets_from_tile(bgt_filepath, bounding_box)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.empty

def test_get_relevant_assets():
    """Test processing relevant pavements from a BGT dataset without mocking."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        bgt_df_path = os.path.join(tmpdirname, "test_bgt.gpkg")
        bgt_outfile = os.path.join(tmpdirname, "output.gpkg")
        bounds_gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])]}, crs="EPSG:28992")
        
        # Create mock BGT data
        mock_pavement_data = gpd.GeoDataFrame({"KLASSE": ["Wegdeel"], "geometry": [Polygon([(1, 1), (1, 5), (5, 5), (5, 1)])]}, crs="EPSG:28992")
        mock_pavement_data.to_file(bgt_df_path, driver="GPKG")
        
        get_relevant_assets(bgt_df_path, bounds_gdf, bgt_outfile)
        
        assert os.path.exists(bgt_outfile)
        result = gpd.read_file(bgt_outfile, layer="asset")
        assert not result.empty

def test_convert_asset_to_single_layer_gpkg():
    """Test converting a BGT dataset into a single-layer GeoPackage without mocking."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_filename = os.path.join(tmpdirname, "test_output.gpkg")
        bgt_gdf = gpd.GeoDataFrame(
            {"KLASSE": ["Voetpad", "Rijwielpad", "Wegdeel"], "geometry": [
                Polygon([(1,1), (1,5), (5,5), (5,1)]),
                Polygon([(6,6), (6,8), (8,8), (8,6)]),
                Polygon([(2,2), (2,4), (4,4), (4,2)])]},
            crs="EPSG:28992"
        )
        data_class = AssetClassifier()
        
        convert_asset_to_single_layer_gpkg(bgt_gdf, output_filename, data_class)
        
        assert os.path.exists(output_filename)
        result = gpd.read_file(output_filename, layer="asset")
        assert not result.empty


@mock.patch("geopandas.read_file")
def test_get_assets_from_tile_mock(mock_read_file):
    """Test retrieving pavements from a BGT tile within a bounding box."""
    bgt_filepath = "test.gpkg"
    bounding_box = [0, 0, 10, 10]
    
    # Mock data
    mock_data = gpd.GeoDataFrame(
        {"geometry": [Polygon([(1, 1), (1, 5), (5, 5), (5, 1)])]}, crs="EPSG:28992"
    )
    mock_read_file.return_value = mock_data
    
    result = get_assets_from_tile(bgt_filepath, bounding_box)
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert not result.empty
    mock_read_file.assert_called_once_with(bgt_filepath)

@mock.patch("pc_prep.pavement_prep.utils.get_assets_from_tile")
@mock.patch("pc_prep.pavement_prep.utils.convert_asset_to_single_layer_gpkg")
def test_get_relevant_assets_mock(mock_convert, mock_get_pavements):
    """Test processing relevant pavements from a BGT dataset."""
    bgt_df_path = "test.gpkg"
    bounds_gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])]}, crs="EPSG:28992")
    bgt_outfile = "output.gpkg"
    
    mock_pavement_data = gpd.GeoDataFrame({"KLASSE": ["Voetpad"], "geometry": [Polygon([(1, 1), (1, 5), (5, 5), (5, 1)])]}, crs="EPSG:28992")
    mock_get_pavements.return_value = mock_pavement_data
    
    get_relevant_assets(bgt_df_path, bounds_gdf, bgt_outfile)
    
    mock_get_pavements.assert_called_once_with(asset_filepath=bgt_df_path, bounding_box=bounds_gdf.iloc[0].geometry.bounds)
    mock_convert.assert_called_once()

@mock.patch("geopandas.GeoDataFrame.to_file")
def test_convert_asset_to_single_layer_gpkg_mock(mock_to_file):
    """Test converting a BGT dataset into a single-layer GeoPackage."""
    bgt_gdf = gpd.GeoDataFrame({"KLASSE": ["Voetpad", "Rijwielpad"], "geometry": [Polygon([(1,1), (1,5), (5,5), (5,1)]), Polygon([(6,6), (6,8), (8,8), (8,6)])]}, crs="EPSG:28992")
    output_filename = "test_output.gpkg"
    data_class = AssetClassifier()
    
    convert_asset_to_single_layer_gpkg(bgt_gdf, output_filename, data_class)
    
    assert "classification" in bgt_gdf.columns
    assert "instance" in bgt_gdf.columns
    mock_to_file.assert_called_once_with(output_filename, layer="asset", driver="GPKG")
