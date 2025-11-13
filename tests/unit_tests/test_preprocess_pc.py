from unittest import mock
import tempfile

from shapely.geometry import Point
import geopandas as gpd

from pc_prep.tree_prep.preprocess_pc import preprocess_pc
from pc_prep.tree_prep.utils import setup_pc_paths

@mock.patch("pc_prep.tree_prep.preprocess_pc.preprocess_pc_metadata", return_value=None)
@mock.patch("pc_prep.tree_prep.preprocess_pc.preprocess_tree_df", return_value=None)
@mock.patch("pc_prep.tree_prep.preprocess_pc.get_pc_bounds", return_value=None)
@mock.patch("pc_prep.tree_prep.preprocess_pc.get_trees_from_tile", return_value=gpd.GeoDataFrame(
    {"ID": [1, 2, 3], "geometry": [Point(10, 20), Point(20, 40), Point(30, 60)]},
    crs="EPSG:28992"
))
@mock.patch("pc_prep.tree_prep.preprocess_pc.determine_tree_area_of_interest", return_value=None)
@mock.patch("pc_prep.tree_prep.preprocess_pc.enlarge_pc_tile")
@mock.patch("pc_prep.tree_prep.preprocess_pc.get_relevant_assets")
@mock.patch("pc_prep.tree_prep.preprocess_pc.add_asset_classification_to_pc", return_value=([], [], [], []))
def test_preprocess_pc(
    mock_add_asset_classification_to_pc,
    mock_get_relevant_assets,
    mock_enlarge_pc_tile,
    mock_determine_tree_area_of_interest,
    mock_get_trees_from_tile,
    mock_get_pc_bounds,
    mock_preprocess_tree_df,
    mock_preprocess_pc_metadata,
):
    """Test preprocess_pc function using a temporary directory for file output."""

    with tempfile.TemporaryDirectory() as temp_dir:
        pc_path = f"{temp_dir}/pointcloud.laz"
        pc_code = "pc123"
        pc_dir = f"{temp_dir}/pc_mount"
        bgt_dir = f"{temp_dir}/bgt_mount"
        pc_metadata_df_path = f"{temp_dir}/metadata.pkl"
        tree_df_path = f"{temp_dir}/trees.pkl"
        bgt_df_path = f"{temp_dir}/bgt.gpkg"

        # Update mock setup_pc_paths to return paths inside temp_dir
        pc_output_path, tree_gpkg_path, bgt_pavements_path = setup_pc_paths(
            pc_path=pc_path, pc_code=pc_code, pc_out_dir=pc_dir, bgt_out_dir=bgt_dir
        )

        result = preprocess_pc(
                pc_path=pc_path,
                pc_code=pc_code,
                pc_metadata_df_path=pc_metadata_df_path,
                tree_df_path=tree_df_path,
                bgt_df_path=bgt_df_path,
                pc_output_path=pc_output_path,
                tree_gpkg_path=tree_gpkg_path,
                bgt_pavements_path=bgt_pavements_path,
        )

        # Ensure called functions were executed
        mock_preprocess_pc_metadata.assert_called_once()
        mock_preprocess_tree_df.assert_called_once()
        mock_get_pc_bounds.assert_called_once()
        mock_get_trees_from_tile.assert_called_once()
        mock_determine_tree_area_of_interest.assert_called_once()
        mock_enlarge_pc_tile.assert_called_once()
        mock_get_relevant_assets.assert_called_once()
        mock_add_asset_classification_to_pc.assert_called_once()

        # Validate output structure
        assert isinstance(result, tuple)
        assert len(result) == 2
