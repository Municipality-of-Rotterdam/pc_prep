import argparse
import tempfile
import pytest
import argparse
import os
from unittest.mock import patch, mock_open, MagicMock
from unittest import mock

import geopandas as gpd

from pc_prep.tree_prep.prep_pc import (
    configure_arg_parser,
    process_single_pc,
    prep_pc
)

def test_configure_arg_parser():
    """Test that the argument parser correctly parses arguments."""
    test_args = [
        "--tree_df_path", "test_tree.pkl",
        "--pc_raw_metadata", "metadata.gpkg",
        "--pc_raw", "test_mount",
        "--reference_trees_path", "ref_trees.pkl",
        "--bgt_pavements_raw", "bgt.gpkg",
        "--resolution", "0.1",
        "--pc_dir", "pc_out",
        "--pc_metadata", "pc_files.json",
        "--img_dir", "img_out",
        "--img_metadata", "img_files.json",
        "--bgt_dir", "bgt_out",
        "--bgt_metadata", "bgt_files.json",
        "--num_workers", "10",
        "--overwrite", "False",
        "--debug", "True"
    ]
    
    with mock.patch("argparse._sys.argv", ["script_name"] + test_args):
        args = configure_arg_parser()
        assert args.tree_df_path == "test_tree.pkl"
        assert args.pc_raw_metadata == "metadata.gpkg"
        assert args.pc_raw == "test_mount"
        assert args.reference_trees_path == "ref_trees.pkl"
        assert args.bgt_pavements_raw == "bgt.gpkg"
        assert args.resolution == 0.1
        assert args.pc_dir == "pc_out"
        assert args.pc_metadata == "pc_files.json"
        assert args.img_dir == "img_out"
        assert args.img_metadata == "img_files.json"
        assert args.bgt_dir == "bgt_out"
        assert args.bgt_metadata == "bgt_files.json"
        assert args.debug is True
        assert args.num_workers == 10
        assert args.overwrite is False


@pytest.fixture
def mock_args():
    """Fixture to create mock argparse arguments."""
    return argparse.Namespace(
        tree_df_path="dummy_tree_df.csv",
        pc_raw_metadata="dummy_metadata.csv",
        pc_raw="dummy_mounted_pc/",
        pc_metadata="output_pc.json",
        img_metadata="output_img.json",
        bgt_metadata="output_bgt.json",
        num_workers=10,
    )

def direct_call_mock(func_var, iterable):
    """Simulates multiprocessing by calling the function directly for each iterable item,
    ensuring results are collected properly in a tqdm loop."""
    for item in iterable:
        yield func_var(item)  # Yield results instead of returning a list

@patch("pc_prep.tree_prep.prep_pc.prepare_pc_paths")
@patch("pc_prep.tree_prep.prep_pc.process_single_pc")
@patch("pc_prep.tree_prep.prep_pc.tqdm")
@patch("pc_prep.tree_prep.prep_pc.multiprocessing.Pool")
@patch("pc_prep.tree_prep.prep_pc.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("pc_prep.tree_prep.prep_pc.json.dump")
def test_prep_pc(
    mock_json_dump,
    mock_open_file,
    mock_makedirs,
    mock_multiprocessing_pool,
    mock_tqdm,
    mock_process_single_pc,
    mock_prepare_pc_paths,
    mock_args,
):
    """Test prep_pc function with multiprocessing mocked out entirely, ensuring the result update logic is correctly simulated."""
    
    # Mock prepare_pc_paths to return predefined file paths
    mock_prepare_pc_paths.return_value = ["pc1.las", "pc2.las"]

    # Define how process_single_pc behaves for given paths
    def mock_process_func(path, args=None):
        base_name = path.replace(".las", "").replace("pc", "")  # Ensure we remove 'pc' from filename
        return {
            "pc": {path: f"processed_{path}"},
            "img": {path: f"processed_img{base_name}.las"},  # Ensure correct format
            "bgt": {path: f"processed_bgt{base_name}.gpkg"}
        }

    mock_process_single_pc.side_effect = mock_process_func

    # Mock tqdm to behave like a normal iterable
    mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

    # Mock the multiprocessing pool instance
    mock_pool_instance = MagicMock()
    mock_pool_instance.imap_unordered.side_effect = lambda func, iterable: direct_call_mock(func, iterable)
    mock_multiprocessing_pool.return_value.__enter__.return_value = mock_pool_instance

    # Run the function
    result_pc, result_img, result_bgt = prep_pc(mock_args)

    # Expected results
    expected_pc = {
        "pc1.las": "processed_pc1.las",
        "pc2.las": "processed_pc2.las",
    }
    expected_img = {
        "pc1.las": "processed_img1.las",
        "pc2.las": "processed_img2.las",
    }
    expected_bgt = {
        "pc1.las": "processed_bgt1.gpkg",
        "pc2.las": "processed_bgt2.gpkg",
    }

    # Validate function outputs
    assert result_pc == expected_pc, f"Expected {expected_pc}, but got {result_pc}"
    assert result_img == expected_img, f"Expected {expected_img}, but got {result_img}"
    assert result_bgt == expected_bgt, f"Expected {expected_bgt}, but got {result_bgt}"

    # Ensure directories were created
    mock_makedirs.assert_any_call(os.path.dirname(mock_args.pc_metadata), exist_ok=True)
    mock_makedirs.assert_any_call(os.path.dirname(mock_args.img_metadata), exist_ok=True)
    mock_makedirs.assert_any_call(os.path.dirname(mock_args.bgt_metadata), exist_ok=True)

    # Ensure JSON files were written
    mock_open_file.assert_any_call("output_pc.json", "w")
    mock_open_file.assert_any_call("output_img.json", "w")
    mock_open_file.assert_any_call("output_bgt.json", "w")
    mock_json_dump.assert_any_call(expected_pc, mock_open_file(), indent=4)
    mock_json_dump.assert_any_call(expected_img, mock_open_file(), indent=4)
    mock_json_dump.assert_any_call(expected_bgt, mock_open_file(), indent=4)

    # Ensure multiprocessing was mocked properly (should never have been called)
    mock_multiprocessing_pool.assert_called_once()
    mock_pool_instance.imap_unordered.assert_called_once()


        
@mock.patch("pc_prep.tree_prep.prep_pc.get_collection_info", return_value="pc_code")
@mock.patch(
    "pc_prep.tree_prep.prep_pc.preprocess_pc",
    return_value=(gpd.GeoDataFrame({"ID": [1, 2, 3], "x": [10, 20, 30], "y": [20, 40, 60]}), [])  # Ensure at least one tree
)
@mock.patch("pc_prep.tree_prep.prep_pc.tree_pc_ready_for_img_conversion", return_value=True)
@mock.patch("pc_prep.tree_prep.prep_pc.setup_img_paths", return_value=("img.tif", "prompt_img.tif", "prompt.json"))
@mock.patch("pc_prep.tree_prep.prep_pc.calculate_point_coords_and_labels", return_value=({}, {}))
@mock.patch("pc_prep.tree_prep.prep_pc.convert_pc_to_img")
@mock.patch("pc_prep.tree_prep.prep_pc.save_image_prompt")
def test_process_single_pc(
    mock_save_image_prompt,
    mock_convert_pc_to_img,
    mock_calculate_point_coords_and_labels,
    mock_prepare_img_paths,
    mock_tree_pc_ready_for_img_conversion,
    mock_preprocess_pc,
    mock_get_collection_info,
):
    """Test processing a single point cloud without multiprocessing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        args = argparse.Namespace(
            tree_df_path=f"{temp_dir}/test_tree.pkl",
            pc_raw_metadata=f"{temp_dir}/metadata.gpkg",
            pc_raw=f"{temp_dir}/test_mount",
            reference_trees_path=f"{temp_dir}/ref_trees.pkl",
            bgt_pavements_raw=f"{temp_dir}/bgt.gpkg",
            resolution=0.1,
            pc_dir=f"{temp_dir}/pc_out",
            pc_metadata=f"{temp_dir}/pc_files.json",
            img_dir=f"{temp_dir}/img_out",
            img_metadata=f"{temp_dir}/img_files.json",
            bgt_dir=f"{temp_dir}/bgt_out",
            bgt_metadata=f"{temp_dir}/bgt_files.json",
            debug=True,
            overwrite=False,
        )
        
        result = process_single_pc("pc1.laz", args)
        
        assert result is not None
        assert "pc" in result
        assert "img" in result
        assert "bgt" in result
        assert "pc1.laz" in result["pc"]
        assert "pc1.laz" in result["img"]
        assert "pc1.laz" in result["bgt"]
        mock_get_collection_info.assert_called_once()
        mock_preprocess_pc.assert_called_once()
        mock_prepare_img_paths.assert_called_once()
        mock_calculate_point_coords_and_labels.assert_called_once()
        mock_convert_pc_to_img.assert_called_once()
        mock_save_image_prompt.assert_called_once()
