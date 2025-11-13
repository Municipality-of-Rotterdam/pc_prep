import os
import tempfile
import rasterio
import numpy as np
import pytest
from rasterio.crs import CRS

from pc_prep.general_prep.convert_pc_to_img import convert_pc_to_img
from pc_prep.general_prep.utils import read_laz_points
from pc_prep.pavement_prep.csf_with_pavements import add_asset_classification_to_pc

@pytest.mark.parametrize("area_bounds", [25, 50], indirect=True)
@pytest.mark.parametrize("pointcloud_fixture", [
    {"num_trees": 5, "ground_points": 1000, "tree_points": 100},
    {"num_trees": 15, "ground_points": 100000, "tree_points": 10000},
], indirect=True)
class TestPointCloudProcessing:
    """Test suite for processing point cloud data including asset classification and image conversion."""

    def test_add_asset_classification_to_pc(self, pointcloud_fixture, bgt_fixture):
        """Tests adding asset classification to the point cloud."""
        # Arrange
        input_file = pointcloud_fixture
        bgt_file = bgt_fixture
        csf_threshold = 0.5
        csf_resolution = 0.2
        csf_smooth = False

        # Act
        non_ground_points, ground_points, non_ground_indices, ground_indices = add_asset_classification_to_pc(
            input_file, bgt_file, csf_threshold, csf_resolution, csf_smooth
        )

        # Assert
        assert isinstance(non_ground_points, np.ndarray)
        assert isinstance(ground_points, np.ndarray)
        assert isinstance(non_ground_indices, np.ndarray)
        assert isinstance(ground_indices, np.ndarray)
        assert non_ground_points.shape[1] == 6  # Ensuring correct point format
        assert ground_points.shape[1] == 6  # Ensuring correct point format
        assert len(non_ground_indices) + len(ground_indices) > 0, "Some points should be classified"
        
    @pytest.mark.parametrize("resolution", [0.05, 0.5])  # Image resolution parameter
    def test_convert_pc_to_img(self, pointcloud_fixture, resolution):
        """Tests the conversion of a point cloud to a raster image."""
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "test_image.tif")
        points = read_laz_points(input_file=pointcloud_fixture)

        # Act
        convert_pc_to_img(image_path=image_path, points=points, resolution=resolution)

        # Assert
        assert os.path.exists(image_path), "Raster image was not created"

        with rasterio.open(image_path) as dataset:
            assert dataset.count == 3, "Expected 3 bands (RGB)"
            assert dataset.width > 0 and dataset.height > 0, "Image width and height should be positive"
            assert dataset.dtypes[0] == "uint8", "Expected uint8 dtype for raster data"
            assert dataset.crs == CRS.from_epsg(28992), "CRS should match EPSG:28992"
            for band in range(dataset.count):
                band_data = dataset.read(band + 1)
                assert band_data.min() >= 0 and band_data.max() <= 255, "Pixel values should be between 0-255"
                assert band_data.max() >= 1, "Pixel values should be greater than range 0-1"

        # Cleanup
        os.remove(image_path)
        os.rmdir(temp_dir)
