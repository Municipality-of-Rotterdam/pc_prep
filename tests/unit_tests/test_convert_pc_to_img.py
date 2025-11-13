import os
import numpy as np
import geopandas as gpd
import cv2
import tempfile
import rasterio
from shapely.geometry import Point
from pc_prep.general_prep.convert_pc_to_img import (
    calculate_point_coords_and_labels,
    cloud_to_image,
    save_image_prompt,
    convert_pc_to_img,
)
from pc_prep.general_prep.utils import setup_img_paths

def test_calculate_point_coords_and_labels():
    """Test calculating point coordinates and labels."""
    filtered_points = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]])
    resolution = 1.0
    trees_in_laz = gpd.GeoDataFrame({"geometry": [Point(5, 5), Point(15, 15)]}, crs="EPSG:28992")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        prompt_path = os.path.join(tmpdirname, "prompt.json")
        point_coords, point_labels = calculate_point_coords_and_labels(filtered_points, resolution, trees_in_laz, prompt_path)
        
        assert os.path.exists(prompt_path)
        assert len(point_coords) == 2
        assert len(point_labels) == 2

def test_cloud_to_image():
    """Test converting a point cloud to an image."""
    points = np.array([[0, 0, 10], [10, 10, 20], [20, 20, 30]])
    image = cloud_to_image(points, 0, 20, 0, 20, 1.0)
    
    assert isinstance(image, np.ndarray)
    assert image.shape == (21, 21, 3)

def test_setup_img_paths():
    """Test preparing image paths."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        image_path, prompt_img_path, prompt_path = setup_img_paths(tmpdirname, "pc_code", "basename")
        
        assert os.path.dirname(image_path) == os.path.join(tmpdirname, "basename", "pc_code")
        assert os.path.exists(os.path.dirname(image_path))

def test_convert_pc_to_img():
    """Test converting point cloud to image and saving it."""
    points = np.array([[0, 0, 10], [10, 10, 20], [20, 20, 30]])
    resolution = 1.0
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        image_path = os.path.join(tmpdirname, "raster.tif")
        convert_pc_to_img(image_path, points, resolution)
        
        assert os.path.exists(image_path)
        with rasterio.open(image_path) as src:
            assert src.count == 3  # 3 bands (RGB)
            assert src.width > 0
            assert src.height > 0

def test_save_image_prompt():
    """Test saving an image with prompt points."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        image_path = os.path.join(tmpdirname, "image.png")
        output_path = os.path.join(tmpdirname, "output.png")
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
        
        point_coords = [[10, 10], [20, 20]]
        save_image_prompt(image_path, point_coords, output_path)
        
        assert os.path.exists(output_path)
