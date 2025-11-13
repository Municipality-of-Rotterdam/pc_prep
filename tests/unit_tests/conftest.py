import pytest
import tempfile
import os

import laspy
import numpy as np

from test_utils import get_area_bounds, generate_bgt, generate_pointcloud

@pytest.fixture
def area_bounds(request):
    """Fixture to provide area bounds with optional size parameter."""
    size = getattr(request, "param", 50)  # Default size = 50
    return get_area_bounds(size)

@pytest.fixture
def pointcloud_fixture(area_bounds, request):
    """Fixture to generate a point cloud with optional tree count parameter."""
    x_min, x_max, y_min, y_max = area_bounds
    num_trees = request.param.get("num_trees", 10)
    ground_points = request.param.get("ground_points", 10000)
    tree_points = request.param.get("tree_points", 1000)
    las = generate_pointcloud(area_bounds=(x_min, x_max, y_min, y_max), num_trees=num_trees, ground_points=ground_points, tree_points=tree_points)

    temp_dir = tempfile.mkdtemp()
    las_fname = os.path.join(temp_dir, "test_pc.laz")
    las.write(las_fname)

    yield las_fname

    os.remove(las_fname)
    os.rmdir(temp_dir)
    
@pytest.fixture
def bgt_fixture(area_bounds):
    x_min, x_max, y_min, y_max = area_bounds
    temp_dir = tempfile.mkdtemp()
    bgt_fname = os.path.join(temp_dir, "test_bgt.gpkg")
    gdf = generate_bgt(x_min, x_max, y_min, y_max)
    gdf.to_file(bgt_fname, layer="asset", driver="GPKG")
    yield bgt_fname
    os.remove(bgt_fname)
    os.rmdir(temp_dir)

@pytest.fixture
def mock_las_file(tmp_path):
    """Creates a temporary LAS file with mock point cloud data."""
    input_file = tmp_path / "test_input.laz"
    output_file = tmp_path / "test_output.laz"

    # Define a simple LAS file header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]

    # Create sample point cloud data (some inside, some outside AOI)
    num_points = 5
    points = laspy.LasData(header)
    points.x = np.array([10, 20, 30, 40, 50])  # Only 10 & 20 should be kept
    points.y = np.array([10, 20, 30, 40, 50])
    points.z = np.array([1, 2, 3, 4, 5])  # Random height values

    # Write mock LAS file
    points.write(input_file)

    return input_file, output_file