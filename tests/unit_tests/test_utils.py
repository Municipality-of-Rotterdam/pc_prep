import numpy as np
import geopandas as gpd
import laspy
from shapely.geometry import Polygon

# Constants
EPSG_CODE = 28992
ROTTERDAM_BOUNDS = {
    "x_min": 80000,
    "x_max": 92000,
    "y_min": 430000,
    "y_max": 442000
}
STEM_RADIUS_MAX = 0.4
STEM_HEIGHT_RANGE = (4, 10)
CROWN_DIAMETER_RANGE = (0.5, 7.5)
CROWN_HEIGHT_RANGE = (3, 5)

# Define the colors with uint16 type but intended use as 0-255 scaled values
TRUNK_COLOR = [np.uint16(x * 255) for x in [139, 69, 19]]  # Brown
CANOPY_COLOR = [np.uint16(x * 255) for x in [34, 139, 34]]  # Green
GROUND_COLOR_RANGE = tuple(np.uint16(x * 255) for x in [100, 160])  # Greyish


# Utility Functions
def random_unit_vector(ndim=3):
    vec = np.random.randn(ndim, 1)
    return vec / np.linalg.norm(vec, axis=0)

def scale_to_ellipsoid(x, y, z, a, b, c):
    return np.array([a * x, b * y, c * z])

def should_keep_point(x, y, z, a, b, c):
    mu_xyz = ((a * c * y) ** 2 + (a * b * z) ** 2 + (b * c * x) ** 2) ** 0.5
    return mu_xyz / (b * c) > np.random.uniform(0.0, 1.0)

def get_area_bounds(size=50):
    x_min = np.random.uniform(ROTTERDAM_BOUNDS["x_min"], ROTTERDAM_BOUNDS["x_max"] - size)
    y_min = np.random.uniform(ROTTERDAM_BOUNDS["y_min"], ROTTERDAM_BOUNDS["y_max"] - size)
    return x_min, x_min + size, y_min, y_min + size

def generate_ground(x_min, x_max, y_min, y_max, num_points=10000):
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = 2.0 * np.sin(x / 30.0) + 1.5 * np.cos(y / 40.0) + np.random.normal(0, 0.5, num_points)
    colors = np.random.randint(*GROUND_COLOR_RANGE, (num_points, 3))
    return np.column_stack((x, y, z)), colors

def generate_tree(tree_x, tree_y, ground_z, num_points=1000):
    np.random.seed(42)
    tree_points, tree_colors = [], []

    stem_radius = np.random.uniform(0.05, STEM_RADIUS_MAX)
    stem_height = max(
        np.random.uniform(*STEM_HEIGHT_RANGE) * (stem_radius / STEM_RADIUS_MAX + np.random.uniform(-0.1, 0.1)),
        STEM_HEIGHT_RANGE[0]
    )
    crown_diameter = max(
        np.random.uniform(*CROWN_DIAMETER_RANGE) * (stem_height / STEM_HEIGHT_RANGE[1] + np.random.uniform(-0.1, 0.1)),
        CROWN_DIAMETER_RANGE[0]
    )
    crown_a = crown_b = crown_diameter * np.random.uniform(0.6, 1.2)
    crown_c = np.random.uniform(*CROWN_HEIGHT_RANGE) * (stem_height / STEM_HEIGHT_RANGE[1] + np.random.uniform(-0.1, 0.1))

    trunk_top_z = ground_z + stem_height
    crown_base_z = trunk_top_z - 0.5

    trunk_x = np.random.uniform(tree_x - stem_radius, tree_x + stem_radius, num_points)
    trunk_y = np.random.uniform(tree_y - stem_radius, tree_y + stem_radius, num_points)
    trunk_z = np.random.uniform(ground_z, trunk_top_z - 1, num_points)
    tree_points.extend(np.column_stack((trunk_x, trunk_y, trunk_z)))
    tree_colors.extend([TRUNK_COLOR] * num_points)

    canopy_points, canopy_colors = [], []
    while len(canopy_points) < max(100 * crown_diameter ** 2, num_points):
        [px], [py], [pz] = random_unit_vector()
        if should_keep_point(px, py, pz, crown_a, crown_b, crown_c):
            canopy_points.append(scale_to_ellipsoid(px, py, pz, crown_a, crown_b, crown_c) + [tree_x, tree_y, crown_base_z])
            canopy_colors.append(CANOPY_COLOR)

    tree_points.extend(canopy_points)
    tree_colors.extend(canopy_colors)
    return np.array(tree_points), np.array(tree_colors)

def generate_trees(x_min, x_max, y_min, y_max, ground, num_trees, num_points):
    np.random.seed(42)
    tree_points, tree_colors = [], []
    placed_trees = []

    while len(placed_trees) < num_trees:
        tree_x, tree_y = np.random.uniform(x_min + 5, x_max - 5), np.random.uniform(y_min + 5, y_max - 5)
        if all(np.linalg.norm(np.array([tree_x, tree_y]) - np.array(t)) > 2.0 for t in placed_trees):
            placed_trees.append((tree_x, tree_y))
            ground_z = ground[np.argmin(np.linalg.norm(ground[:, :2] - np.array([tree_x, tree_y]), axis=1)), 2]
            points, colors = generate_tree(tree_x=tree_x, tree_y=tree_y, ground_z=ground_z, num_points=num_points)
            tree_points.extend(points)
            tree_colors.extend(colors)

    return np.array(tree_points), np.array(tree_colors)

def generate_bgt(x_min, x_max, y_min, y_max, epsg_code=EPSG_CODE):
    """Generates a BGT pavement polygon within the given bounds."""
    return gpd.GeoDataFrame({
        "geometry": [Polygon([(x_min + 10, y_min + 10), (x_max - 10, y_min + 10),
                                (x_max - 10, y_max - 10), (x_min + 10, y_max - 10)])],
        "classification": [2],
        "instance": [1]
    }, crs=f"EPSG:{epsg_code}")
    
def generate_pointcloud(area_bounds, num_trees=25, ground_points=10000, tree_points=1000):
    x_min, x_max, y_min, y_max = area_bounds
    ground, ground_colors = generate_ground(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, num_points=ground_points)
    trees, tree_colors = generate_trees(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, ground=ground, num_trees=num_trees, num_points=tree_points)

    points = np.vstack((ground, trees))
    colors = np.vstack((ground_colors, tree_colors))

    header = laspy.LasHeader(point_format=3, version='1.2')
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.red, las.green, las.blue = colors[:, 0], colors[:, 1], colors[:, 2]

    return las