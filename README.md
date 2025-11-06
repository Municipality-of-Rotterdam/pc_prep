# 🛰️ pc_prep — Point Cloud Preparation Toolkit
[![pc_prep CICD](https://github.com/Municipality-of-Rotterdam/pc_prep/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/Municipality-of-Rotterdam/pc_prep/actions/workflows/main.yml)

`pc_prep` is a Python package for **automated preparation, tiling, and management of LiDAR point cloud data**.
It provides a reproducible preprocessing pipeline that transforms raw point cloud data (`.las` / `.laz`) into analysis-ready, per-tile datasets.  
The workflow integrates **tree AOI raster preparation**, **ground/non-ground classification (CSF)**, and **BGT asset segmentation** with PDAL and GeoPandas, producing both:
- **Filtered point cloud subsets** (tree AOI pointcloud tiles, asset segments (greenery, road), ground/non-ground points), and  
- **Geo-referenced raster images and prompts** for downstream computer vision or GIS workflows.  

This allows seamless preparation of geospatial data for segmentation and urban-scale environmental analysis.


| **Pointcloud tile** |   | **Tree and asset geometries** |   | **2D raster w/ prompts, segmented assets** |
|:------------:|:-:|:------------:|:-:|:------------:|
| ![pc_tile_cc](docs/images/pc_tile_cc.png =800x) | ➕ | ![bgt_pavement_assets](docs/images/bgt_pavement_assets.png =800x) | ➡️ | ![debug_prompt_image](docs/images/debug_prompt_image.png =400x) ![pc_tile_assettype](docs/images/pc_tile_assettype.png =300x) |
*General workflow: Tree and asset geometries are combined with a pointcloud tile to create a promptable raster image and a pointcloud with asset segmentation.*


## 📘 Overview

The toolkit supports:

* **Ground / Non-ground classification** — Separate terrain from vegetation and structural elements using the **Cloth Simulation Filter (CSF)** through PDAL for precise segmentation of ground points.

* **Segmentation of asset polygons** — Perform spatial overlays and segmentation of **BGT** or custom asset polygons with PDAL and GeoPandas to produce per-tile pavement and surface classifications.

* **Point cloud tiling** — Divide large `.las` / `.laz` datasets into uniform tiles using PDAL with multiprocessing and grid-based management for scalable processing.

* **Image preparation** — Convert filtered point clouds into **top-down RGB GeoTIFFs**, and generate **prompt-based image annotations** (`.json`) suitable for segmentation models like **Segment Anything (SAM)**.

* **Metadata generation and consistency checks** — Automatically build structured JSON and GeoPackage metadata linking each tile’s point cloud, raster, and asset layers, ensuring reproducibility across processing stages.

These functionalities together create a robust preprocessing pipeline for transforming raw LiDAR data into analysis-ready geospatial and vision-model training inputs. It is designed to handle **Cyclomedia-style datasets** (NL RD New / EPSG:28992, 50 m grid tiles) but is general enough to adapt to other grid-based LiDAR data.

![pc_tile_cc](docs/images/pc_tile_cc.png =450x) ![pc_tile_ground_classification](docs/images/pc_tile_ground_classification.png =450x)
*Ground-nonground classification.*

## 📂 Required Data Inputs

To run `pc_prep` (and specifically the `tree_prep/prep_pc.py` pipeline), you need the following inputs:

| Data Type                      | Format            | Description                                                                                                                         |
| ----------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Point Cloud Tiles**         | `.las` / `.laz`   | Raw point cloud tiles (Cyclomedia / drone / MLS). Must contain at least X, Y, Z. RGB is used if present.                           |
| **Point Cloud Metadata Index**| `.gpkg`           | GeoPackage describing the point cloud tiles. Must include geometry for each tile and filename info (e.g. `FileName`).              |
| **Tree Locations**         | `.pkl`            | Pickled pandas DataFrame of tree point geometries to process. Each row represents a tree of interest, including geometry in RD New |
| **Reference Trees**           | `.pkl`            | Pickled pandas DataFrame of the canonical / reference tree dataset. This determines the trees in the subset of pc tiles to process; could be either just the input above, or all trees. In case of the latter, all trees within the subset of pc_tiles will be processed.                         |
| **BGT / Asset Polygons**      | `.gpkg`           | GeoPackage of assets (e.g. pavements, paths, road surfaces). Used to extract and classify per-tile pavement/ground surface data.   |
| **Output Directory Roots**    | folder paths      | Base folders where processed point cloud tiles, rasters, prompts, and per-tile BGT assets will be written.                         |


![tree_df_pkl](docs/images/tree_df_pkl.png =250x)
*Example input .pkl file with point geometries for tree locations.*

### Notes
- The preprocessing step derives the per-tile AOI automatically by:
  - finding trees in and around that tile,
  - buffering trees so crowns spanning tile edges are included,
  - merging neighboring tiles if needed,
  - clipping the merged point cloud to that AOI.
- The metadata index (`.gpkg`) is also used to locate and sort point cloud tiles that contain trees, so only relevant tiles are processed.


Example directory layout:

```
data/
│
├── input/
│   ├── drone_2024_05/
│   │   ├── filtered_1846_8712.laz
│   │   ├── filtered_1846_8713.laz
│   │   └── ...
│   └── tile_metadata.gpkg
│
└── output/
    └── processed/
        └── filtered_1846_8712/
            └── nl-rott-20240512-drone-laz/
                ├── raster.tif
                ├── img_with_prompts.png
                ├── point_prompt.json
                └── filtered_1846_8712.laz
```

---

## ⚙️ Installation

To create the environment to run the code, do the following from the root of the repo.

```python
conda env create -f environment.yml
conda activate pc_prep
[pip install poetry &&] poetry install
```

---

## 🚀 Usage Guide

### 4️⃣ Tree & Pavement Preprocessing Pipeline (`tree_prep/prep_pc.py`)

This script is the high-level batch pipeline for preparing per-tile training data for tree segmentation and related analysis.

It does all of the following for each point cloud tile:

1. **Loads + preprocesses** the raw point cloud tile:

   * filters points (keeps e.g. non-ground / relevant classes),
   * extracts tree geometries inside that tile,
   * extracts BGT pavements (assets / surface types).

2. **Rasterizes** the filtered point cloud to a georeferenced top-down RGB GeoTIFF (`raster.tif`).

3. **Generates model prompts**:

   * Converts tree point locations to pixel-space click prompts.
   * Saves them to `point_prompt.json`.
   * (If `--debug true`) overlays them on the raster and exports `img_with_prompts.png` for QA.

4. **Writes per-tile outputs** to structured folders under `--pc_dir`, `--img_dir`, and `--bgt_dir`.

5. **Builds summary metadata JSONs**:

   * `--pc_metadata`: per-tile processed point cloud + tree geometries
   * `--img_metadata`: per-tile raster and prompt files
   * `--bgt_metadata`: per-tile BGT/pavement references

Those metadata JSON files are the handoff to downstream model training code.

---

#### CLI usage

You run it directly:

```bash
python -m pc_prep.tree_prep.prep_pc \
  --tree_df_path data/trees_to_process.pkl \
  --reference_trees_path data/reference_trees.pkl \
  --pc_raw data/pointcloud_tiles/ \
  --pc_raw_metadata data/pointcloud_metadata.gpkg \
  --bgt_pavements_raw data/bgt_assets.gpkg \
  --pc_dir data/output/pc_processed \
  --pc_metadata data/output/pc_processed/pc_metadata.json \
  --img_dir data/output/img_tiles \
  --img_metadata data/output/img_tiles/img_metadata.json \
  --bgt_dir data/output/bgt_processed \
  --bgt_metadata data/output/bgt_processed/bgt_metadata.json \
  --resolution 0.05 \
  --num_workers 8 \
  --overwrite false \
  --debug false
```

---

### Argument Details

#### Required input arguments

* `--tree_df_path`
  Path to a pickle (`.pkl`) containing the trees you want to process.
  This is typically a DataFrame where each row is a tree of interest.

* `--reference_trees_path`
  Path to reference tree data (`.pkl`) that’s used to match trees in the tile. 
  This determines the trees in the subset of pc tiles to process

* `--pc_raw`
  Directory where the raw point cloud tiles live (`.las` / `.laz`).
  Example: `data/pointcloud_tiles/filtered_1846_8712.laz`, etc.

* `--pc_raw_metadata`
  A GeoPackage (`.gpkg`) containing metadata for those tiles.
  This should have geometries and columns like `FileName`, `Basename`, etc.
  It’s used to locate the right tile(s) for each tree.

* `--bgt_pavements_raw`
  A GeoPackage (`.gpkg`) with the BGT pavements / assets layer.
  Used to overlay asset classes like pavements, sidewalks, etc. per tile.

#### Required output arguments

* `--pc_dir`
  Directory for processed point cloud + extracted tree geometries.

* `--pc_metadata`
  Path to a JSON file that will be created.
  Maps tile ID → processed point cloud info + tree geometry file.

* `--img_dir`
  Directory for rasterized top-down RGB images and prompt overlays:

  * `raster.tif`
  * `img_with_prompts.png` (if debug)
  * `point_prompt.json`

* `--img_metadata`
  Path to a JSON file that will be created.
  Maps tile ID → raster/prompt paths for each tile.

* `--bgt_dir`
  Directory for per-tile BGT pavement geometries (cropped pavement assets).

* `--bgt_metadata`
  Path to a JSON file that will be created.
  Maps tile ID → pavement layer output file.

---

### Processing options

* `--resolution` (default: `0.05`)
  Pixel size in meters/pixel when rasterizing the point cloud.
  Tiles are ~50 m × 50 m.
  At 0.05 m/px you get ~1000 × 1000 px rasters, which is near 1024 × 1024 and lines up with Segment Anything's encoder expectations.

* `--num_workers`
  How many parallel workers to use.
  If not provided, the script uses `CPU cores - 1`.
  Each worker runs `process_single_pc()` for one tile.

* `--overwrite` (`true` / `false`)
  If `true`, reprocess tiles even if output files already exist.
  If `false`, tiles that already produced all expected outputs are skipped and just referenced in metadata.

* `--debug` (`true` / `false`)
  If `true`, we also create a diagnostic image per tile:
  `img_with_prompts.png`, which is the raster with large red star markers showing the per-tree click prompts that are sent to your segmentation model.

---

### Outputs

After a run, you’ll have:

* A folder per tile in `--pc_dir`, containing:

  * filtered point cloud (`.laz`)
  * per-tile tree geometries (`obsurv_trees.gpkg`)

* A folder per tile in `--img_dir`, containing:

  * top-down raster (`raster.tif`)
  * point prompts (`point_prompt.json`)
  * optional debug overlay (`img_with_prompts.png`)

* A folder per tile in `--bgt_dir`, containing:

  * BGT pavements GeoPackage (`BGT_pavements.gpkg`)

* Three JSON manifests:

  * `pc_metadata.json`
  * `img_metadata.json`
  * `bgt_metadata.json`

Those JSONs are important — they are the lookup tables telling downstream code:

* which raster goes with which point cloud,
* where the prompt clicks are,
* where the pavement data lives for that same tile.

They are the “index” for model training, inference, visualization, etc.


### General prep module

The general processing utilities live in the general_prep module.
They handle merging, filtering, and image path setup.

#### Example: Merge and Filter Tiles Around an AOI

```python
from pc_prep.general_prep.utils import enlarge_pc_tile, get_pc_bounds
import geopandas as gpd

tiles_aoi = gpd.read_file("data/input/tile_metadata.gpkg")
pc_path = "data/input/drone_2024_05/filtered_1846_8712.laz"

success = enlarge_pc_tile(
    pc_path=pc_path,
    tiles_aoi=tiles_aoi,
    outfname="data/output/processed/filtered_1846_8712/nl-rott-20240512-drone-laz/merged_filtered.laz",
)

if success:
    print("Tile successfully enlarged and filtered!")
```

This function:

1. Finds neighboring point clouds around the AOI.
2. Merges them into one temporary file.
3. Clips them to the AOI bounds.
4. Removes any intermediate files.

---

#### Point Cloud Tiling (`pc_prep/general_prep/tile_pc.py`)

`tile_pc.py` can be run **from the command line** to:

* Generate a tile grid (`grid` command)
* Tile a large `.laz` file (`tile` command)
* Do both in one step (`pc` command)

#### CLI Usage

```bash
python -m pc_prep.tile_pc <subcommand> [options]
```

#### Commands and Options

##### Create Grid

```bash
python -m pc_prep.tile_pc grid \
  -i data/input/large_dataset.laz \
  -g data/output/grid_metadata.gpkg \
  --tile-size 50.0 \
  --recording-tag drone
```

This creates a 50 m × 50 m grid based on the LAS file’s extent and writes it as a GeoPackage.

##### Tile Point Cloud Using a Grid

```bash
python -m pc_prep.tile_pc tile \
  -i data/input/large_dataset.laz \
  -g data/output/grid_metadata.gpkg \
  -o data/output/tiles/ \
  --recording-tag drone \
  --processes 8
```

Tiles the LAS file into individual `.laz` files using PDAL, with multiprocessing.

##### One-Step Grid + Tiling

```bash
python -m pc_prep.tile_pc pc \
  -i data/input/large_dataset.laz \
  -g data/output/grid_metadata.gpkg \
  -o data/output/tiles/ \
  --tile-size 50.0 \
  --recording-tag drone
```

This runs both grid creation and tiling in a single command.

#### Output

* Individual `.laz` tiles, named by upper-left corner (e.g. `tile_UL_1200_4500.laz`) or the "Filename" col value if provided by the grid.
* Updated `tile_metadata.gpkg` stored under the same output directory
* Optional progress bar with `tqdm`

---

## 📑 Data Structures

### Tile Metadata (GeoPackage)

| Column     | Type      | Description                                                       |
| ---------- | --------- | ----------------------------------------------------------------- |
| `FileName` | `str`     | Relative path or filename of tile (e.g. `filtered_1846_8712.laz`). |
| `geometry` | `Polygon` | Polygon bounds of the tile (EPSG:28992).                          |

### Point Cloud Tiles

| Attribute       | Description                                              |
| --------------- | -------------------------------------------------------- |
| `.laz` / `.las` | LAS/LAZ tiles cropped to 50 m bounding boxes.            |
| `.tif`          | Rasterized image generated from point cloud.             |
| `.png`          | Visualization of point prompts overlaid on raster.       |
| `.json`         | Point prompt coordinates for segmentation or SAM models. |

---

## 🧠 Developer Notes

* CRS is assumed to be **EPSG:28992 (Amersfoort / RD New)** by default.
* Tile size defaults to **50 m × 50 m** to match Cyclomedia NL specifications.
* Temporary files (e.g., `_tmp.laz`) are automatically removed after filtering.
* Empty or small tiles (below 1 MB or 1 million points) are skipped.

---

## 🧾 License

This project is distributed under the EUPL License.
See [LICENSE](./LICENSE) for details.

---

## 🗾️ Citation

This module builds on modified routines from [**segment-lidar**](https://github.com/Yarroudh/segment-lidar), adapted for
prompt-based tree segmentation in urban LiDAR workflows.

## 👥 Authors & Contact

Developed by the **City of Rotterdam** for point cloud preprocessing automation.

---
