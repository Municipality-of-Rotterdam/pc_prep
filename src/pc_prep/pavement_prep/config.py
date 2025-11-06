from typing import TypedDict


class ClassSpec(TypedDict):
    """
    Schema for a single asset class definition used in AssetClassifier.

    Attributes
    ----------
    class_nr : int
        The numeric ID assigned to this class.
        This value is written into the `classification` column of processed
        GeoPackages and used as the point cloud class code.

    class_values : list[str]
        The list of raw string labels (from the BGT / asset source data)
        that map to this numeric class. Example: ["Voetpad", "Voetgangersgebied"]
    """

    class_nr: int
    class_values: list[str]


class AssetClassifier:
    """
    Defines the mapping between raw BGT (or similar) asset class names
    and the numeric class IDs used in point cloud classification.

    This class is used during preprocessing to:
    - Map raw class strings (from the `KLASSE` column) to integer codes.
    - Assign per-class instance indices for segmentation or labeling tasks.
    - Filter out any geometries that do not belong to known asset classes.

    The mapping is stored in the `classifications` dictionary, which has:
        key   -> str  : logical asset group name (e.g. "voetpad", "fietspad")
        value -> ClassSpec : typed dictionary defining the numeric class ID
                             and corresponding raw class string values.
    """

    classifications: dict[str, ClassSpec] = {
        "voetpad": {
            "class_nr": 1,
            "class_values": ["Voetpad", "Voetgangersgebied"],
        },
        "fietspad": {
            "class_nr": 2,
            "class_values": ["Rijwielpad"],
        },
        "autoweg": {
            "class_nr": 3,
            "class_values": [
                "Wegdeel",
                "Autoweg",
                "Autosnelweg",
                "Rijbaan",
                "Regionale Weg",
                "Parkeerplaats",
                "OV Baan",
            ],
        },
        "beplanting": {
            "class_nr": 4,
            "class_values": ["Beplantingen"],
        },
        "gras": {
            "class_nr": 5,
            "class_values": ["Grassen"],
        },
        "haag": {
            "class_nr": 6,
            "class_values": ["Hagen"],
        },
    }
