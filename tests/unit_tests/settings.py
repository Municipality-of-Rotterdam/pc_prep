"""Settings for the tests."""

import os
import tempfile
import uuid
from pathlib import Path

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(Path.home())
TEMP_FOLDER = f"tmp_{uuid.uuid4().hex}"
TEMP_PATH_NT = os.sep.join([PROJECT_ROOT, TEMP_FOLDER])
TEMP_PATH_UNIX = tempfile.mkdtemp()
TEMP_DESTINATION = TEMP_PATH_NT if os.name == "nt" else TEMP_PATH_UNIX
