import os
import subprocess as sp
import tempfile
from pathlib import Path

WORK_DIR = Path(__file__).parent.parent

def install_requirements(requirements_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(requirements_content.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        sp.check_call(["pip", "install", "-r", temp_file_path])
    finally:
        # Optionally, you can remove the temporary file after installation
        import os

        os.remove(temp_file_path)
