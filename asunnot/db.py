import os
import subprocess as sp
import tempfile
from pathlib import Path

from sqlalchemy import create_engine
import pandas as pd

TABLE_NAME = "asunnot"
DATABASE_NAME = f"{TABLE_NAME}.db"
GET_ALL_QUERY = f"SELECT DISTINCT * FROM {TABLE_NAME}"
WORK_DIR = Path(__file__).parent.parent


class DB:
    def __init__(self):
        self._engine = create_engine(f"sqlite:///{DATABASE_NAME}", echo=True)

    def get_df(self):
        with self._engine.connect() as conn:
            return pd.read_sql(GET_ALL_QUERY, conn)


class Col:
    DISTRICT = "Kaupunginosa"
    ROOMS = "Huoneisto"
    BUILDING = "Talot."
    AREA = "m2"
    PRICE = "Vh Eur"
    PRICE_M2 = "Eur/m2"
    YEAR = "Rv"
    FLOOR = "Krs"
    ELEVATOR = "Hissi"
    STATE = "Kunto"
    PLOT = "Tontti"
    ENERGY = "Energial."
    POSTAL = "Postin"
    DATE = "Vuosi"


def encode_df(input_df, normalize=True):
    ENCODING_DICT = {
        Col.BUILDING: {"kt": 0, "ok": 1, "rt": 2},
        Col.STATE: {"huono": 0, "tyyd.": 1, "hyvä": 2},
        Col.PLOT: {"vuokra": 0, "oma": 1},
        Col.ENERGY: {
            "A": 6,
            "B": 5,
            "C": 4,
            "D": 3,
            "E": 2,
            "F": 1,
            "G": 0,
        },
    }

    # Encoding
    for key, mapping in ENCODING_DICT.items():
        input_df[key] = input_df[key].map(mapping)

    # Convert postal codes to integers
    input_df[Col.POSTAL] = input_df[Col.POSTAL].map(lambda x: int(x))

    return input_df


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
