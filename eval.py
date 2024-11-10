import pickle
import logging

import pandas as pd

from asunnot.db import encode_df, Col, WORK_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path, input_array):
    """Load the model from the specified path and evaluate it on the given input array."""
    with open(model_path, "rb") as fid:
        clf = pickle.load(fid)

    predictions = clf.predict(input_array)
    predictions = predictions.reshape(-1, 1)

    logger.info(f"Predictions: {predictions}")


if __name__ == "__main__":
    dict_ = {
        Col.ROOMS: 1,
        Col.AREA: 30.5,
        Col.BUILDING: "rt",
        Col.YEAR: 2018,
        Col.FLOOR: 3,
        Col.ELEVATOR: 1,
        Col.STATE: "tyyd.",
        Col.PLOT: "oma",
        Col.ENERGY: "C",
        Col.POSTAL: 2780,
        Col.DATE: 202410,
    }
    df = pd.DataFrame([dict_])
    df = df.sort_index(axis=1)  # same order as in train
    encode_df(df)
    logger.info(df)
    evaluate_model("clf.pkl", df.iloc[0].to_numpy().reshape(1, -1))
