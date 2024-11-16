from datetime import datetime
from time import sleep
import logging
import pickle

from metaflow import FlowSpec, step, IncludeFile, S3, batch
from asunnot.utils import install_requirements, WORK_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://asuntojen.hintatiedot.fi/haku/?cr=1&ps={index}&t=3&l=0&z={page}&search=1&sf=0&so=a&renderType=renderTypeTable&print=0&submit=seuraava+sivu+%C2%BB"  # l=0 means 'finnish'
TEST = True
INDICES_DIR_NAME = "postinumerot"
MAX_PAGES_PER_INDEX = 10
dt = datetime.now()
# the online database only keeps the last 12 months
YEAR_AND_MONTH = f'{dt.year}{0 if dt.month < 10 else ""}{dt.month}'


def get_indices_from_postinumerot_file(str_):
    indices = list()
    str_ = str_.split("\n")
    for line in str_:
        index = line.split(" ")[0]
        indices.append(index)
    return indices


def _get_elevator_bool(str_):
    str_ = str_.strip().lower()
    if str_ == "on":
        return True
    elif str_ == "ei":
        return False
    else:
        raise ValueError


def _str_to_int(str_):
    try:
        return int(float(str_))
    except ValueError:
        return float("nan")




class Scrape(FlowSpec):
    requirements = IncludeFile(
        "requirements",
        is_text=True,
        help="Dependencies on 3rd party packages",
        default="requirements.txt",
    )

    indices_str = IncludeFile(
        "indices",
        is_text=True,
        help="Postal indices considered",
        default=str(WORK_DIR / INDICES_DIR_NAME / "test.txt"),  # small subset for debug
        # default=str(WORK_DIR / INDICES_DIR_NAME / "all.txt"),  # all indices for training
    )

    @step
    def start(self):
        """Initial step"""
        logger.info("Starting the flow")
        self.next(self.scrape)

    @batch(cpu=1, memory=500)
    @step
    def scrape(self):
        """Start the scraping process by iterating through postal indices and collecting data."""
        install_requirements(self.requirements)
        import pandas as pd
        from asunnot.db import (
            DATABASE_NAME,
            Col,
        )

        logger.info(f"db: {DATABASE_NAME}, date: {YEAR_AND_MONTH}")
        def get_df_for_index(index):
            def _get_df_from_url(url):
                tables = pd.read_html(url)  # Returns list of all tables on page
                return pd.DataFrame(tables[0])

            df = None
            page = 0
            for _ in range(MAX_PAGES_PER_INDEX):
                page += 1
                url = DEFAULT_URL.format(index=index, page=page)
                try:
                    df_ = _get_df_from_url(url)
                except ValueError:
                    continue
                if len(df_) <= 4:  # empty unfiltered dfs have that many rows
                    break

                if df is None:
                    df = df_
                else:
                    df = pd.concat([df, df_])
            sleep(0.1)  # not scrape too fast
            return df

        def _get_clean_df(df, index, year_month):
            # remove annoying arrow
            rename_dict = {}
            for col in df.columns:
                new_col = col.replace("◄", "").strip()
                rename_dict[col] = new_col
            df = df.rename(columns=rename_dict)

            rename_dict = {"Vh €": Col.PRICE, "€/m2": Col.PRICE_M2}
            df = df.rename(columns=rename_dict)

            # Drop rows with any NaN values
            df = df.dropna()

            # Remove divider rows specifying how many rooms
            df = df[~df[Col.ROOMS].str.contains("huonetta|Yksiö|Tuloksia|Kaksiot|Kolmiot")]

            # Filter out districts with names shorter than 3 characters
            df = df[df[Col.DISTRICT].str.len() >= 3]

            # Convert area from dm^2 to m^2
            try:
                df[Col.AREA] = df[Col.AREA].apply(lambda x: 0.01 * float(x))
            except ValueError:
                raise ValueError

            # Simplify energy classification to first character
            df[Col.ENERGY] = df[Col.ENERGY].apply(lambda x: x[:1])

            df[Col.PRICE] = df[Col.PRICE].apply(_str_to_int)
            df[Col.PRICE_M2] = df[Col.PRICE_M2].apply(_str_to_int)
            df[Col.YEAR] = df[Col.YEAR].apply(_str_to_int)
            df[Col.FLOOR] = df[Col.FLOOR].apply(lambda x: abs(_str_to_int(x.split("/")[0])))
            df[Col.ROOMS] = df[Col.ROOMS].apply(lambda x: _str_to_int(x[0]))
            df[Col.ELEVATOR] = df[Col.ELEVATOR].apply(_get_elevator_bool)

            df[Col.POSTAL] = index
            df[Col.DATE] = year_month

            # any failed conversion to int return nan
            df = df.dropna()
            return df


        df_all_list = []
        indices = get_indices_from_postinumerot_file(self.indices_str)
        for i, index in enumerate(indices):
            log_str = f"{index} {i} / {len(indices)}"
            df = get_df_for_index(index)
            if df is not None:
                df = _get_clean_df(df, index, YEAR_AND_MONTH)
                df_all_list.append(df)
                log_str += " -> success"
            else:
                log_str += f" -> fail"
            logger.info(log_str)

        df_all = pd.concat(df_all_list)
        logger.info(f"{len(df_all)} entries")
        self._df_all = df_all
        self.next(self.write_db)

    @batch(cpu=1, memory=500)
    @step
    def write_db(self):
        """End the scraping process by saving the collected data to the database."""
        install_requirements(self.requirements)
        from sqlalchemy import create_engine
        from asunnot.db import (
            TABLE_NAME,
            DB,
            Col,
        )

        # write; leads to duplicates if already exists
        db = DB()
        self._df_all.to_sql(TABLE_NAME, con=db._engine, if_exists="append", index=False)

        # read; with duplicates potentially
        df = db.get_df()

        # finally overwrite the table without any duplicates
        non_date_columns = df.columns.difference([Col.DATE])
        df_deduplicated = df.loc[df.groupby(list(non_date_columns))[Col.DATE].idxmin()]
        engine = create_engine(f"sqlite:///{TABLE_NAME}.db", echo=True)
        df_deduplicated.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
        logger.info(f"{len(df)} rows, {len(df.columns)} cols")
        logger.info(f"Example {df[-5:]}")
        with S3(run=self) as s3:
            with open(f"{TABLE_NAME}.db", "rb") as f:
                url = s3.put("database", f.read())
            logger.info(f"Database saved at {url}")
        self.next(self.prepare_train_data)

    @batch(cpu=1, memory=500)
    @step
    def prepare_train_data(self):
        """Load data from the database, preprocess it, and split it into training and testing sets."""
        install_requirements(self.requirements)
        import numpy as np
        from sklearn.model_selection import train_test_split
        from asunnot.db import DB, encode_df, Col, TABLE_NAME

        with S3(run=self) as s3:
            obj = s3.get("database")
            with open(f"{TABLE_NAME}.db", "wb") as f:
                f.write(obj.blob)  # Write the binary content to a local file
        db = DB()
        df = db.get_df()
        not_input_cols = [Col.DISTRICT, Col.PRICE, Col.PRICE_M2]
        output_cols = [Col.PRICE_M2]
        input_df = df.drop(columns=not_input_cols)
        output_df = df.drop(columns=df.columns.difference(output_cols))

        encode_df(input_df)
        input_df = input_df.sort_index(axis=1)  # same order as in eval
        logger.info(
            f"Input DataFrame rows: {len(input_df)}, cols: {len(input_df.columns)}"
        )
        logger.info(
            f"Unique values: { {col: np.unique(input_df[col].values) for col in [Col.BUILDING, Col.ELEVATOR, Col.STATE, Col.PLOT, Col.ENERGY]} }"
        )
        logger.info(f"First 5 rows: {input_df[:5]}")
        logger.info(f"First 5 rows of output_df: {output_df[:5]}")

        logger.info(f"Missing values in input_df: {input_df.isna().sum()}")
        logger.info(f"Missing values in output_df: {output_df.isna().sum()}")

        X = input_df.to_numpy()
        y = output_df.to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=0, shuffle=True
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.next(self.train_model)

    @batch(cpu=4, memory=2000)
    @step
    def train_model(self):
        """Train a Random Forest model on the training data and evaluate its performance on the test data. Save the model"""
        install_requirements(self.requirements)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        import numpy as np

        clf_base = RandomForestRegressor(criterion="squared_error")

        n_estimators = [200, 400]
        max_depth = [None]
        min_samples_split = [4]
        min_samples_leaf = [1]
        min_weight_fraction_leaf = [0.0]
        max_features = [5, 10, 20]
        max_leaf_nodes = [None]

        random_grid = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
        }

        rf_random = RandomizedSearchCV(
            estimator=clf_base,
            param_distributions=random_grid,
            n_iter=30,
            verbose=0,
            random_state=42,
            n_jobs=4,
        )

        rf_random.fit(self.X_train, self.y_train)
        logger.info(rf_random.best_params_)
        clf = rf_random.best_estimator_
        logger.info(clf.feature_importances_)

        pickled_data = pickle.dumps(clf)
        with S3(run=self) as s3:
            url = s3.put("trained_model", pickled_data)
        logger.info(f"Model saved to {url}")

        predictions = clf.predict(self.X_test)
        predictions = predictions.reshape(-1, 1)

        errors = (predictions - self.y_test) ** 2
        logger.info(f"RMSE: {np.sqrt(np.mean(errors))}")
        self.next(self.end)

    @step
    def end(self):
        """Final step"""
        logger.info("Database and trained model stored to s3 - see logs")

if __name__ == "__main__":
    Scrape()
