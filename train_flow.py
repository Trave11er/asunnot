import logging
import pickle

from metaflow import FlowSpec, step, IncludeFile

from asunnot.db import DB, encode_df, Col, install_requirements, WORK_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train(FlowSpec):
    requirements = IncludeFile(
        "requirements",
        is_text=True,
        help="Dependencies on 3rd party packages",
        default="requirements.txt",
    )

    @step
    def start(self):
        install_requirements(self.requirements)
        import numpy as np
        from sklearn.model_selection import train_test_split

        """Load data from the database, preprocess it, and split it into training and testing sets."""
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
        self.next(self.end)

    @step
    def end(self):
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

        with open(WORK_DIR / "clf.pkl", "wb") as fid:
            pickle.dump(clf, fid)

        predictions = clf.predict(self.X_test)
        predictions = predictions.reshape(-1, 1)

        errors = (predictions - self.y_test) ** 2
        logger.info(f"RMSE: {np.sqrt(np.mean(errors))}")


if __name__ == "__main__":
    Train()
