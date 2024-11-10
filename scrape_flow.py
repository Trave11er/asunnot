from datetime import datetime
from time import sleep
from pathlib import Path
import logging

from metaflow import FlowSpec, step, IncludeFile

from asunnot.db import (
    DATABASE_NAME,
    TABLE_NAME,
    DB,
    Col,
    install_requirements,
    WORK_DIR,
)

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


def get_clean_df(df, index, year_month):
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
        default=str(WORK_DIR / INDICES_DIR_NAME / "test.txt"),
    )

    @step
    def start(self):
        """Start the scraping process by iterating through postal indices and collecting data."""
        install_requirements(self.requirements)
        import pandas as pd

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

        df_all_list = []
        indices = get_indices_from_postinumerot_file(self.indices_str)
        for i, index in enumerate(indices):
            log_str = f"{index} {i} / {len(indices)}"
            df = get_df_for_index(index)
            if df is not None:
                df = get_clean_df(df, index, YEAR_AND_MONTH)
                df_all_list.append(df)
                log_str += " -> success"
            else:
                log_str += f" -> fail"
            logger.info(log_str)

        df_all = pd.concat(df_all_list)
        logger.info(f"{len(df_all)} entries")
        self._df_all = df_all
        self.next(self.end)

    @step
    def end(self):
        install_requirements(self.requirements)
        from sqlalchemy import create_engine

        """End the scraping process by saving the collected data to the database."""
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


if __name__ == "__main__":
    logger.info(f"db: {DATABASE_NAME}, date: {YEAR_AND_MONTH}")
    Scrape()
