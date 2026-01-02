import os
import zipfile
import requests
from io import BytesIO
from typing import Tuple

import pandas as pd
from config import POCConfig


class MovieLensLoader:
    """
    Responsible for loading the MovieLens dataset in a safe, repeatable way.

    Design principles:
    - Download only if data is missing
    - Never re-download unnecessarily
    - Fail loudly if data is corrupted
    - Return clean, ready-to-use DataFrames
    """

    def __init__(self, cfg: POCConfig):
        self.cfg = cfg

    def _dataset_exists(self) -> bool:
        """
        Check whether the expected MovieLens files already exist.
        """
        ratings_path = os.path.join(self.cfg.data_dir, "ratings.csv")
        movies_path = os.path.join(self.cfg.data_dir, "movies.csv")
        return os.path.exists(ratings_path) and os.path.exists(movies_path)

    def _download_and_extract(self) -> None:
        """
        Download and extract the MovieLens dataset.
        """
        print("MovieLens dataset not found. Downloading...")

        response = requests.get(self.cfg.movielens_url, timeout=120)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall("data")

        print("Download and extraction complete.")

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ratings and movies DataFrames.

        This method is safe to call multiple times.
        """
        if not self._dataset_exists():
            self._download_and_extract()
        else:
            print("MovieLens dataset found locally. Skipping download.")

        ratings_path = os.path.join(self.cfg.data_dir, "ratings.csv")
        movies_path = os.path.join(self.cfg.data_dir, "movies.csv")

        try:
            ratings = pd.read_csv(ratings_path)
            movies = pd.read_csv(movies_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to load MovieLens CSV files. "
                "The dataset may be corrupted."
            ) from e

        required_rating_cols = {"userId", "movieId"}
        required_movie_cols = {"movieId", "title"}

        if not required_rating_cols.issubset(ratings.columns):
            raise ValueError("ratings.csv is missing required columns")

        if not required_movie_cols.issubset(movies.columns):
            raise ValueError("movies.csv is missing required columns")

        return ratings, movies
