from contextlib import contextmanager
import logging

import time
from pathlib import Path
from typing import Optional, List

import numpy as np
from sqlalchemy import Table, Column, Integer, TEXT, MetaData, create_engine, text
from sqlalchemy.orm import sessionmaker

import polars as pl
from rapidfuzz import fuzz


from tire_vision.config import IndexConfig


metadata = MetaData()


def calculate_score(name: str, queries: List[str]):
    return sorted(
        list(map(lambda x: (fuzz.ratio(name, x) / 100.0, x), queries)),
        reverse=True,
    )


class TireModelDatabase:
    def __init__(self, config: IndexConfig):
        self.config = config
        self.logger = logging.getLogger("tire_db")
        self.models_table = Table(
            self.config.table_name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", TEXT, nullable=False),
            Column("parent_id", Integer, nullable=False, default=0),
        )

        connection_url = (
            f"mysql+pymysql://{config.db_user}:{config.db_password}@"
            f"{config.db_host}:{config.db_port}/{config.db_name}"
        )
        self.engine = create_engine(
            connection_url, pool_pre_ping=True, pool_recycle=3600, echo=False
        )
        self.session = sessionmaker(self.engine, expire_on_commit=False)
        self.last_table_update = -float("inf")
        self.table_path = Path(self.config.table_cache_path)
        self._table = None

        similarity_metrics = {
            "product": self.product,
            "arithmetic_mean": self.arithmetic_mean,
            "harmonic_mean": self.harmonic_mean,
            "geometric_mean": self.geometric_mean,
            "euclidean": self.euclidean,
        }
        self.similarity_metric = similarity_metrics[self.config.similarity_metric]

        self.logger.info("TireModelDatabase initialized successfully")

    @contextmanager
    def get_session(self):
        with self.session() as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise

    def execute_query(self, query: str):
        with self.get_session() as session:
            result = session.execute(text(query))
            return result.fetchall()

    def _load_table_from_db(self) -> Optional[pl.DataFrame]:
        result = None
        try:
            result = pl.read_database(
                f"select * from {self.config.table_name}", self.engine
            )
            self.last_table_update = time.perf_counter()

        except Exception as e:
            self.logger.error(f"Error loading table from db: {e}")
            return None

        return result

    def _is_cache_expired(self) -> bool:
        return (
            time.perf_counter() - self.last_table_update
            > self.config.table_cache_ttl_seconds
        )

    def _save_table_to_disk(self):
        if self._table is not None:
            try:
                self.table_path.parent.mkdir(parents=True, exist_ok=True)
                self._table.write_parquet(self.table_path)
                self.logger.debug("Table saved to disk successfully")
            except Exception as e:
                self.logger.error(f"Failed to save table to disk: {e}")

    def _load_table_from_disk(self) -> Optional[pl.DataFrame]:
        try:
            if self.table_path.exists():
                table = pl.read_parquet(self.table_path)
                self.logger.info("Table loaded from disk cache")
                return table
        except Exception as e:
            self.logger.error(f"Error loading table from disk: {e}")
        return None

    def _normalize_table_name(self, table: pl.DataFrame) -> pl.DataFrame:
        return table.with_columns(
            pl.col("name")
            .str.to_lowercase()
            .str.strip_chars(" ")
            .alias("name_normalized")
        )

    @property
    def table(self) -> pl.DataFrame:
        if self._is_cache_expired():
            self.logger.info("Cache expired, attempting to update from database")
            fresh_table = self._load_table_from_db()
            if fresh_table is not None:
                self._table = self._normalize_table_name(fresh_table)
                self._save_table_to_disk()
                return self._table
            else:
                self.logger.warning(
                    "Failed to load from database, falling back to cache"
                )

        if self._table is not None:
            return self._table

        self.logger.info("Table not loaded, attempting to load from disk")
        disk_table = self._load_table_from_disk()
        if disk_table is not None:
            self._table = self._normalize_table_name(disk_table)
            return self._table

        raise RuntimeError(
            "Table unavailable: failed to load from database, disk cache, and RAM"
        )

    def _normalize_queries(self, queries: List[str]) -> List[str]:
        return list(map(lambda x: x.lower().strip(), queries))

    def get_scores(self, queries: List[str]):
        result = self.table.with_columns(
            pl.col("name_normalized")
            .map_elements(
                lambda x: calculate_score(x, self._normalize_queries(queries)),
                return_dtype=pl.List(
                    pl.Struct(
                        fields=[
                            pl.Field("score", pl.Float64),
                            pl.Field("candidate", pl.String),
                        ]
                    )
                ),
            )
            .alias("query_scores")
        )

        return result

    def get_scores_lazy(self, queries: List[str]):
        builder = self.table.lazy().with_columns(
            pl.col("name_normalized")
            .map_elements(
                lambda x: calculate_score(x, self._normalize_queries(queries)),
                return_dtype=pl.List(
                    pl.Struct(
                        fields=[
                            pl.Field("score", pl.Float64),
                            pl.Field("candidate", pl.String),
                        ]
                    )
                ),
            )
            .alias("query_scores")
        )

        return builder

    def get_joined_scores(self, queries: List[str]):
        table_scores = self.get_scores(queries)
        suffix = "_right"
        joined_table = (
            table_scores.filter(pl.col("parent_id") != 0)
            .join(
                table_scores.filter(pl.col("parent_id") == 0),
                left_on="parent_id",
                right_on="id",
                how="left",
                suffix=suffix,
            )
            .select(
                pl.col("id").alias("model_id"),
                pl.col("name").alias("model_name"),
                pl.col("parent_id").alias("brand_id"),
                pl.col(f"name{suffix}").alias("brand_name"),
                pl.col("query_scores").alias("model_query_scores"),
                pl.col(f"query_scores{suffix}").alias("brand_query_scores"),
            )
        )

        return joined_table

    def get_joined_scores_lazy(self, queries: List[str]):
        builder_scores = self.get_scores_lazy(queries)
        suffix = "_right"
        builder_joined = (
            builder_scores.filter(pl.col("parent_id") != 0)
            .join(
                builder_scores.filter(pl.col("parent_id") == 0),
                left_on="parent_id",
                right_on="id",
                how="left",
                suffix=suffix,
            )
            .select(
                pl.col("id").alias("model_id"),
                pl.col("name").alias("model_name"),
                pl.col("parent_id").alias("brand_id"),
                pl.col(f"name{suffix}").alias("brand_name"),
                pl.col("query_scores").alias("model_query_scores"),
                pl.col(f"query_scores{suffix}").alias("brand_query_scores"),
            )
        )

        return builder_joined

    def get_best_matches(self, queries: List[str], top_n: int = 10):
        joined_scores_df = self.get_joined_scores(queries)

        best_matches_lazy = (
            joined_scores_df.lazy()
            .explode("model_query_scores")
            .explode("brand_query_scores")
            .filter(
                pl.col("model_query_scores").struct.field("candidate")
                != pl.col("brand_query_scores").struct.field("candidate")
            )
            .with_columns(
                (
                    self.similarity_metric(
                        pl.col("model_query_scores").struct.field("score"),
                        pl.col("brand_query_scores").struct.field("score"),
                    )
                ).alias("combined_score")
            )
            .sort(
                [
                    "combined_score",
                    pl.col("model_name").str.len_chars(),
                    pl.col("brand_name").str.len_chars(),
                ],
                descending=[True, True, True],
            )
            .limit(top_n)
        )

        return best_matches_lazy.collect()

    def get_best_matches_lazy(self, queries: List[str], top_n: int = 10):
        builder_joined = self.get_joined_scores_lazy(queries)
        builder_best_matches = (
            builder_joined.explode("model_query_scores")
            .explode("brand_query_scores")
            .filter(
                pl.col("model_query_scores").struct.field("candidate")
                != pl.col("brand_query_scores").struct.field("candidate")
            )
            .with_columns(
                (
                    self.similarity_metric(
                        pl.col("model_query_scores").struct.field("score"),
                        pl.col("brand_query_scores").struct.field("score"),
                    )
                ).alias("combined_score")
            )
            .sort(
                [
                    "combined_score",
                    pl.col("model_name").str.len_chars(),
                    pl.col("brand_name").str.len_chars(),
                ],
                descending=[True, True, True],
            )
            .limit(top_n)
        )

        return builder_best_matches

    @staticmethod
    def product(first, second):
        return first * second

    @staticmethod
    def arithmetic_mean(first, second):
        return (first + second) / 2

    @staticmethod
    def harmonic_mean(first, second):
        return 2 * first * second / (first + second + 1e-6)

    @staticmethod
    def geometric_mean(first, second):
        return np.sqrt(first * second)

    @staticmethod
    def euclidean(first, second):
        return 1 - np.sqrt((first - 1) ** 2 + (second - 1) ** 2)
