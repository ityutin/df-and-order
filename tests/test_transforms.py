import pandas as pd
from typing import List

from df_and_order.df_transform_step import DfTransformStep


class ZeroTransformStep(DfTransformStep):
    def __init__(self, zero_cols: List[str]):
        super().__init__()

        self._zero_cols = zero_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._zero_cols:
            df[col] = 0

        return df


class DatesTransformStep(DfTransformStep):
    def __init__(self, dates_cols: List[str]):
        super().__init__()

        self._dates_cols = dates_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._dates_cols:
            df[col] = pd.to_datetime(df[col])

        return df
