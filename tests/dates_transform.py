from typing import List

import pandas as pd

from df_and_order.df_transform_step import DfTransformStep


class TestDatesTransformStep(DfTransformStep):
    def __init__(self, dates_cols: List[str]):
        super().__init__()

        self._dates_cols = dates_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._dates_cols:
            df[col] = pd.to_datetime(df[col])

        return df