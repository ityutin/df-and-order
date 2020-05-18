from typing import List
import pandas as pd

from df_and_order.df_transform_step import DfTransformStep, DfTransformStepConfig


class DropColsTransformStep(DfTransformStep):
    """
    Simply drops some undesired columns from a dataframe.
    """
    def __init__(self, cols: List[str]):
        super().__init__()

        self._cols_to_drop = cols

    @staticmethod
    def step_last_modified_ts(step_config: DfTransformStepConfig) -> float:
        return 1589728055.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(self._cols_to_drop, axis=1)


class DatesTransformStep(DfTransformStep):
    """
    Converts cols to datetime type
    """
    def __init__(self, cols: List[str]):
        super().__init__()

        self._dates_cols = cols

    @staticmethod
    def step_last_modified_ts(step_config: DfTransformStepConfig) -> float:
        return 1589728055.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._dates_cols:
            df[col] = pd.to_datetime(df[col])

        return df