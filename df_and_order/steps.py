import pandas as pd
from typing import List

from df_and_order.df_transform_step import DfTransformStep


class DropColsTransformStep(DfTransformStep):
    """
    Example built-in transform that simply drops some
    undesired columns from a dataframe.
    """
    def __init__(self, cols_to_drop: List[str]):
        super().__init__()

        self._cols_to_drop = cols_to_drop

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(self._cols_to_drop, axis=1)
