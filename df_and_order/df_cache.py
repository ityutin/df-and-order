from abc import ABC, abstractmethod
from typing import Optional

from df_and_order.common import DF_TYPE


class DfCache(ABC):
    """
    Encapsulates logic how to save a dataframe to disk and how to load it afterwards.

    Init parameters
    ----------
    save_args: list
        Custom args for save method you may want to pass to your implementation.
    save_kwargs: dict
        Custom kwargs for save method you may want to pass to your implementation.
    load_args: list
        Custom args for load method you may want to pass to your implementation.
    load_kwargs: dict
        Custom kwargs for load method you may want to pass to your implementation.
    """

    def __init__(
        self,
        save_args: Optional[list] = None,
        save_kwargs: Optional[dict] = None,
        load_args: Optional[list] = None,
        load_kwargs: Optional[dict] = None,
    ):
        self._save_args = save_args or []
        self._save_kwargs = save_kwargs or {}
        self._load_args = load_args or []
        self._load_kwargs = load_kwargs or {}

    def save(self, df: DF_TYPE, path: str):
        self._save(df=df, path=path, *self._save_args, **self._save_kwargs)  # type: ignore

    def load(self, path: str) -> DF_TYPE:
        return self._load(path=path, *self._load_args, **self._load_kwargs)  # type: ignore

    @abstractmethod
    def _save(self, df: DF_TYPE, path: str, *args, **kwargs):
        pass

    @abstractmethod
    def _load(self, path: str, *args, **kwargs) -> DF_TYPE:
        pass
