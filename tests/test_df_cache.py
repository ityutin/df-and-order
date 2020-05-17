import pandas as pd
import pytest

from df_and_order.df_cache import DfCache


@pytest.fixture()
def save_args():
    return [1, 2, 3]


@pytest.fixture()
def save_kwargs():
    return {
        'a': 1,
        'b': 2
    }


@pytest.fixture()
def load_args():
    return [4, 5, 6]


@pytest.fixture()
def load_kwargs():
    return {
        'c': 3,
        'd': 4
    }


class TestDfCache(DfCache):
    def _save(self, df: pd.DataFrame, path: str, *args, **kwargs):
        pass

    def _load(self, path: str, *args, **kwargs) -> pd.DataFrame:
        pass


@pytest.fixture()
def instance(save_args, save_kwargs, load_args, load_kwargs):
    result = TestDfCache(save_args=save_args,
                         save_kwargs=save_kwargs,
                         load_args=load_args,
                         load_kwargs=load_kwargs)
    return result


def test_save(mocker, instance, save_args, save_kwargs):
    save_mock = mocker.patch.object(instance, '_save')
    df = mocker.Mock()
    path = '/some/path'
    instance.save(df=df, path=path)

    save_mock.assert_called_with(df=df,
                                 path=path,
                                 *save_args,
                                 **save_kwargs)


def test_load(mocker, instance, load_args, load_kwargs):
    load_mock = mocker.patch.object(instance, '_load')
    path = '/some/path'
    instance.load(path=path)

    load_mock.assert_called_with(path=path,
                                 *load_args,
                                 **load_kwargs)
