import importlib
import inspect
import os
from typing import Optional, Tuple


def build_class_instance(module_path: str, init_params: Optional[dict] = None):
    """
    Create an object instance from absolute module_path string.

    Parameters
    ----------
    module_path: str
        Full module_path that is valid for your project or some external package.
    init_params: optional dict
        These parameters will be used as init parameters for the given type.

    Returns
    -------
    Some object instance
    """
    class_ = get_type_from_module_path(module_path=module_path)
    result = class_(**(init_params or {}))
    return result


def get_type_from_module_path(module_path: str):
    """
    Takes full module path and takes the class
    name to convert it to Python type.

    Parameters
    ----------
    module_path: str
        Full module_path that is valid for your project or some external package.

    Returns
    -------
    Python type.
    """
    module_name, class_name = split_module_path(module_path=module_path)

    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)
    return class_type


def split_module_path(module_path: str) -> Tuple[str, str]:
    """
    Splits module_path strings into a module name and a class name

    Parameters
    ----------
    module_path: str
        Full module_path that is valid for your project or some external package.

    Returns
    -------
    Tuple of strings, one for the module name, second for the class name
    """
    components = module_path.split('.')
    class_name = components[-1]
    module_name = '.'.join(components[:-1])

    return module_name, class_name


def get_file_path_from_module_path(module_path: str) -> str:
    """
    Returns absolute file path given the module path

    Parameters
    ----------
    module_path: str
        Full module_path that is valid for your project or some external package.

    Returns
    -------
    Absolute file path for module path
    """
    module_name, _ = split_module_path(module_path=module_path)
    module = importlib.import_module(module_name)
    file_path = module.__file__
    return file_path


def get_module_path_from_type(py_type) -> str:
    """
    Gets full module path + class name by the given class type

    Parameters
    ----------
    cls: python type
        Class you want to get full module path for

    Returns
    -------
    String representation of full module path like module1.submodule2.ClassName
    """
    result = inspect.getmodule(py_type).__name__ + '.' + py_type.__name__
    return result


class FileInspector:
    @staticmethod
    def last_modified_date(file_path: str) -> float:
        """
        Returns the last modification date unix timestamp for a file at the given path

        Parameters
        ----------
        file_path: str
            Absolute path for any file

        Returns
        -------
        Unix timestamp of the file's last modification date
        """
        result = os.path.getmtime(file_path)
        return result
