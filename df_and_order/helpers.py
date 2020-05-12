import importlib
from typing import Optional


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
    components = module_path.split('.')
    class_name = components[-1]
    module_name = '.'.join(components[:-1])

    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)
    return class_type
