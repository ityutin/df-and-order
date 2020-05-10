import importlib
from typing import Optional


def build_class_instance(module_path: str, init_params: Optional[dict] = None):
    class_ = get_type_from_module_path(module_path=module_path)
    result = class_(**(init_params or {}))
    return result


def get_type_from_module_path(module_path: str):
    components = module_path.split('.')
    class_name = components[-1]
    module_name = '.'.join(components[:-1])

    module = importlib.import_module(module_name)
    class_type = getattr(module, class_name)
    return class_type
