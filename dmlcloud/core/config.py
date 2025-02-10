import importlib
from functools import partial, update_wrapper
from typing import Any, Callable, Mapping


__all__ = [
    'import_object',
    'factory_from_cfg',
    'obj_from_cfg',
]


def import_object(object_path: str) -> Any:
    """
    Imports an object from a module.

    The object path should be in the form of "module.submodule.object".
    The function imports the module and returns the object.

    Args:
        object_path (str): The path to the object to import.

    Returns:
        Any: The imported object.

    Raises:
        ImportError: If the module containing the object cannot be imported or the object cannot be found.

    Example:
        >>> import_object("dmlcloud.core.stage.Stage")
        <class 'dmlcloud.core.stage.Stage'>
    """

    module_name, obj_name = object_path.rsplit(".", 1)
    module = importlib.import_module(module_name)

    try:
        return getattr(module, obj_name)
    except AttributeError as e:
        raise ImportError(f"Object '{obj_name}' not found in module '{module_name}'") from e


def factory_from_cfg(config: Mapping | str, *args, **kwargs) -> Callable:
    """
    Creates a factory function from a configuration dictionary or a string.

    If a string is provided, it is assumed to be the path to the factory function (or class).

    If a dictionary is provided, it must contain a "factory" key with the path to the factory function.
    Additional keys in the dictionary are passed as keyword arguments to the factory function.

    Args:
        config (Mapping | str): Configuration dictionary or string with the path to the factory function.
        *args: Additional positional arguments to pass to the factory function.
        **kwargs: Additional keyword arguments to pass to the factory function.

    Returns:
        Callable: A factory function with the provided configuration and arguments.

    Raises:
        ImportError: If the factory function cannot be imported.
        KeyError: If the configuration dictionary does not contain the "factory" key.

    Example:
        >>> factory = dml.factory_from_cfg('datetime.date', 2025, month=1, day=1)
        >>> factory
        <class 'datetime.date'>
        >>> factory()
        datetime.date(2025, 1, 1)
        >>> factory(month=12, day=31)
        datetime.date(2025, 12, 31)

        Instead of providing a string, you can also use a configuration dictionary:

        >>> config = {'factory': 'datetime.date', 'year': 2025, 'month': 1, 'day': 1}
        >>> factory = dml.factory_from_cfg(config)
        >>> factory()
        datetime.date(2025, 1, 1)
    """

    if isinstance(config, str):
        factory = import_object(config)
        kwargs = kwargs.copy()
    else:
        factory = import_object(config['factory'])
        merged_kwargs = config.copy()
        merged_kwargs.update(kwargs)
        kwargs = merged_kwargs
        del kwargs['factory']

    wrapper = partial(factory, *args, **kwargs)
    return update_wrapper(wrapper, factory)


def obj_from_cfg(config: Mapping | str, *args, **kwargs) -> Any:
    """
    Creates an object from a configuration dictionary or a string.

    This is equivalent to calling `factory_from_cfg(config)(*args, **kwargs)`.

    If a string is provided, it is assumed to be the path to the object (class).
    If a dictionary is provided, it must contain a "factory" key with the path to the object.

    Additional keys in the dictionary are passed as keyword arguments to the object constructor.

    Args:
        config (Mapping | str): Configuration dictionary or string with the path to the object.
        *args: Additional positional arguments to pass to the object constructor.
        **kwargs: Additional keyword arguments to pass to the object constructor.

    Returns:
        Any: The created object.

    Raises:
        ImportError: If the object cannot be imported.
        KeyError: If the configuration dictionary does not contain the "factory" key.

    Example:
        >>> dml.obj_from_cfg('datetime.date', 2025, month=1, day=1)
        datetime.date(2025, 1, 1)
    """

    return factory_from_cfg(config)(*args, **kwargs)
