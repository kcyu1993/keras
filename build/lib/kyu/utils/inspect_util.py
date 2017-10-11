"""
Implement the arguments parsing and default dictionary composition

"""

import inspect
from functools import wraps

import six
from .dict_utils import merge_dicts


def compose_default_call_arguments(func):
    args_spec = inspect.getargspec(func)
    # make the dictionary by
    if len(args_spec.args) == 0:
        # It may be into decorators
        try:
            return dict(args_spec.args)
        except IOError as e:
            raise e

    return dict(zip(
        args_spec.args[len(args_spec.args) - len(args_spec.defaults):],
        args_spec.defaults
    ))


def get_default_args(identifier, module_object=None, custom_object=None, method_name='__init__'):
    """
    General get default args given module objects and identifier

    Parameters
    ----------
    identifier
    module_object
    custom_object
    method_name

    Returns
    -------

    """

    module_object = merge_dicts(module_object, custom_object)
    if isinstance(identifier, six.string_types):
        func = module_object.get(identifier)
    elif callable(identifier):
        func = identifier
    else:
        raise ValueError("secondstat only supports {}".format(module_object))
    if func is None:
        return None
    if hasattr(func, method_name):
        return compose_default_call_arguments(getattr(func, method_name))
    else:
        return None
