"""
Define the system utils

"""
from logging import warning


def merge_dicts(*dict_args):
    """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
    result = {}
    for dictionary in dict_args:
        if dictionary is not None:
            result.update(dictionary)
    return result


def update_source_dict_by_given_dict(source, target):
    """
    Update the source dict by target dict, only update the values stored in target

    Parameters
    ----------
    target
    source

    Returns
    -------

    """
    if not isinstance(target, dict):
        warning('target is not a dict')
        return source
    return update_source_dict_by_given_kwargs(source, **target)


def update_source_dict_by_given_kwargs(source, **target_kwargs):
    """

    Parameters
    ----------
    source
    target_kwargs

    Returns
    -------

    """
    source = dict(source)
    for entry in target_kwargs:
        if source.has_key(entry):
            source[entry] = target_kwargs[entry]
        else:
            warning("key not found {}".format(entry))
    return source
