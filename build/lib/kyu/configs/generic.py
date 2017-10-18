import os
from configobj import ConfigObj
from kyu.utils.dict_utils import load_dict, save_dict


class KCConfig(object):
    """
     define SuperClass Config
     add file system support. Save to the default places ??
    """

    def write_to_config(self, outfile, section=None):
        """ Wrapper with self defined locations if default is not supported. """
        if outfile is None:
            raise ValueError("Cannot pass None to save method ")
        if os.path.exists(outfile):
            os.remove(outfile)
        with open(outfile, 'w') as f:
            config = ConfigObj()
            config.merge(self.__dict__)
            # Remove some parts
            remove_keywords = ['self', 'kwargs']
            for key in remove_keywords:
                if key in config.keys():
                    del config[key]

            config.write(f)

            # export_dict = self.__dict__
            # # Remove some parts
            # remove_keywords = ['self', 'kwargs']
            # for key in remove_keywords:
            #     if key in export_dict.keys():
            #         del export_dict[key]
            # save_dict(export_dict, outfile)

    @classmethod
    def load_config_from_file(cls, infile):
        config = ConfigObj(infile, raise_errors=True, interpolation=True)
        # config = load_dict(infile)
        if 'self' in config.keys():
            del config['self']
        res_dict = load_str_dict_to_original_type(**config)
        return cls(**res_dict)


def load_str_dict_to_original_type(source_dict):
    import ast
    for key, value in source_dict.items():
        if key == 'comments':
            continue

        if isinstance(value, str):
            if value[0] == '<':
                del source_dict[key]
                continue
            source_dict[key] = ast.literal_eval(value)
    return source_dict

