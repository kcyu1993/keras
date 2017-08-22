import os
from configobj import ConfigObj


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
            config.write(f)

    @classmethod
    def load_config_from_file(cls, infile):
        config = ConfigObj(infile, raise_errors=True)
        return cls(**config.dict())