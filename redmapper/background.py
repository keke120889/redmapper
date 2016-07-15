import fitsio
import numpy as np
from catalog import Entry


class Background(Entry):

    """ docstring """

    def __init__(self, *initial_data, **kwargs):
        for data in initial_data:
            field_names = data if isinstance(data, dict) else data.dtype.names
            for key in field_names:
                setattr(self, key, data[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if 'scale' not in kwargs: self.scale = 1.0
        