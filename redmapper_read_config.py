import yaml

def read_simpleconf(filename, defaults=None, required=False):
    """
    Docstring goes here
    """

    if required and (defaults is None):
        print "Error: if required=True defaults must be an input dictionary."
        return {}

    outdict = defaults if defaults is not None else {}

    with open(filename) as f:
    	yaml_data = yaml.load(f)
    	for tag in yaml_data: outdict[tag] = yaml_data[tag]

    return outdict
