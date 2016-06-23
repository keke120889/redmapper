import yaml

def read_simpleconf(filename, defaults=None):
    """
    Docstring goes here
    """
	
	outdict = defaults if defaults is not None else {}
    with open(filename) as f: yaml_data = yaml.load(f)
    for tag in outdict:
        if outdict[tag] is None:
            raise ValueError('A value for the required tag \"' 
                                + tag + '\" must be specified.')
    
    for tag in yaml_data: outdict[tag] = yaml_data[tag]
    
    return outdict