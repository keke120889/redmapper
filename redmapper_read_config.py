import yaml

def read_yaml(filename, defaults=None):
    
    """
    Name:
        read_yaml
    Purpose:
        Read in a YAML file with key/value pairs and put into dict
    Calling Sequence:
        outdict = read_simpleconf(filename, defaults=None)
    Inputs:
        filename: configuration file
    Optional Inputs:
        defaults: Dictionary of required/default fields/values. All fields in 
                  defaults are required to be specified. If they are set to 
                  None, they must be specified by filename.
    """
	
    outdict = defaults if defaults is not None else {}
    
    with open(filename) as f: yaml_data = yaml.load(f)
    for tag in outdict:
        if outdict[tag] is None:
            raise ValueError('A value for the required tag \"' 
                                + tag + '\" must be specified.')
    
    for tag in yaml_data: outdict[tag] = yaml_data[tag]
    
    return outdict