import json

# Returns the configuration from the given path
def get_configuration(path_to_conf):
    print("[CONF] Loading configuration file {}".format(path_to_conf))

    conf = json.load(open(path_to_conf, "r"))

    return conf