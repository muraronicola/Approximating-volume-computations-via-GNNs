import os

def mean_relative_error(out, y):
    error = 0
    for i in range(len(out)):
        error += abs(out[i] - y[i]) / y[i]
    
    return error / len(out)


def mean_squared_error(out, y):
    se_global = 0
    
    for i in range(len(out)):
        se = (out[i] - y[i])**2
        se_global += se
        
    return se_global / len(out)


def mean_absolute_error(out, y):
    error_global = 0
    
    for i in range(len(out)):
        error = abs(out[i] - y[i])
        error_global += error
    
    return error_global / len(out)


def get_run_id(conf_path):
    i = 0
    found = False
    
    while not found:
        i += 1
        file_name = conf_path + "conf_{}.json".format(i)
        
        if not os.path.exists(file_name):
            found = True

    return i

def generate_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_paths(out_path, base_filename):
    folder_out_conf_path = out_path + base_filename + "/config/"
    folder_out_data_path = out_path + base_filename + "/data/"
    folder_out_model_path = out_path + base_filename + "/model/"
    
    #generate_folder_structure(out_path)
    generate_folder_structure(folder_out_conf_path)
    generate_folder_structure(folder_out_data_path)
    generate_folder_structure(folder_out_model_path)
    
    id_run = get_run_id(folder_out_conf_path)
    
    out_conf_path = folder_out_conf_path + "conf_{}.json".format(id_run)
    out_data_path = folder_out_data_path + "out_{}.csv".format(id_run)
    out_data_model = folder_out_model_path + "model_{}.pt".format(id_run)
    
    return out_conf_path, out_data_path, out_data_model