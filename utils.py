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


def find_filename(base_filename):
    i = 1
    while True:
        file_name = base_filename + "_" + "run" + "_" + str(i)
        if not os.path.exists("./runs/" + file_name + ".csv"):
            return file_name
        i += 1
