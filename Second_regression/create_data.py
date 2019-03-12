import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import sys
import pickle
import os
import shutil

def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 3)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def pulse(t, carr_fre, first_order = 0., second_order = 0.):
    amp = 1.0
    pulse_duration = 4.66
    t_center = 5. * pulse_duration
    shape = amp * np.exp( - (t - t_center)**2 / pulse_duration**2)
    phase = carr_fre * (t - t_center) + first_order * (t - t_center)**2 \
            + second_order * (t - t_center)**3
    pulse = shape * np.cos(phase)
    return pulse


def pulse_roof(t, carr_fre, first_part = 0., second_part = 0.):
    amp = 1.0
    pulse_duration = 4.66
    t_center = 5. * pulse_duration
    shape = amp * np.exp( - (t - t_center)**2 / pulse_duration**2)
    first_order = first_part if t < t_center else second_part
    phase = carr_fre * (t - t_center) + first_order * (t - t_center)**2 
    pulse = shape * np.cos(phase)
    return pulse


def create_data(kind, num, labels = {}, output_path = 'classifier_data', seed = 13):
    np.random.seed(seed)
    epsilon = 1e-5
    count = 0
    time = np.linspace(0, 50, 2500)
    first_part = 0.0
    second_part = 0.0
    linear_chirp = 0.0
    second_chirp = 0.0
    if  os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.makedirs(output_path)
    else:
        os.makedirs(output_path)
    while count < num:
        carr_fre = np.random.uniform(low = 0, high = 10)
        if carr_fre < epsilon: continue
        if kind == 'linear':
            linear_chirp = np.random.uniform(low = -10, high = 10)
            signal = [pulse(t, carr_fre, first_order = linear_chirp) for t in time]
        elif kind == 'second':
            second_chirp = np.random.uniform(low = -5, high = 5)
            if second_chirp < epsilon: continue
            linear_chirp = np.random.uniform(low = -10, high = 10)
            signal = [pulse(t, carr_fre, first_order = linear_chirp, 
                            second_order = second_chirp) for t in time]
        elif kind == 'roof':
            first_part = np.random.uniform(low = -10, high = 10)
            second_part = np.random.uniform(low = -10, high = 10)
            if np.abs(first_part - second_part) < epsilon: continue
            signal = [pulse_roof(t = t, carr_fre = carr_fre, 
                      first_part = first_part, 
                      second_part = second_part) for t in time]
        else:
            raise KindError('Kind needs to be linear, second or roof!')
        signal = np.array(signal)
        thisID = kind + '_' + str(count)
        np.save(os.path.join(output_path, thisID), signal)
        labels[thisID] = {'roof':[carr_fre, first_part, second_part], 
                          'linear':[carr_fre, linear_chirp], 
                          'second':[carr_fre, linear_chirp, second_chirp]}[kind]
        count += 1
        progress(count, num, status = 'Working on ' + kind + '.')
    print('Done!')
    return labels
        

if __name__ == "__main__":
    labels = {}
    output_path = 'regression_data'
    total_num = 50000

    labels = create_data('second', total_num, labels = labels, output_path = output_path)
    with open(os.path.join(output_path, 'labels.pickle'), 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
