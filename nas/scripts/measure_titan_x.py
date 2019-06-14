#!/home/arturo/tensorflow-py27/bin/python2

from pynvml import nvmlDeviceGetHandleByIndex
from pynvml import nvmlInit
from pynvml import nvmlDeviceGetMemoryInfo
from pynvml import nvmlDeviceGetPowerUsage
from pynvml import nvmlShutdown

import threading
import logging
import time
import json
import subprocess
import os
import sys
import pickle

PATH_TO_HACONE = '/mnt/sdh/lile/Projects/hacone'

dico = {}
power_list = []

def inference_task(model_name, case):
    """
    This function defines the main task, i.e. performing inference for a given number of batches.
    It only calls for the predict function defined in the inference.py script.
    """
    logging.debug('Starting')

    subprocess.call('python {}/scripts/inference_titan_x.py {} {}'.format(PATH_TO_HACONE, model_name, case), shell=True)


def monitoring_task(inference_thread):
    """
    This function defines the background action, i.e. monitoring the GPU  memory and power usage
    for a given number of batches.
    For now it only listens to the GPU indexed 0.
    I will have to make it more flexible.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(3)
    
    while True:
        try:
            power = float(nvmlDeviceGetPowerUsage(handle)) / 1000 # convert to W
            power_list.append(power)
	except:
            pass
        time.sleep(0.02)
        
        if not inference_thread.is_alive():
           nvmlShutdown()
           break 
        
def get_start_end(power_list, thresh = 80):
    ids = [i for i, p in enumerate(power_list) if p > thresh]
    return min(ids), max(ids)

def integrate_power(power_list, start, end):
	energy = 0
	for i in range(start, end):
	    energy = energy + power_list[i]
	energy = energy * 0.02
	return energy
    
def main(model_name, case):  
    inference_thread = threading.Thread(target = inference_task, kwargs ={'model_name':model_name,'case':case})
    monitoring_thread = threading.Thread(target = monitoring_task, kwargs = {'inference_thread':inference_thread})

    inference_thread.start()
    monitoring_thread.start()

    inference_thread.join()
    monitoring_thread.join()

    #print('power list is:{}'.format(power_list))
    
    start, end = get_start_end(power_list, thresh = 80)
    inference_time = (end - start) * 0.02

    energy = integrate_power(power_list, start, end)
    dico['time'] = inference_time
    dico['energy'] = energy
    
    print("integration time: {}\n".format(inference_time))
    print("integration energy: {}\n".format(energy))
    save_dir = PATH_TO_HACONE + '/measurements/{}'.format(case)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, model_name + '.txt'),'w') as f:
        json.dump(dico, f)

    pickle.dump(power_list, open(os.path.join(save_dir, model_name + '_power_list.pkl'),'wb'))


if __name__ == '__main__':
    model_name = sys.argv[1]
    case = sys.argv[2]

    save_dir = PATH_TO_HACONE + '/measurements/{}'.format(case)

    if not os.path.exists(save_dir):
       os.makedirs(save_dir)
    main(model_name, case)
