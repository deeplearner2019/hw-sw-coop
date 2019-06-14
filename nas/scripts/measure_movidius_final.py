"""
@author: lile
@brief: 
"""

from pymongo import MongoClient
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os
from spearmint.utils.database.mongodb import MongoDB
import json
import pickle
from pynvml import *
import time
from nn_search import eval_image_classifier 
import threading
import subprocess
import csv

PATH_TO_HACONE =  '/home/lile/Projects/git_repo/hacone'
PATH_TO_HACONE_LOCAL = PATH_TO_HACONE

start_button = '783 584'
stop_button = '856 582'
export_csv = '784 626'
ok_export = '531 495'
close_graph = '182 188'

user_id_windows = 'User'
ip_windows = '10.217.132.138'

sample_interval = 0.1

def get_power(file):
    '''
    Read the power measurement on the phone during the inference.
    Input: String, path of the file containing the phone measurements
    Return: List[float], power consumption
    '''
    power = []
    with open(file) as fp:
        reader = csv.reader(fp)
        counter = 0
        for row in reader:
            if counter > 1:
                power.append(float(row[3]))
            counter += 1
    return power

def get_start_end(power_list, thresh = 0.45):
    
    ids = [i for i, p in enumerate(power_list) if p > thresh]
    return min(ids), max(ids)



def integrate_power(power_list, start, end):
	energy = 0
	for i in range(start, end):
	    energy = energy + power_list[i]
	energy = energy * sample_interval
	return energy
   
def compute_energy(power):
    '''
    Compute the energy consumption on the phone during the inference. Return the total energy - idle energy.
    Input: String, path of the phone measurements
    Return: float, energy consumption
    '''
    idle_n = int(5.0 / sample_interval)
    mean_value = sum([float(p) for p in power[:idle_n]])/idle_n
    power_0 = [float(p) - mean_value for p in power]
    print(mean_value)
    energy = sum(power_0)
    energy = energy * sample_interval
    return energy

def lock_process(cmd,filename,lock_name):
    '''
    Lock a process. If another process with the same lock_name is running, wait until it finishes. Create a commands file to execute when it is allowed to.
    Input:
        - cmd: String, the commands to execute and protect
        - filename: String, name of the commands file to be created
        - lock_name: String, name of the category of process
    '''
    #cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; {}; sleep 5) 9>/var/lock/{}'.format(cmd,lock_name) #Allow us to execute only one command on the GPU at a time
    cmd_s = cmd    
    command_file = os.path.join(PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files',filename+'.sh')
    
    save_dir = os.path.join(PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files')
    if not os.path.exists(save_dir):
     os.makedirs(save_dir)
     
    with open(command_file,'w') as f:
        f.write(cmd_s)
    subprocess.call('bash {0}'.format(command_file), shell = True)
    

def test_movidius(model_name, case, N, F, iteration):
    PATH_TO_HACONE =  '/home/lile/Projects/git_repo/hacone'
    PATH_TO_HACONE_LOCAL = PATH_TO_HACONE

    start_button = '783 584'
    stop_button = '856 582'
    export_csv = '784 626'
    ok_export = '531 495'
    close_graph = '182 188'

    user_id_windows = 'User'
    ip_windows = '10.217.132.138'
    
    def get_power(file):
        power = []
        with open(file) as fp:
            reader = csv.reader(fp)
            counter = 0
            for row in reader:
                if counter > 1:
                    power.append(float(row[3]))
                counter += 1
        return power

    def get_start_end(power_list, thresh = 0.45): 
        ids = [i for i, p in enumerate(power_list) if p > thresh]
        return min(ids), max(ids)



    def integrate_power(power_list, start, end):
	    energy = 0
	    for i in range(start, end):
	        energy = energy + power_list[i]
	    energy = energy * sample_interval
	    return energy

    final_model_name = model_name + '_N{}_F{}'.format(N, F)

    subprocess.call("python {0}/tensorflow/nn_search/export_inference_graph_movidius_final.py {1} {2} {3}".format(PATH_TO_HACONE_LOCAL, model_name, N, F),shell=True)
 
     #Freeze the weights in the graph in a protobuf file
    subprocess.call("python {0}/tensorflow/nn_search/freeze_graph_14.py \
     --input_graph={0}/outputs/final_models/{1}/inference_graph.pb \
     --input_checkpoint={0}/outputs/final_models/{1}/model.ckpt-{2} \
     --input_binary=true \
     --output_graph={0}/outputs/final_models/{1}/frozen_graph.pb \
     --output_node_names=CifarNet/Predictions/Reshape_1".format(PATH_TO_HACONE_LOCAL, final_model_name, iteration),shell=True)

    dico_hardware = {}

    path_to_movidius = '/home/lile/Projects/git_repo/hacone/movidius'
    model_dir = '{}/outputs/final_models/{}'.format(PATH_TO_HACONE_LOCAL, final_model_name)
    log_file = '{}/measurements/{}/{}.profile'.format(PATH_TO_HACONE_LOCAL, case, final_model_name)
    #Get back the power and inference time measurement on the Movidius
    #Start and stop monitoring power
    #cmd_00 = "WID=$(xdotool search --name windows-astar | head -1); xdotool windowfocus $WID"
    cmd_00 = "sleep 1; WID=$(xdotool search --name 10.217.132.138 | head -1); xdotool windowactivate $WID".format(ip_windows)
    cmd_0 = "xdotool mousemove {} click 1".format(start_button)#click on Start
    cmd_1 = "sleep 1"
    cmd_2 = "python3 {}/mvNCProfile.py -s 12 -network {}/frozen_graph.pb -in input -on CifarNet/Predictions/Reshape_1 >{}".format(path_to_movidius, 
             model_dir, log_file) #Launch the model on Movidius
    cmd_3 = "sleep 5"
    cmd_4 = "xdotool windowfocus $WID; xdotool mousemove {} click 1".format(stop_button) #click on Stop
    cmd_5 = "sleep 1; xdotool windowfocus $WID; xdotool mousemove {} click 1".format(export_csv) #click on export
    cmd_6 = "sleep 1; xdotool windowfocus $WID; xdotool mousemove {};  xdotool click 1".format(ok_export) #click on OK
    cmd_61 = "sleep 1; xdotool windowfocus $WID; xdotool mousemove {};  xdotool click 1".format(close_graph) #click on close graph
    cmd_7 =  "ssh {}@{} \'filename=$(ls C:/Users/User/movidius -t | head -1); sleep 1; mv \"C:/Users/User/movidius/$filename\" C:/Users/User/movidius/{}.csv\';echo \"file done\"".format(user_id_windows, ip_windows,model_name) # Get back the measurements
    cmd_8 = "sleep 2"
    cmd = ' ; '.join([cmd_00, cmd_0, cmd_1, cmd_2, cmd_3, cmd_4,cmd_5, cmd_6, cmd_61, cmd_7, cmd_8])
    lock_process(cmd, model_name, 'movidius')

    #Retrieve inference time   
    with open(log_file) as my_file:
        for line in my_file:
            if 'Total inference time' in line:
                time_inference = float(line.replace(' ','').split('Totalinferencetime')[1].split('\n')[0])
                dico_hardware['time'] = time_inference

    #Retrieve power
    result_file = '{}/measurements/{}/{}.csv'.format(PATH_TO_HACONE_LOCAL, case, final_model_name)
    
    subprocess.call('scp {}@{}:\'C:/Users/User/movidius/{}.csv\' {}'.format(user_id_windows, ip_windows, model_name, result_file), shell = True)
    
    power_list = get_power(result_file)
    start, end = get_start_end(power_list)
    integration_energy = integrate_power(power_list, start, end)
    dico_hardware['energy'] = integration_energy

    print("inference time: {}\n".format(time_inference))
    print("integration energy: {}\n".format(integration_energy))

    #Save results
    with open(PATH_TO_HACONE_LOCAL + '/measurements/{}/{}.txt'.format(case, final_model_name), 'w') as my_file:
        json.dump(dico_hardware, my_file)

    return dico_hardware

if __name__ == '__main__':
    model_name = sys.argv[1]
    case = sys.argv[2]
    N = int(sys.argv[3])
    F = int(sys.argv[4])
    iteration = int(sys.argv[5])
   

    measure_dir = PATH_TO_HACONE + '/measurements/{}'.format(case)
    if not os.path.exists(measure_dir):
       os.makedirs(measure_dir)

    test_movidius(model_name, case, N, F, iteration) 

