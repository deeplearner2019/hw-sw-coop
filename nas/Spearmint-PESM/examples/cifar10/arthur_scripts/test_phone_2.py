"""
This script aims at sending a unique command to AWS.
It commands to train a given model in the first GPU, ie number 0.
"""
import ssm
import time as t
import buckets
import json
import subprocess
import os
import time
import glob
import csv

# Defining the variables for AWS.
PATH_TO_HACONE = '/home/arthur/Documents/hacone'
instance_id = 'i-08163c3c8269d7a3a' #p2.8xlarge
public_DNS = 'ec2-18-208-184-58.compute-1.amazonaws.com'
private_key = PATH_TO_HACONE+"/AWS/arthur_key.pem"
ANDROID_NDK_ROOT = '/home/arthur/Qualcomm/Hexagon_SDK/3.3.3/tools/android-ndk-r14b'
TENSORFLOW_HOME = '/home/arthur/anaconda3/envs/python2/lib/python2.7/site-packages/tensorflow_gpu-1.5.0.dist-info'
debug = True
nb_gpus_to_use = 2
gpus_to_use = '4,5'

def exists_remote(file_path):
    status = subprocess.check_output(['ssh -i {} ubuntu@{} \'if [ -e {} ]; then echo \'yes\'; else echo \'no\'; fi\''.format(private_key,public_DNS, file_path)],shell=True)
    print('status :{}'.format(status))
    if status == 'yes\n':
        return True
    if status == 'no\n':
        return False

def listen_to_remote_process(file_path):
    timer = 0
    while True:
        if debug:
            time.sleep(30)
        else:
            time.sleep(150)
        timer = timer + 1
        print('Searching for the result file {}'.format(file_path))
        if exists_remote(file_path) or timer > 72: # more than 3 hours
            print('Timeout: waiting for accuracy file for too long'*(timer==72))
            break

def lock_instance_gpu(command,filename,file_to_get_back):
    cmd = ['import utils','import ssm','command_id = ssm.run_command(\'{0}\',{1})','utils.listen_to_remote_process(\'{2}\',\'{3}\',\'{4}\',debug={5})']
    python_script = '\n'.join(cmd)
    python_script = python_script.format(instance_id,command,file_to_get_back,private_key,public_DNS,debug)

    python_file = os.path.join(PATH_TO_HACONE,'movidius','command_files',filename+'.py')
    command_file = os.path.join(PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files',filename+'.sh')

    cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; cd {0}/movidius; export PYTHONPATH={0}/movidius:$PYTHONPATH;python {1}; sleep 5) 9>/var/lock/p28_2'.format(PATH_TO_HACONE,python_file)

    with open(python_file,'w') as f:
        f.write(python_script)
    with open(command_file,'w') as f:
        f.write(cmd_s)
    #subprocess.call('sh {0}; rm {0}; rm {1}'.format(command_file,py_file), shell = True)
    subprocess.call('sh {}'.format(command_file), shell = True)

def create_name(dico):
    dico = json.loads(dico)
    name=''
    for key, value in dico.items():
        name+='_'
        name+=str(value)
    return name

def main(job_id, params):
    job_id = str(job_id)
    command = []
    print(params)
    my_params={}
    # Encode params to pass it through the run_command
    for key in params.keys():
        my_params[key.replace('"','\'')] = int(params[key])
    my_params = json.dumps(my_params)
    dico_to_save = {}
    dico_to_save['platform'] = 'Phone'
    dico_to_save['job']= job_id
    dico_to_save['params'] = my_params
    name = create_name(my_params)

    done_jobs = glob.glob('{}/Spearmint-PESM/examples/cifar10/output_0/_*'.format(PATH_TO_HACONE))
    trained_jobs = [job.split('/')[-1] for job in done_jobs]

    if debug:
        max_number_of_steps = 100
    else :
        max_number_of_steps = 28125

    if name not in trained_jobs:
        file_to_send = "{}/jobs/job{}.txt".format(PATH_TO_HACONE,name)
        file_to_get_back = '/home/ubuntu/accuracy_{}.txt'.format(job_id)
        with open(file_to_send, 'wb') as fp:
            json.dump(dico_to_save, fp)

        #check if job number X isn't already stored on the instance
        remote_output_dir = '/home/ubuntu/outputs/cifar10_nns/{}'.format(job_id)
        if exists_remote(remote_output_dir):
            ssm.run_command(instance_id,['sudo rm -rf ' + remote_output_dir])

        # now we need to send this file : job_file to the AWS instance.
        subprocess.call("scp -i {} {} ubuntu@{}:~/job_file_{}.txt".format(private_key, file_to_send, public_DNS, job_id), shell=True)
        command.append("export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH")

        # Launch the train and eval script
        command.append("./train_eval_image_classifier_bis.py {} {} --max_number_of_steps={} --num_clones={}".format(gpus_to_use, job_id, max_number_of_steps, nb_gpus_to_use))
        print(command)

        #Run the command to launch the training and waiting for it to finish
        lock_instance_gpu(command,'lock_instance'+name,file_to_get_back) #Allow us to be the only one to use the instance gpu

        #Get back the accuracy and checkpoints of the model
        subprocess.call("scp -i {} -r ubuntu@{}:~/outputs/cifar10_nns/{}/ {}/Spearmint-PESM/examples/cifar10/output_0/{}/".format(private_key, public_DNS, job_id, PATH_TO_HACONE, name), shell=True)
        subprocess.call("scp -i {} ubuntu@{}:~/accuracy_{}.txt {}/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt".format(private_key, public_DNS, job_id,PATH_TO_HACONE, name), shell=True)
        output_dir = os.path.join('outputs',str(job_id))

        with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt'.format(name)) as my_file:
            dico = json.load(my_file)

    else:
        print("Model already trained")
        subprocess.call("scp -r arturo@10.217.128.217:/home/data/arturo/models_trained/{1} {0}/Spearmint-PESM/examples/cifar10/output_0/{1}".format(PATH_TO_HACONE,name), shell = True)
        with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt'.format(name)) as my_file:
            dico = json.load(my_file)

    #Export the graph to a protobuf file
    cmd = "python {0}/tensorflow/nn_search/export_inference_graph_movidius.py \
    --job_name=cifar10_phone \
    --name_job={1} \
    --output_file={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/inference_graph.pb \
    --PATH_TO_HACONE={0}".format(PATH_TO_HACONE,name)

    cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; {}) 9>/var/lock/gpu'.format(cmd) #Allow us to execute only one command on the GPU at a time

    command_file = os.path.join(PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files','pb_'+name+'.sh')
    with open(command_file,'w') as f:
        f.write(cmd_s)

    subprocess.call('sh {0}; rm {0}'.format(command_file), shell = True)

    #Freeze the weights in the graph in a protobuf file
    cmd = "python {0}/tensorflow/nn_search/freeze_graph_16.py \
    --input_graph={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/inference_graph.pb \
    --input_checkpoint={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/model.ckpt-{2} \
    --input_binary=true \
    --output_graph={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/{1}.pb \
    --output_node_names=CifarNet/Predictions/Reshape_1".format(PATH_TO_HACONE,name,max_number_of_steps)

    cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; {}) 9>/var/lock/gpu'.format(cmd) #Allow us to execute only one command on the GPU at a time

    command_file = os.path.join(PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files','fr_'+name+'.sh')
    with open(command_file,'w') as f:
        f.write(cmd_s)

    subprocess.call('sh {0}; rm {0}'.format(command_file), shell = True)

    #Measure power and inference time on the phone
    subprocess.call("cp {0}/Spearmint-PESM/examples/cifar10/output_0/{1}/{1}.pb {0}/snpe-sdk/models/cifarnet/tensorflow/".format(PATH_TO_HACONE,name),shell=True)
    subprocess.call("cd {0}/snpe-sdk; python ./models/cifarnet/scripts/setup_cifarnet.py -S {0}/snpe-sdk -A {3} -t {4} -a ./models/cifarnet/data -f {1} {2}".format(PATH_TO_HACONE, name, '-d'*debug, ANDROID_NDK_ROOT, TENSORFLOW_HOME), shell=True)

    dico_hardware = {}

    #Retrieve inference time in us
    stats_file = '{}/snpe-sdk/benchmarks/cifarnet/benchmark/{}/latest_results/benchmark_stats_CifarNet.csv'.format(PATH_TO_HACONE,name)
    timer=0
    while not os.path.exists(stats_file) and timer < 120:
        time.sleep(30)
	timer += 1

    with open(stats_file) as fp:
        reader = csv.reader(fp, delimiter=',')
        for row in reader:
            if 'Total Inference Time' in row:
                inference_time_us = float(row[3]) #in micro-seconds
                dico_hardware['time'] = inference_time_us/1000000 #in seconds
            elif 'energy [J]' in row:
                energy_joules = float(row[3])
                dico_hardware['power'] = energy_joules

    #Retrieve accuracy
    with open('{}/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt'.format(PATH_TO_HACONE,name)) as acc:
        dico = json.load(acc)
        dico_hardware['f'] = dico['f']

    with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/hardware_metrics.txt'.format(name), 'w') as my_file:
        json.dump(dico_hardware, my_file)
    return dico_hardware
