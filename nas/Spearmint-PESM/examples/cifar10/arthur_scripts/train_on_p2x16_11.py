"""
This script aims at sending a unique command to AWS.
It commands to train a given model in the first GPU, ie number 11.
"""
import ssm
import buckets
import json
import subprocess
import os
import time

# Defining the variables for AWS.
PATH_TO_HACONE = '/home/arthur/Documents/hacone'
instance_id = 'i-0722eb73d34f2199a' #p2.16xlarge
public_DNS = 'ec2-54-227-227-113.compute-1.amazonaws.com'
private_key = PATH_TO_HACONE+"/AWS/arthur_key.pem"
user = 'arturo'
server_DNS = '10.217.128.217'

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
        time.sleep(600)
        timer = timer + 1
        print('Searching for the result file {}'.format(file_path))
        if exists_remote(file_path) or timer > 15:
            break

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
    dico_to_save['job']= job_id
    dico_to_save['params'] = my_params
    name= create_name(my_params)

    file_to_send = PATH_TO_HACONE + "/jobs/job{}.txt".format(name)
    file_to_get_back = '/home/ubuntu/accuracy_{}.txt'.format(job_id)
    with open(file_to_send, 'wb') as fp:
        json.dump(dico_to_save, fp)

    # now we need to send this file : job_file to the AWS instance.
    subprocess.call("scp -i {} {} ubuntu@{}:~/job_file_{}.txt".format(private_key, file_to_send, public_DNS, job_id), shell=True)

    command.append("export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH")
    # Launch the train and eval script

    command.append("./train_eval_image_classifier_bis.py 11 {} --max_number_of_steps=28125".format(job_id))
    print(command)

    command_id = ssm.run_command(instance_id,command)

    print('Training launched')
    listen_to_remote_process(file_to_get_back)
    print('Training finished')

    subprocess.call("scp -i {} -r ubuntu@{}:~/outputs/cifar10_nns/{}/ {}/Spearmint-PESM/examples/cifar10/output/{}/".format(private_key, public_DNS, job_id, PATH_TO_HACONE, name), shell=True)
    subprocess.call("scp -i {} ubuntu@{}:~/accuracy_{}.txt {}/Spearmint-PESM/examples/cifar10/output/{}/accuracy.txt".format(private_key, public_DNS, job_id,PATH_TO_HACONE, name), shell=True)
    output_dir = os.path.join('outputs',str(job_id))

    dirname = buckets.download_from_s3('astar-trainedmodels')

    with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output/{}/accuracy.txt'.format(name)) as my_file:
        dico = json.load(my_file)

    print(file_to_send)

    subprocess.call("scp {} {}@{}:/home/{}".format(file_to_send, user, server_DNS,user),shell=True)
    subprocess.call("ssh {}@{} './monitor_inference_4_GPU.py {} {}\'".format(user, server_DNS, name, job_id), shell=True)
    subprocess.call("scp {}@{}:/home/arturo/hardware_metrics_{}.txt {}/Spearmint-PESM/examples/cifar10/output/{}/hardware_metrics.txt ".format(user, server_DNS, job_id, PATH_TO_HACONE,name), shell=True)
    with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output/{}/hardware_metrics.txt'.format(name)) as my_file:
        dico_hardware = json.load(my_file)
    for key, value in dico.items():
        dico_hardware["{}".format(key)] = value


    with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output/{}/hardware_metrics.txt'.format(name), 'w') as my_file:
        json.dump(dico_hardware, my_file)
    return dico_hardware
