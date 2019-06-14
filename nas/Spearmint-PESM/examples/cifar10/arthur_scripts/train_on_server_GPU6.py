import subprocess
import json
import time
import os

#Defining the variables for the server
working_dir = '/home/anne-maelle/test_dir/hacone'
server_id = '10.217.128.198'
server_user = 'annemaelle'

def exists_remote(file_path):
    status = subprocess.call(['ssh {}@{} \'if [ -e {} ]; then echo 0; else echo 1; fi\''.format(server_user,server_id,file_path)],shell=True)
    print('status :{}'.format(status))
    if status == 0:
        return True
    if status == 1:
        return False

def listen_to_remote_process(file_path):
    while True:
        print('Searching for the result file {}'.format(file_path))
        if exists_remote(file_path):
            break

def main(job_id, params):
    job_id = str(job_id)

    my_params={}
    file_to_get_back = "/mnt/data_b/anne-maelle/hacone/outputs/cifar10_nns/{}/results.txt".format(job_id)

    saving_path = working_dir+'/results_server/{}'.format(job_id)
    file_to_send = saving_path+"/job_file.txt"
    if not os.path.exists(saving_path):
       print('Saving directory {} non-existent, creating it'.format(saving_path))
       os.makedirs(saving_path)

    # Encode params to pass it through the run_command
    for key in params.keys():
        my_params[key.replace('"','\'')] = int(params[key])
    my_params = json.dumps(my_params)
    dico_to_save = {}
    dico_to_save['job']= job_id
    dico_to_save['params'] = my_params
    with open(file_to_send, 'w') as fp:
        json.dump(dico_to_save, fp)

    subprocess.call('scp {} {}@{}:/mnt/data_b/anne-maelle/hacone'.format(file_to_send,server_user,server_id),shell=True)
    subprocess.call('ssh {}@{} \'sh /mnt/data_b/anne-maelle/hacone/train_and_eval_GPU6.sh\''.format(server_user,server_id),shell=True)

    print('Training launched')
    listen_to_remote_process(file_to_get_back)
    print('Training finished')

    fs = subprocess.check_output(['ssh {}@{} \'cat {}\''.format(server_user,server_id,file_to_get_back)],shell=True)
    f = float(fs.split(': ')[1].split('}')[0])

    files_to_download = "/mnt/data_b/anne-maelle/hacone/outputs/cifar10_nns/{}/*".format(job_id)



    print('Downloading results files')
    subprocess.call(['scp {}@{}:{} {}'.format(server_user,server_id,files_to_download,saving_path)],shell=True)
    subprocess.call(['ssh {}@{} \'rm -rf /mnt/data_b/anne-maelle/hacone/outputs/cifar10_nns/{}\''.format(server_user,server_id,job_id)],shell=True)

    # Now it's time to find the hardware characterictics : time, memory, power.
    print('Monitoring GPU')
    subprocess.call("{}/monitoring_GPU/monitor_inference.py {}".format(working_dir,job_id), shell=True)
    print('Processing monitoring results')
    with open('{}/results_server/{}/hardware_metrics.txt'.format(working_dir,job_id)) as my_file:
        dico_hardware = json.load(my_file)
    dico_hardware['f']=f
    print('Results processed, Finished')
    return dico_hardware
    #return f
