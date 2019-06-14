import ssm
import buckets
import json
import subprocess
import os
import time
import glob
import csv
import config as cfg

def exists_remote(file_path, public_DNS):
    '''
    Check if a file exists on a remote AWS server.
    Input :
        - file_path : String, path of the file on the server
        - public_DNS : String, DNS of the AWS instance
    Return boolean
    '''
    status = subprocess.check_output(['ssh -i {} ubuntu@{} \'if [ -e {} ]; then echo \'yes\'; else echo \'no\'; fi\''.format(cfg.private_key, public_DNS, file_path)],shell=True)
    print('status :{}'.format(status))
    if status == 'yes\n':
        return True
    if status == 'no\n':
        return False

def exists_server(file_path, server_DNS = '10.217.128.217', user = 'arturo'):
    '''
    Check if a file exists on a remote server.
    Input :
        - file_path : String, path of the file on the server
        - server_DNS : String, DNS of the server
        - user : String, Username on the server
    Return boolean
    '''
    status = subprocess.check_output(['ssh {1}@{0}  \'if [ -e {2} ]; then echo \'yes\'; else echo \'no\'; fi\''.format(server_DNS, user, file_path)], shell=True)
    print('status :{}'.format(status))
    if status == 'yes\n':
        return True
    if status == 'no\n':
        return False

def lock_process(cmd,filename,lock_name):
    '''
    Lock a process. If another process with the same lock_name is running, wait until it finishes. Create a commands file to execute when it is allowed to.
    Input:
        - cmd: String, the commands to execute and protect
        - filename: String, name of the commands file to be created
        - lock_name: String, name of the category of process
    '''
    cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; {}; sleep 5) 9>/var/lock/{}'.format(cmd,lock_name) #Allow us to execute only one command on the GPU at a time
    command_file = os.path.join(cfg.PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files',filename+'.sh')
    with open(command_file,'w') as f:
        f.write(cmd_s)
    subprocess.call('sh {0}; rm {0}'.format(command_file), shell = True)

def lock_instance_gpu(command,filename,file_to_get_back, ID_NUMBER, instance_id, public_DNS):
    '''
    Lock a set of gpus on the cloud. If another model is already training on it, wait until it finishes. Create a commands file to execute when it is allowed to.
    Input:
        - command: String, the commands to send to the AWS instance
        - filename: String, name of the commands file to be created
        - instance_id: String, id of the AWS instance
        - public_DNS: String, DNS of the AWS instance
        - ID_NUMBER: String, id of the set of GPUs to protect
        - file_to_get_back: String, path of the file we want to get back from the AWS instance
    '''
    cmd = ['import functions as f','import ssm','command_id = ssm.run_command(\'{0}\',{1})','f.listen_to_remote_process(\'{2}\',\'{3}\')']
    python_script = '\n'.join(cmd)
    python_script = python_script.format(instance_id,command,file_to_get_back,public_DNS)

    python_file = os.path.join(cfg.PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files',filename+'.py')
    command_file = os.path.join(cfg.PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files',filename+'.sh')

    cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; cd {}/Spearmint-PESM/examples/cifar10; python {}; sleep 5) 9>/var/lock/p28_{}'.format(cfg.PATH_TO_HACONE,python_file,ID_NUMBER)

    with open(python_file,'w') as f:
        f.write(python_script)
    with open(command_file,'w') as f:
        f.write(cmd_s)

    subprocess.call('sh {0}; rm {0}; rm {1}'.format(command_file, python_file), shell = True)

def lock_server_gpu(command,filename,file_to_get_back, ID_NUMBER):
    '''
    Lock a set of gpus on a server. If another model is already training on it, wait until it finishes. Create a commands file to execute when it is allowed to.
    Input:
        - command: String, the commands to send to the AWS instance
        - filename: String, name of the commands file to be created
        - instance_id: String, id of the AWS instance
        - public_DNS: String, DNS of the AWS instance
        - ID_NUMBER: String, id of the set of GPUs to protect
        - file_to_get_back: String, path of the file we want to get back from the AWS instance
    '''
    cmd = ['import functions as f','f.listen_to_server_process(\'{0}\')']
    python_script = '\n'.join(cmd)
    python_script = python_script.format(file_to_get_back, cfg.server_DNS_train, cfg.server_user_train)

    python_file = os.path.join(cfg.PATH_TO_HACONE,'movidius','command_files',filename+'.py')
    command_file = os.path.join(cfg.PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files',filename+'.sh')

    cmd = ['export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH','export PYTHONPATH={0}/tensorflow/slim:{0}/tensorflow:$PYTHONPATH',command]
    cmd = '\n'.join(cmd).format(cfg.server_working_dir)
    cmd_s = '#!/bin/bash\n(flock -w 3600 9 || exit 1; cd {0}/Spearmint-PESM/examples/cifar10; {1}; python {2}; sleep 5) 9>/var/lock/p28_{3}'.format(cfg.PATH_TO_HACONE, cmd, python_file, ID_NUMBER)

    with open(python_file,'w') as f:
        f.write(python_script)
    with open(command_file,'w') as f:
        f.write(cmd_s)

    subprocess.call('sh {0}; rm {0}; rm {1}'.format(command_file, python_file), shell = True)

def listen_to_remote_process(file_path, public_DNS):
    timer = 0
    while True:
        if cfg.debug:
            time.sleep(30)
        else:
            time.sleep(600)
        timer = timer + 1
        print('Searching for the result file {}'.format(file_path))
        if exists_remote(file_path, public_DNS) or timer > 20: # more than 3 hours
            break

def listen_to_server_process(file_path):
    timer = 0
    while True:
        if cfg.debug:
            time.sleep(30)
        else:
            time.sleep(600)
        timer = timer + 1
        print('Searching for the result file {}'.format(file_path))
        if exists_server(file_path, server_DNS = cfg.server_DNS_train, user = cfg.server_user_train) or timer > 20: # more than 3 hours
            break

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
                power.append(row[3])
            counter += 1
    return power

def compute_energy(power):
    '''
    Compute the energy consumption on the phone during the inference. Return the total energy - idle energy.
    Input: String, path of the phone measurements
    Return: float, energy consumption
    '''
    mean_value = sum([float(p) for p in power[:100]])/len(power[:100])
    power_0 = [float(p) - mean_value for p in power]
    print(mean_value)
    energy = sum(power_0)
    energy = energy * 0.02
    return energy

def create_name(dico):
    '''
    Create a unique identifier from the parameters of the model
    Input:
        - dico: Dict, parameters of the model
    Return:
        - name: String, name of the model (ex: _0_-2_2_3_0_2_3_2_-2_-1_1_4_1_0_5_3_4_4_0_3)
    '''
    dico = json.loads(dico)
    name=''
    for key, value in dico.items():
        name+='_'
        name+=str(value)
    return name

def is_already_trained(name):
    '''
    Check on the server to see if the model was already trained.
    Input:
        - name: String, name of the model (ex: _0_-2_2_3_0_2_3_2_-2_-1_1_4_1_0_5_3_4_4_0_3)
    Return:
        - boolean
    '''
    done_jobs = subprocess.check_output('ssh {0} ls {1}'.format(cfg.server_DNS_where_models_are_stored,cfg.server_dir_models),shell=True)
    trained_jobs = done_jobs.split('\n')
    return name in trained_jobs

def train_on_cloud(name, job_id, public_DNS, instance_id, nb_gpus_to_use, gpus_to_use, max_number_of_steps, dico_to_save, ID_NUMBER):
    '''
    Train the model on the cloud.
    Input:
        - name: String, name of the model (ex: _0_-2_2_3_0_2_3_2_-2_-1_1_4_1_0_5_3_4_4_0_3)
        - job_id: id of the current job
        - public_DNS: String, DNS of the AWS instance
        - instance_id: String, id of the AWS instance
        - nb_gpus_to_use: int, number of GPUs to use for the training of the model
        - gpus_to_use: String, ids of the GPUs to use
        - max_number_of_steps: int, number of steps the model need to train
        - dico_to_save: String, path to the file containing the job_id and the parameters
        - ID_NUMBER: String, id of the set of GPUs to use to train the model
    '''
    command = []

    if not is_already_trained(name):
        file_to_send = "{}/jobs/job{}.txt".format(cfg.PATH_TO_HACONE,name)
        file_to_get_back = '/home/ubuntu/accuracy_{}.txt'.format(job_id)
        with open(file_to_send, 'wb') as fp:
            json.dump(dico_to_save, fp)

        #check if job number X isn't already stored on the instance
        remote_output_dir = '/home/ubuntu/outputs/cifar10_nns/{}'.format(job_id)
        if exists_remote(remote_output_dir, public_DNS):
            ssm.run_command(instance_id,['sudo rm -rf ' + remote_output_dir])

        # now we need to send this file : job_file to the AWS instance.
        subprocess.call("scp -i {} {} ubuntu@{}:~/job_file_{}.txt".format(cfg.private_key, file_to_send, public_DNS, job_id), shell=True)
        command.append("export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH")

        # Launch the train and eval script
        command.append("./train_eval_image_classifier_bis.py {} {} --max_number_of_steps={} --num_clones={}".format(gpus_to_use, job_id, max_number_of_steps, nb_gpus_to_use))
        print(command)

        #Run the command to launch the training and waiting for it to finish
        lock_instance_gpu(command,'lock_instance'+name,file_to_get_back,ID_NUMBER, instance_id, public_DNS) #Allow us to be the only one to use the instance gpu

        #Get back the accuracy and checkpoints of the model
        subprocess.call("scp -i {} -r ubuntu@{}:~/outputs/cifar10_nns/{}/ {}/Spearmint-PESM/examples/cifar10/output_0/{}/".format(cfg.private_key, public_DNS, job_id, cfg.PATH_TO_HACONE, name), shell=True)
        subprocess.call("scp -i {} ubuntu@{}:~/accuracy_{}.txt {}/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt".format(cfg.private_key, public_DNS, job_id,cfg.PATH_TO_HACONE, name), shell=True)

        #Send the checkpoints and the accuracy on the server
        if not cfg.debug:
            subprocess.call("scp -r {0}/Spearmint-PESM/examples/cifar10/output_0/{1} {2}:{3}/{1}".format(cfg.PATH_TO_HACONE,name,cfg.server_DNS_where_models_are_stored,cfg.server_dir_models), shell = True)

    else:
        print("Model already trained")
        subprocess.call("scp -r {2}:{3}/{1} {0}/Spearmint-PESM/examples/cifar10/output_0/{1}".format(cfg.PATH_TO_HACONE,name,cfg.server_DNS_where_models_are_stored,cfg.server_dir_models), shell = True)

def train_on_server(name, job_id, nb_gpus_to_use, gpus_to_use, max_number_of_steps, dico_to_save, ID_NUMBER):
    '''
    Train the model on the server specified in the config file.
    Input:
        - name: String, name of the model (ex: _0_-2_2_3_0_2_3_2_-2_-1_1_4_1_0_5_3_4_4_0_3)
        - job_id: id of the current job
        - nb_gpus_to_use: int, number of GPUs to use for the training of the model
        - gpus_to_use: String, ids of the GPUs to use
        - max_number_of_steps: int, number of steps the model need to train
        - dico_to_save: String, path to the file containing the job_id and the parameters
        - ID_NUMBER: String, id of the set of GPUs to use to train the model
    '''
    if not is_already_trained(name):
        file_to_send = "{}/jobs/job{}.txt".format(cfg.PATH_TO_HACONE,name)
        file_to_get_back = "{}/outputs/cifar10_nns/{}/results.txt".format(cfg.server_working_dir, job_id)
        with open(file_to_send, 'wb') as fp:
            json.dump(dico_to_save, fp)

        #check if job number X isn't already stored on the instance
        remote_output_dir = '{}/outputs/cifar10_nns/{}'.format(cfg.server_working_dir, name)
        if exists_server(remote_output_dir, server_DNS = cfg.server_DNS_train, user = cfg.server_user_train):
            subprocess.call('sudo rm -rf {}'.format(remote_output_dir), shell = True)

        # now we need to send this file : job_file to the server.
        subprocess.call('scp {} {}@{}:/home/annemaelle/jobs/'.format(file_to_send,cfg.server_user_train,cfg.server_DNS_train),shell=True)

        # Launch the train and eval script
        command = 'ssh {0}@{1} \'sh {2}/train_and_eval.sh\' {3} {4} {5} {6} {7}'.format(cfg.server_user_train,cfg.server_DNS_train, cfg.server_working_dir, job_id, gpus_to_use, max_number_of_steps, nb_gpus_to_use, name)

        #Run the command to launch the training and waiting for it to finish
        lock_server_gpu(command,'lock_instance'+name,file_to_get_back, ID_NUMBER)

        #Get back the accuracy and checkpoints of the model
        files_to_download = "{}/outputs/cifar10_nns/{}/*".format(cfg.server_working_dir, job_id)
        subprocess.call(['scp {0}@{1}:{2} {3}/Spearmint-PESM/examples/cifar10/output_0/{4}/'.format(cfg.server_user_train,cfg.server_DNS_train,files_to_download,cfg.PATH_TO_HACONE,job_id)],shell=True)
        subprocess.call(['ssh {}@{} \'rm -rf {}/outputs/cifar10_nns/{}\''.format(cfg.server_user_train,cfg.server_DNS_train,cfg.server_working_dir, job_id)],shell=True)

        #Send the checkpoints and the accuracy on the server
        if not cfg.debug:
            subprocess.call("scp -r {0}/Spearmint-PESM/examples/cifar10/output_0/{1} {2}:{3}/{1}".format(cfg.PATH_TO_HACONE,name,cfg.server_DNS_where_models_are_stored,cfg.server_dir_models), shell = True)

    else:
        print("Model already trained")
        subprocess.call("scp -r {2}:{3}/{1} {0}/Spearmint-PESM/examples/cifar10/output_0/{1}".format(cfg.PATH_TO_HACONE,name,cfg.server_DNS_where_models_are_stored,cfg.server_dir_models), shell = True)

def test_movidius(name, max_number_of_steps, job_id):
    '''
    Test the model on the movidius. We first need to export the graph to a protobuf file and freeze the weights
    The software to measure energy consumption only works on Windows and does not have an API. So we are connecting via Remmina to a Windows machine and then moving the cursor on the screen to start, stop and export the measurements.
    Input:
        - name: String, name of the model (ex: _0_-2_2_3_0_2_3_2_-2_-1_1_4_1_0_5_3_4_4_0_3)
        - max_number_of_steps: int, number of steps the model need to train
        - job_id: id of the current job
    Return:
        - dico_hardware: Dict, dictionnary containing the results f, power and time
    '''

    #Export the graph to a protobuf file
    subprocess.call("python {0}/tensorflow/nn_search/export_inference_graph_movidius.py \
    --job_name=cifar10_movidius \
    --name_job={1} \
    --output_file={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/inference_graph.pb \
    --PATH_TO_HACONE={0}".format(cfg.PATH_TO_HACONE,name),shell=True)

    #Freeze the weights in the graph in a protobuf file
    subprocess.call("python {0}/tensorflow/nn_search/freeze_graph_16.py \
    --input_graph={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/inference_graph.pb \
    --input_checkpoint={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/model.ckpt-{2} \
    --input_binary=true \
    --output_graph={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/frozen_graph.pb \
    --output_node_names=CifarNet/Predictions/Reshape_1".format(cfg.PATH_TO_HACONE,name,max_number_of_steps),shell=True)

    dico_hardware = {}

    #Get back the power and inference time measurement on the Movidius
    #Start and stop monitoring power
    cmd_00 = "WID=$(xdotool search --name windows-astar | head -1); xdotool windowfocus $WID"
    cmd_0 = "xdotool mousemove {} click 1".format(cfg.start_button)#click on Start
    cmd_1 = "sleep 30"
    cmd_2 = "cd {0}/movidius; make all MODEL_DIR={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/".format(cfg.PATH_TO_HACONE, name) #Launch the model on Movidius
    cmd_3 = "sleep 30"
    cmd_4 = "xdotool windowfocus $WID; xdotool mousemove {} click 1".format(cfg.stop_button) #click on Stop
    cmd_5 = "sleep 5; xdotool windowfocus $WID; xdotool mousemove {} click 1".format(cfg.export_csv) #click on export
    cmd_6 = "xdotool windowfocus $WID; xdotool mousemove {}; sleep 10; xdotool click 1".format(cfg.ok_export) #click on OK
    cmd_7 = "ssh {0}@{1} \'filename=$(ls C:/Users/User/movidius -t | head -1); mv \"C:/Users/User/movidius/$filename\" C:/Users/User/movidius/result_{2}.csv\';echo \"file done\"".format(cfg.user_id_windows,cfg.DNS_windows,job_id) # Get back the measurements
    cmd_8 = "sleep 60"
    cmd = ' ; '.join([cmd_00, cmd_0, cmd_1, cmd_2, cmd_3, cmd_4,cmd_5, cmd_6, cmd_7, cmd_8])
    lock_process(cmd, 'mov'+name, 'movidius')

    #Retrieve inference time
    log_file = '{}/Spearmint-PESM/examples/cifar10/output/{}.out'.format(cfg.PATH_TO_HACONE,job_id.zfill(8))
    with open(log_file) as my_file:
        for line in my_file:
            if 'Total inference time' in line:
                time_inference = float(line.replace(' ','').split('Totalinferencetime')[1].split('\n')[0])
                dico_hardware['time'] = time_inference

    #Retrieve accuracy
    with open('{}/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt'.format(cfg.PATH_TO_HACONE,name)) as acc:
        dico = json.load(acc)
        dico_hardware['f'] = dico['f']

    #Retrieve power
    subprocess.call('scp {0}@{1}:\'C:/Users/User/movidius/result_{2}.csv\' {3}/Spearmint-PESM/examples/cifar10/output_0/{4}/'.format(cfg.user_id_windows, cfg.DNS_windows, job_id, cfg.PATH_TO_HACONE, name), shell = True)
    result_file = '{}/Spearmint-PESM/examples/cifar10/output_0/{}/result_{}.csv'.format(cfg.PATH_TO_HACONE,name, job_id)
    power = get_power(result_file)
    energy = compute_energy(power)
    dico_hardware['power'] = energy

    #Save results
    with open(cfg.PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/hardware_metrics.txt'.format(name), 'w') as my_file:
        json.dump(dico_hardware, my_file)

    return dico_hardware

def test_phone(name, max_number_of_steps):
    #Export the graph to a protobuf file
    cmd = "python {0}/tensorflow/nn_search/export_inference_graph_movidius.py \
    --job_name=cifar10_phone \
    --name_job={1} \
    --output_file={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/inference_graph.pb \
    --PATH_TO_HACONE={0}".format(cfg.PATH_TO_HACONE,name)

    command_file = os.path.join(cfg.PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files','pb_'+name+'.sh')

    #Protect the GPU of the computer
    lock_process(cmd,command_file,'gpu')

    #Freeze the weights in the graph in a protobuf file
    cmd = "python {0}/tensorflow/nn_search/freeze_graph_16.py \
    --input_graph={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/inference_graph.pb \
    --input_checkpoint={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/model.ckpt-{2} \
    --input_binary=true \
    --output_graph={0}/Spearmint-PESM/examples/cifar10/output_0/{1}/{1}.pb \
    --output_node_names=CifarNet/Predictions/Reshape_1".format(cfg.PATH_TO_HACONE,name,max_number_of_steps)

    command_file = os.path.join(PATH_TO_HACONE,'Spearmint-PESM','examples','cifar10','command_files','fr_'+name+'.sh')

    #Protect the GPU of the computer
    lock_process(cmd,command_file,'gpu')

    #Measure power and inference time on the phone
    subprocess.call("cp {0}/Spearmint-PESM/examples/cifar10/output_0/{1}/{1}.pb {0}/snpe-sdk/models/cifarnet/tensorflow/".format(cfg.PATH_TO_HACONE,name),shell=True)
    subprocess.call("cd {0}/snpe-sdk; python ./models/cifarnet/scripts/setup_cifarnet.py -S {0}/snpe-sdk -A {3} -t {4} -a ./models/cifarnet/data -f {1} {2}".format(cfg.PATH_TO_HACONE, name, '-d'*cfg.debug, cfg.ANDROID_NDK_ROOT, cfg.TENSORFLOW_HOME), shell=True)

    dico_hardware = {}

    #Retrieve inference time in us
    stats_file = '{}/snpe-sdk/benchmarks/cifarnet/benchmark/{}/latest_results/benchmark_stats_CifarNet.csv'.format(cfg.PATH_TO_HACONE,name)
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
                energy

                _joules = float(row[3])
                dico_hardware['power'] = energy_joules

    #Retrieve accuracy
    with open('{}/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt'.format(cfg.PATH_TO_HACONE,name)) as acc:
        dico = json.load(acc)
        dico_hardware['f'] = dico['f']

    with open(cfg.PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/hardware_metrics.txt'.format(name), 'w') as my_file:
        json.dump(dico_hardware, my_file)

    return dico_hardware

def test_gpu(name, job_id):
    dico_hardware = {}
    file_to_send = "{}/jobs/job{}.txt".format(cfg.PATH_TO_HACONE,name)

    #Send model to test to the server
    subprocess.call("scp {0} {1}@{2}:/home/data/{1}/jobs/".format(file_to_send, cfg.user_test, cfg.server_DNS_test),shell=True)

    #Test the model on the server
    subprocess.call("ssh {}@{} './monitor_inference_4_GPU.py {} {}\'".format(cfg.user, cfg.server_DNS, name, job_id), shell=True)

    #Get back time and power
    subprocess.call("scp {}@{}:/home/data/arturo/hardware_metrics/hardware_metrics_{}.txt {}/Spearmint-PESM/examples/cifar10/output_0/{}/hardware_metrics.txt ".format(cfg.user_test, cfg.server_DNS_test, job_id, cfg.PATH_TO_HACONE,name), shell=True)

    #Retrieve accuracy
    with open('{}/Spearmint-PESM/examples/cifar10/output_0/{}/accuracy.txt'.format(cfg.PATH_TO_HACONE,name)) as acc:
        dico = json.load(acc)
        dico_hardware['f'] = dico['f']

    #Retrieve power and inference time
    with open(cfg.PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/hardware_metrics.txt'.format(name)) as my_file:
        dico_hardware = json.load(my_file)
    for key, value in dico.items():
        dico_hardware["{}".format(key)] = value

    #Save results
    with open(cfg.PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output_0/{}/hardware_metrics.txt'.format(name), 'w') as my_file:
        json.dump(dico_hardware, my_file)

    return dico_hardware
