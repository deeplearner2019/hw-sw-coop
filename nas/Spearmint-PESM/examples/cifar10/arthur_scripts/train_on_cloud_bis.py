"""
This script aims at sending a unique command to AWS.
"""
import ssm
import buckets
import json
import subprocess
import os

# Defining the variables for AWS.
instance_id = 'i-0d654fa8ba8b5025f' #p2xlarge
public_DNS = 'ec2-34-202-205-178.compute-1.amazonaws.com'
private_key = "/home/arthur/Documents/hacone/AWS/arthur_key.pem"
PATH_TO_HACONE = '/home/arthur/Documents/hacone'

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
    file_to_send = PATH_TO_HACONE + "/jobs/job_file_{}.txt".format(job_id)
    with open(file_to_send, 'wb') as fp:
        json.dump(dico_to_save, fp)

    # now we need to send this file : job_file to the AWS instance.
    subprocess.call("scp -i {} {} ubuntu@{}:~/job_file.txt".format(private_key, file_to_send, public_DNS), shell=True)

    command.append("export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH")
    # Launch the train and eval script

    command.append("./train_eval_image_classifier_bis.py")
    print(command)

    command_id = ssm.run_command(instance_id,command)

    dic = {}

    while True:
        dic = ssm.notif_listener(dic);
        print('\n Global dic : ',dic,'\n')

        if dic[command_id]!='InProgress':
            print(dic[command_id])
            command_2 = []

            if dic[command_id]=='Success':
                subprocess.call("scp -i {} -r ubuntu@{}:~/outputs/cifar10_nns/{}/ /home/arthur/Documents/hacone/Spearmint-PESM/examples/cifar10/output/".format(private_key, public_DNS, job_id), shell=True)
                subprocess.call("scp -i {} ubuntu@{}:~/accuracy.txt /home/arthur/Documents/hacone/Spearmint-PESM/examples/cifar10/output/accuracy_{}.txt".format(private_key, public_DNS, job_id), shell=True)
                output_dir = os.path.join('outputs',str(job_id))

                #go to the output dir and copy everything to a bucket, then remove everything from the instance
                command_2.append('cd {}'.format(output_dir))
                command_2.append('aws s3 cp ./{} s3://astar-trainedmodels/{}/{}/awsrunShellScript/0.awsrunShellScript/ --recursive'.format(output_dir,command_id,instance_id))
                command_2.append('rm -rf ./{}'.format(output_dir))
                command_id_2 = ssm.run_command(instance_id,command_2,save_bucket=False)

                while True:
                    dic = ssm.notif_listener(dic)
                    if dic[command_id_2]!='InProgress':
                        print(dic[command_id_2])
                        dic.pop(command_id_2,None)
                        break

            dirname = buckets.download_from_s3('astar-trainedmodels')
            dic.pop(command_id,None)
            with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output/accuracy_{}.txt'.format(job_id)) as my_file:
                dico = json.load(my_file)
            break
    dic = ssm.notif_listener(dic)
    #Now it's time to find the hardware characterictics : time, memory, power.
    subprocess.call(PATH_TO_HACONE + "/monitoring_GPU/monitor_inference.py {}".format(job_id), shell=True)
    with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output/hardware_metrics_{}.txt'.format(job_id)) as my_file:
        dico_hardware = json.load(my_file)
    for key, value in dico.items():
        dico_hardware["{}".format(key)] = value


    with open(PATH_TO_HACONE + '/Spearmint-PESM/examples/cifar10/output/hardware_metrics_{}.txt'.format(job_id), 'w') as my_file:
        json.dump(dico_hardware, my_file)
    return dico_hardware
