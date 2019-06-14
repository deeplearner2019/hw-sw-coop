import ssm
import time
import buckets
import json
import unicodedata

instance_id = 'i-008df83f650db538b' #2xlarge

def main(job_id, params):
    #ssm.purge_queue()
    job_id = str(job_id)
    command = []
    print(params)
    my_params={}
    #Encode params to pass it through the run_command
    for key in params.keys():
        my_params[key.replace('"','\'')] = int(params[key])
    my_params = json.dumps(my_params)

    command.append("export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH")
    #Launch the train and eval script
    command.append("./train_eval_image_classifier.py --job_id={} --params='{}'".format(job_id, my_params))
    print(command)

    command_id = ssm.run_command(instance_id,command)

    dic = {}

    while True:
        dic = ssm.notif_listener(dic);
        print('\nGlobal dic : ',dic,'\n')

        if dic[command_id]!='InProgress':
            print(dic[command_id])
            command = []

            if dic[command_id]=='Success':
                output_dir = os.path.join('outputs',str(job_id))

                #go to the output dir and copy everything to a bucket, then remove everything from the instance
                command.append('cd {}'.format(output_dir))
                command.append('aws s3 cp ./gpu_0 s3://astar-trainedmodels/{}/{}/awsrunShellScript/0.awsrunShellScript/ --recursive'.format(command_id,instance_id))
                command.append('rm -rf ./gpu_0')
                command_id = ssm.run_command(instance_id,command,save_bucket=False)
            break

    #f = buckets.download_from_s3('astar_trainedmodels')
    return f
