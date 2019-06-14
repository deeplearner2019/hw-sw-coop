"""
This script aims at training on the cloud and then testing the trained model on the platform.
It commands to train a given model in the first GPU, ie number 0.
"""
import json

import functions as f
import config as cfg

# Variables
nb_gpus_to_use = 2
gpus_to_use = '0,1'
ID_NUMBER = 1

def main(job_id, params):
    job_id = str(job_id)
    print(params)

    # Encode params to pass it through the run_command
    my_params={}
    for key in params.keys():
        my_params[key.replace('"','\'')] = int(params[key])
    my_params = json.dumps(my_params)

    dico_to_save = {}
    dico_to_save['platform'] = 'Movidius'
    dico_to_save['job']= job_id
    dico_to_save['params'] = my_params

    #Create a unique identifier for this model
    name = f.create_name(my_params)

    if cfg.debug:
        max_number_of_steps = 100
    else :
        max_number_of_steps = 28125 #With a batch_size at 32, we train for 20 epochs

    #Launch training of the model on the cloud
    f.train_on_server(name, job_id, nb_gpus_to_use, gpus_to_use, max_number_of_steps, dico_to_save, ID_NUMBER)

    #Test the trained model on the platform
    dico_hardware = f.test_movidius(name, max_number_of_steps, job_id)

    return dico_hardware
