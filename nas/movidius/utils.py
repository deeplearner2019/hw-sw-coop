import time
import subprocess

def exists_remote(file_path,private_key,public_DNS):
    status = subprocess.check_output(['ssh -i {} ubuntu@{} \'if [ -e {} ]; then echo \'yes\'; else echo \'no\'; fi\''.format(private_key,public_DNS, file_path)],shell=True)
    print('status :{}'.format(status))
    if status == 'yes\n':
        return True
    if status == 'no\n':
        return False

def listen_to_remote_process(file_path,private_key,public_DNS,debug=False):
    timer = 0
    while True:
        if debug:
            time.sleep(30)
        else:
            time.sleep(600)
        timer = timer + 1
        print('Searching for the result file {}'.format(file_path))
        if exists_remote(file_path,private_key,public_DNS) or timer > 20: # more than 3 hours
            break
