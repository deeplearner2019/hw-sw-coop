## 1. Setup Spearmint on CIFAR10

HOME_PATH = ./nas

#### (1) Environment setup

Add HOME_PATH/tensorflow, HOME_PATH/tensorflow/slim to your PYTHONPATH.

Finish the setup steps as instructed in HOME_PATH/Spearmint-PESM/README.md
Try the moo example to see if the setup was correct.

Spearmint will check into the 10.217.128.198 server if the model was already trained or not. So you need to run ssh-copy-id in order to connect to the server automatically.

#### (2) Download CIFAR10 dataset and convert to tensorflow format

cd to HOME_PATH/tensorflow/slim, in the command window, run:
```bash
python download_and_convert_data.py --dataset_name=cifar10_val --dataset_dir=HOME_PATH/dataset
```
dataset_dir is the directory where the converted .tfrecord files will be stored.

Note: the difference between cifar10_val and the original cifar10 is that: in cifar10_val, we randomly split 5000 images from the original 50,000 training images as validation set.

#### (3) MongoDB setup

After installing MongoDB, open a command window, create two new folders named logs and mongod in current folder, start a mongodb server by running:
```bash
mongod  --logpath ./logs/cifarnet.log --dbpath ./mongod
```
Leave the window open.
You can find in the repo movidius_db_dump, jetson_gpu_db_dump. I used mongodump, so you can use mongorestore --dir <path> to restore them.


## 2. Setup Movidius
#### a. Movidius

(1) Install the SDK
Run:
```bash
mkdir -p ~/workspace
cd ~/workspace
git clone -b ncsdk2 https://github.com/movidius/ncsdk.git
cd ~/workspace/ncsdk2
make install
```

You may need to add /opt/movidius/NCSDK/ncsdk-x86_64/api/python:/opt/movidius/NCSDK/ncsdk-x86_64/tk:/opt/movidius/caffe/python to your PYTHONPATH.

(2) Fix bugs
Open '/usr/local/bin/ncsdk/Models/NetworkStage.py', in attach(), change:
for p in parents.tail:
    if p == self:
       newtail.append(stage)
to:
for p in parents.tail:
    if p == self:
       newtail.append(stage)
    else:
       newtail.append(p)

#### b. Power Z software
(1) Install power z app on Windows
The app can be downloaded from : https://pan.baidu.com/s/1o7AbkYe#list/path=%2F&parentPath=%2F2.%E6%88%91%E7%9A%84%E6%96%87%E6%A1%A3

(2) Launch a ssh server (open-ssh for example) on Windows machine. Run ipconfig to see you IP address.

(3) ssh without password
On linux machine, run ssh-keygen -t rsa to generate a private key (id_rsa) and a public key (id_rsa.pub).
copy the public key to /home/User/.ssh/authorized_keys on your windows machine

(4) remote desktop to Windows machine
Connet to Windows machine using Remmina. 

(5) use xdotool to automatically start and stop the Power-z app running in Remmina.
You need to get the coordinates of start_button, stop_button, export_csv and ok_export. This can be obtained by running xdotool getmouselocation in a cmd window and move your mouse to the corresponding button and record its coordinates.

(6) xdotool windowfocus $WID is supposed to give focus to the Remmina window. But if it is not working, you can manually bring the Remmina window to front.

## 3. Run Spearmint-PESM
Open a new command window, cd to HOME_PATH/Spearmint-PESM, run:
```bash
python spearmint/main.py examples/cifar10 --config=<config_file_name>.json
```

config_file_name can be config_cifarnet_jetson|movidius|titanx.json.

If you want to clean the database and the output folder, run:
```bash
python spearmint/cleanup.py examples/cifar10 --config=<config_file_name>.json
```
