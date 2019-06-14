#General
debug = False
PATH_TO_HACONE = '/home/anne-maelle_2/hacone'

# Defining the variables for AWS.
instance_id_p28 = 'i-04c058a0d9655ab16' #p2.8xlarge
instance_id_p32 = 'i-063c932f5a2004818' #p3.2xlarge
public_DNS_p28 = 'ec2-54-91-170-50.compute-1.amazonaws.com'
public_DNS_p32 = 'ec2-107-23-219-187.compute-1.amazonaws.com'
private_key = PATH_TO_HACONE+"/AWS/anne-maelle_key.pem"

#For the phone
ANDROID_NDK_ROOT = '/home/arthur/Qualcomm/Hexagon_SDK/3.3.3/tools/android-ndk-r14b'
TENSORFLOW_HOME = '/home/anne-maelle/hacone/tensorflow-2.7/local/lib/python2.7/site-packages/tensorflow-1.9.0.dist-info'

#For the storage of the models
server_DNS_where_models_are_stored = 'annemaelle@10.217.128.198'
server_dir_models = '/mnt/data_b/anne-maelle/models_trained'

#For the movidius
start_button = '2600 1150'
stop_button = '2750 1150'
export_csv = '2600 1230'
ok_export = '1600 1130'
user_id_windows = 'User'
DNS_windows = '10.217.131.137'

#For the GPU testing
server_user_test = 'arturo'
server_DNS_test = '10.217.128.217'

#For training on GPU
server_user_train = 'annemaelle'
server_DNS_train = '10.217.128.198'
server_working_dir = '/mnt/data_b/anne-maelle/hacone'
