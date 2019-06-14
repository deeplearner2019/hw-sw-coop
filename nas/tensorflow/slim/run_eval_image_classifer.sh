dataset_name=$1
job_name=$2  

train_dir=outputs/$dataset_name/$job_name

#dataset_dir=/raid/gengxue/data/ILSVRC2012/tfrecords
#dataset_dir=/scratch/users/astar/i2r/caill/dataset/tfrecords

dataset_dir=/home/lile/dataset/cifar10

python tensorflow/slim/eval_image_classifier.py --checkpoint_path=$train_dir --dataset_name=$dataset_name --dataset_split_name=test --dataset_dir=$dataset_dir --model_name=mobilenet_v2 --eval_image_size=32
