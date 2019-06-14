dataset_name=$1
job_name=$2  

train_dir=outputs/$dataset_name/$job_name

#dataset_dir=/raid/gengxue/data/ILSVRC2012/tfrecords
#dataset_dir=/scratch/users/astar/i2r/caill/dataset/tfrecords

dataset_dir=/home/lile/dataset/cifar10

python tensorflow/slim/train_image_classifier.py --train_dir=$train_dir --num_clones=1 --dataset_name=$dataset_name --dataset_split_name=train --dataset_dir=$dataset_dir --model_name=mobilenet_v2 --max_number_of_steps=60000000 --learning_rate=0.1 --batch_size=32 --num_epochs_per_decay=30 --learning_rate_decay_factor=0.1 --train_image_size=32
