
1) setup
CUDA 10.0
CUDNN 7.5
TensorFlow-GPU 10.0


2) train and evaluate 4-bit model

train: python3 train_eval_cnn_4bit.py
evaluate: python3 train_eval_cnn_4bit.py --eval


3) train and evaluate 8-bit model

train: python3 train_eval_cnn_8bit_v2.py
evaluate: python3 train_eval_cnn_8bit_v2.py --eval