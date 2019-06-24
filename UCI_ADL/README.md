# Dataset
The dataset can be retrieved at https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer.
It is about daily life activity recognition and contains more than 1000 samples.

# Pre requisite
- Keras
- Tensorflow
- Numpy

# Files description
- parse_data : retrieve and parse data from the UCI database
- preprocess_classifier : helper functions to preprocess data for classication
- preprocess_AD : same for anomaly detection
- train_lstm_classifier
- train_lstm_AD
- anomaly detector : compute anomaly score

# Results for classification
The baseline is a vanilla LSTM for classification : two stacked LSTM layers followed by a Fully Connected layer.
Data are preprocessed into smaller subsequences using sliding windows.
A naive implementation achieves >97% of categorical accuracy

# Results for anomaly detection
The baseline is composed of two LSTM layers ; it is inspired from [Long Short Term Memory Networks for Anomaly Detection in Time Series, P. Malhotra et al., 2015](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf).
For each class, we train a model on a normal dataset consisting in samples from the 13 other classes and test on the 13+1 (outlier) classes.
