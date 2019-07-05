# Adaptive quantization

We totally have 7 steps, and usually I stored the data in ```adaptive_quantization/variables/```:
1. extract weights/activations from tensorflow model. (folder name: uci_har/extract_variables.py)
    pls note that activations can be output of 100 images.
    ```weights``` are saved in data_dir ```weights_dir```; ```activations``` are saved in ```activations_dir```
2. run c++ codes: get weights & activations codebook. (folder name: dead_zones)
    we denote:
    a. ```number_weights_layer```: the number of quantized weights layer.
    b. ```number_activations_layer```: the number of quantized weights layer. 
    c. ```list_weights_filenames.txt```: weights file names. 
    d: ```list_weights_sizes.txt``` : the number of weights in each layer. 
    e: ```list_dead_zones_ratio.txt```: configuration for dead zones ratios.
    f: ```adaptive_list_quant_levels.txt```: configuration for quantized levels.
    g: ```weights_quantized_paras```: output weights codebook directory.
    The program's arguments for weights codebook is: 
        ```number_weights_layer list_weights_filenames.txt list_weights_sizes.txt list_dead_zones_ratio.txt adaptive_list_quant_levels.txt weights_dir weights_quantized_paras```
    
    The program's arguments for activations codebook is: 
        ```number_activations_layer list_activations_filenames.txt list_activations_sizes.txt list_dead_zones_ratio.txt adaptive_list_quant_levels.txt activations_dir activations_quantized_paras```
3. weights error (folder name: weights_error): 
    a. run bash file: run1.sh & run2.sh
4. activations error (folder name: activations_error):
    a. run bash file: run1.sh & run2.sh
5. run c++: get bit allocation (folder name: dead_zones).
    a. merge the weights error and activations error together into a input file using ```pareto_condition.py``` to get pareto condition input: ```pareto_condition_inputs.txt```. 
    b. run c++ code: update ```data_dir``` & ```filename_data_points``` in ```pareto_condition.cpp``` to get bit allocation for different layers.
6. run inference code (folder name: inference-only):
    a. evaluation: ```bash run1.sh & run2.sh & run3.sh & run4.sh```
7. run fine-tune code (folder name: fine-tune)
    a. training: ``` bash run1.sh & run2.sh & run3.sh & run4.sh```
    b. evaluation: ```bash run.sh```
