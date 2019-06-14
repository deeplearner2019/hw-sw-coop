# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:33:53 2018

@author: wangzhe
"""

# define const numbers in simulator
const_number_PE = 10 # number of PE 
const_CPOP = 10 # clock cycle per single MAC operation 
const_frequency = 100 # frequency 
const_average_read_time = 5 # average reading latency 
const_average_write_time = 5 # average writing lantency 
const_number_channel = 8 # number of channels 
const_energy_op = 0.5 # energy cost per operation 
const_energy_read_per_bit = 0.1 # energy cost reading single bit 
const_energy_write_per_bit = 0.1 # energy cost writing single bit 
const_energy_idle = 0.5
const_number_bits_per_value = 8 # number of bits every value 


# function of estimating the time and energy cost of convolutional layer 
# input
#    - in_height: height of input feature map
#    - in_width: width of input feature map 
#    - in_depth: depth of input feature map
#    - w_k: size of filter w_k * w_k
#    - n_w: number of filters
#    - stride: size of filter stride
# output
#    - total_time_cost: total time cost of convolutional layer
#    - total_energy_cost: total_energy cost of convolutional layer
def estimation_convolutional_time_energy_cost(in_height , in_width , in_depth , w_k , n_w , stride):
    # calculate computing time
    computing_time = calc_computing_time(in_height , in_width , in_depth , w_k , n_w , stride)
    # calculate memory accessing time 
    memory_accessing_time = calc_memory_accessing_time(in_height , in_width , in_depth , w_k , n_w, stride)
    # total time cost equals to the sum of computing time and memory time
    total_time_cost = computing_time + memory_accessing_time;

    #calculate computing energy
    computing_energy = calc_computing_energy(in_height , in_width , in_depth , w_k , n_w , stride)
    #calculete memory energy
    memory_energy = calc_memory_energy(in_height , in_width , in_depth , w_k , n_w , stride)
    #calculate idel energy
    idle_energy = calc_idle_energy(total_time_cost , computing_time)
    #compute energy cost in conv layer
    total_energy_cost = computing_energy + memory_energy + idle_energy
    
    # return time cost and energy cost
    return [total_time_cost, total_energy_cost]

    
# function of calculating computing time
def calc_computing_time(in_height , in_width , in_depth , w_k , n_w , stride):

    computing_time = in_height * in_width * n_w * in_depth * w_k * w_k * const_CPOP

    computing_time = computing_time / (const_number_PE * const_frequency * stride**2)

    
    return computing_time   


# function of calculating memory time
def calc_memory_accessing_time(in_height , in_width , in_depth , w_k , n_w, stride):
    
    reading_input_time = (in_height * in_width * in_depth + w_k**2 * n_w) * const_average_read_time / const_number_channel

    writing_output_time = (in_height * in_width * n_w) / (stride**2) * const_average_write_time / const_number_channel

    memory_accessing_time = reading_input_time + writing_output_time

    
    return memory_accessing_time
    

# function of calculating computing energy
def calc_computing_energy(in_height , in_width , in_depth , w_k , n_w , stride):
    
    computing_energy = (in_height * in_width * in_depth * w_k**2 * n_w) * const_energy_op / (stride**2)

    
    return computing_energy
    
    
# function of calculating memory energy
def calc_memory_energy(in_height , in_width , in_depth , w_k , n_w , stride):
    
    reading_memory_energy = (in_height * in_width * in_depth + w_k**2 * n_w) * const_number_bits_per_value * const_energy_read_per_bit
    
    writing_memory_energy = (in_height * in_width * n_w) / (stride**2) * const_number_bits_per_value * const_energy_write_per_bit

    memory_energy = reading_memory_energy + writing_memory_energy

    
    return memory_energy

# function of calculating idle energy
def calc_idle_energy(total_time_cost , computing_time):
    
    idle_energy = (total_time_cost - computing_time) * const_number_PE * const_energy_idle
    
    return idle_energy
