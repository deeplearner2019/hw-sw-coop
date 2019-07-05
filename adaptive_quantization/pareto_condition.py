import os
import math
import collections


# extract activation config and output error.
list_act_sizes = [2080, 2176, 2432]
results_act = collections.defaultdict(list)
activations_error_dir = './activations_error/results/'
error_files = os.listdir(activations_error_dir)
for error_file in error_files:
    layer_id = int(error_file.split('_')[-2])
    dead_zone = -1
    quant_levels = int(error_file.split('_')[-1])
    with open(os.path.join(activations_error_dir, error_file), 'r') as f:
        error = float(f.readline())
    if error > 1:
        continue
    results_act[layer_id].append((layer_id, dead_zone, quant_levels,
                                  list_act_sizes[layer_id] * math.log(quant_levels, 2), error))

# extract weights config and output error.
list_weights_sizes = [18432, 65536, 131072, 14592]
results_w = collections.defaultdict(list)
weights_error_dir = './weights_error/results/'
error_files = os.listdir(weights_error_dir)
for error_file in error_files:
    layer_id = int(error_file.split('_')[-3])
    dead_zone = int(error_file.split('_')[-2])
    quant_levels = int(error_file.split('_')[-1])
    with open(os.path.join(weights_error_dir, error_file), 'r') as f:
        error = float(f.readline())
    if error > 1:
        continue
    results_w[layer_id].append((layer_id, dead_zone, quant_levels,
                                list_weights_sizes[layer_id] * math.log(quant_levels, 2), error))

# generate pareto condition inputs
num_weights_layers = 4
num_act_layers = 3
pareto_condition_inputs = 'pareto_condition_inputs.txt'
with open(pareto_condition_inputs, 'w') as f:
    for i in xrange(num_weights_layers):
        f.write(str(len(results_w[i])) + '\n')
        for ele in results_w[i]:
            f.write(' '.join([str(e) for e in ele]))
            f.write('\n')

    for i in xrange(num_act_layers):
        f.write(str(len(results_act[i])) + '\n')
        for ele in results_act[i]:
            f.write(' '.join([str(e) for e in ele]))
            f.write('\n')

# draw pareto condition curves.
import matplotlib.pyplot as plt

fig_w = plt.figure()
for i in xrange(num_weights_layers):
    lengths = []
    errors = []
    for ele in results_w[i]:
        length, error = ele[-2], ele[-1]
        if error > 1:
            continue
        lengths.append(length/1000)
        errors.append(error)

    plt.plot(lengths[20:], errors[20:], 'o', label="layer " + str(i))

plt.ylabel("output error")
plt.xlabel("weight sizes (KB)")
plt.legend()
plt.show()
fig_w.savefig('pareto_condition_weights.jpg')

fig_act = plt.figure()
for i in xrange(num_act_layers):
    lengths = []
    errors = []
    for ele in results_act[i]:
        length, error = ele[-2], ele[-1]
        if error > 1:
            continue
        lengths.append(length/1000)
        errors.append(error)

    plt.plot(lengths, errors, 'o', label="layer " + str(i))

plt.ylabel("output error")
plt.xlabel("activation sizes (KB)")
plt.legend()
plt.show()
fig_act.savefig('pareto_condition_activations.jpg')
