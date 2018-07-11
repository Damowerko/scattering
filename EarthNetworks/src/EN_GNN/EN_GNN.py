"""
Pooling. Applied to 20NEWS dataset. Version 02.
2018/04/15
Based on Source Localization Version 21.

TRAINING CONCLUSIONS (See all the way down):
## Learning rate: 0.001, Reg = 0
These are the parameters to keep with ADAM.
"""

import sys, os

sys.path.insert(0, '..')
from cnngs import architecture, graphtools, manager
import tensorflow as tf
import scipy.sparse
import numpy as np
import time
from EN_GNN.graph import get_distance_graph
from EN_GNN.data import pick_greedy, import_data
from sklearn.model_selection import train_test_split

"""
Output
"""
log = False
this_filename = "Results/gnn_out"

"""
SIMULATION SELECTION
"""

# Graph:

GSO = 'norm-Laplacian'  # 'Adjacency', 'max2-Laplacian', 'norm-Laplacian'
# Pre-training graph operations:
do_clustering = True  # Obtain clustering GSOs
do_degree = True  # Obtain nodes selected based on degree
overlap_K = 0  # Number of shifts on where not to consider overlap
# (If K=0 then it is simply degree based ordering)

# Training:

train_method = 'ADAM'  # 'SGD' or 'ADAM'
do_validation = False

# Presentation:

print_data_summary = False

# Methods:

do_c_a = False  # Clustering with our code
do_np = False  # No pooling
do_sp = True  # Selection Pooling
do_ap = True  # Aggregation Pooling
do_hp = True  # Hybrid Pooling (multinode)

# Fields
fields = ['PressureSeaLevelMBar', "TemperatureC", "WindSpeedKph", "PressureSeaLevelMBarRatePerHour"]
kron = np.eye(len(fields))

"""
PARAMETERS SELECTION
"""

print("Setting up problem parameters...", end=" ", flush=True)

# Graph:
N = 25  # Number of nodes
if do_clustering:
    coarsening_levels = 2  # Same as number of layers

# Training:
N_batch = 100  # Number of samples in the batch

# Training split size:
test_split = 0.1

# Specific to the dataset:
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'number_edges', 16,
    'Graph: minimum number of edges per vertex.')
flags.DEFINE_string(
    'metric', 'cosine',
    'Graph: similarity measure (between features).')
flags.DEFINE_bool(
    'normalized_laplacian', True,
    'Graph Laplacian: normalized.')
if do_clustering:
    flags.DEFINE_integer(
        'coarsening_levels', coarsening_levels,
        'Number of coarsened graphs.')
flags.DEFINE_string(
    'dir_data', os.path.join('..', 'data', '20news'),
    'Directory to store data.')
flags.DEFINE_integer(
    'val_size', 400,
    'Size of the validation set.')

l_short_docs = 5  # 5 in the original 20news file, no mention in the paper.

common = {}
common['num_epochs'] = 100  # 20 in the paper (for ADAM), 80 in the source
# file (for SGD)
common['learning_rate'] = 0.001  # This was set in the paper. It was 0.1 in
# the cgconv_softmax in the source file. The other methods will remain the
# same as they are.
common['decay_rate'] = 0.999  # Nothing says anywhere about this in the
# paper. I believe this is for the SGD. I left it at 0.999 which is the one
# that was set for the cgconv. The other ones that were also 0.999 have
# been removed from params. The ones that were different were kept.
common['momentum'] = 0  # Determines whether training is done following
# train_method or with momentum training.

common['regularization'] = 0  # Only for the graph CNN that has no FC layer
# the rest were originally set to zero. The paper says there is
# regularization and this is the value that was found in the source file.
common['dropout'] = 0.5  # Nothing says about dropout in the paper. Only
# in the MNIST. In the source file they were all set to 1 (no dropout).
common['batch_size'] = N_batch
common['eval_frequency'] = 5 * common['num_epochs']

common['dir_name'] = "foo"  # TODO

common['GSO'] = GSO
common['train_method'] = train_method

print("DONE")

"""
GRAPH CREATION & DATA HANDLING
"""

print("Gathering data...", end=" ", flush=True)

stations = pick_greedy(n_stations=N)
data, labels = import_data(fields, stations)  # type: (np.ndarray, np.ndarray)
data = data.astype(dtype=np.float32)
labels = labels.astype(dtype=np.int32).flatten()

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_split)

if print_data_summary:
    print(" ")
    print("DATA Summary:")
    print("    train_data:")
    print("        type = {}".format(type(train_data)))
    print("        dtype = {}".format(train_data.dtype))
    print("        shape = {}".format(train_data.shape))
    print("        average = {:.4}".format(train_data.mean()))
    print("        min = {:.4}".format(train_data.min()))
    print("        max = {:.4}".format(train_data.max()))
    print("    train_labels:")
    print("        type = {}".format(type(train_labels)))
    print("        dtype = {}".format(train_labels.dtype))
    print("        shape = {}".format(train_labels.shape))
    print("        average = {:.4}".format(train_labels.mean()))
    print("        min = {}".format(train_labels.min()))
    print("        max = {}".format(train_labels.max()))
    print("    test_data:")
    print("        type = {}".format(type(test_data)))
    print("        dtype = {}".format(test_data.dtype))
    print("        shape = {}".format(test_data.shape))
    print("        average = {:.4}".format(test_data.mean()))
    print("        min = {:.4}".format(test_data.min()))
    print("        max = {:.4}".format(test_data.max()))
    print("    test_labels:")
    print("        type = {}".format(type(test_labels)))
    print("        dtype = {}".format(test_labels.dtype))
    print("        shape = {}".format(test_labels.shape))
    print("        average = {:.4}".format(test_labels.mean()))
    print("        min = {}".format(test_labels.min()))
    print("        max = {}".format(test_labels.max()))
    print(" ")

"""
FEATURE GRAPH
"""

print("Building graph support...", end=" ", flush=True)
A = get_distance_graph()
A = A[stations.flatten(), :][: , stations.flatten()]  # type: np.ndarray
A = A.astype(dtype=np.float32)
A = scipy.sparse.csr_matrix(A)

t_start = time.process_time()
if GSO == 'Adjacency':
    S = A
else:
    S = graphtools.laplacian(A, normalized=True)
    if GSO == 'max2-Laplacian':
        S = graphtools.rescale_L(S, lmax=2)

if do_clustering:
    graphs_c, perm_c = graphtools.coarsen(
        A, levels=FLAGS.coarsening_levels, self_connections=False)
    if GSO == 'Adjacency':
        S_c = graphs_c
    else:
        L_c = [graphtools.laplacian(A, normalized=True) for A in graphs_c]
        if GSO == 'norm-Laplacian':
            S_c = L_c
        if GSO == 'max2-Laplacian':
            S_c = [graphtools.rescale_L(L, lmax=2) for L in L_c]

if do_degree:
    # REMEMBER: Input the true adjacency matrix!!
    perm_d = graphtools.degree_order(A, overlap_K)
    A_d = A[perm_d][:, perm_d]
    if GSO == 'Adjacency':
        S_d = A_d
    else:
        S_d = graphtools.laplacian(A_d, normalized=True)
        if GSO == 'max2-Laplacian':
            S_d = graphtools.rescale_L(S_d, lmax=2)

if do_clustering:
    L = train_data.shape[2]
    train_data_c = graphtools.perm_data(train_data, perm_c)
    test_data_c = graphtools.perm_data(test_data, perm_c)

if do_degree:
    train_data_d = train_data[:][:, perm_d, :]
    test_data_d = test_data[:][:, perm_d, :]

# Validation set.
if do_validation:
    # Not implemented yet:
    val_data = train_data[:FLAGS.val_size, :, :]
    val_labels = train_labels[:FLAGS.val_size, :]
    train_data = train_data[FLAGS.val_size:, :, :]
    train_labels = train_labels[FLAGS.val_size:, :]
else:
    val_data = test_data
    val_labels = test_labels
    if do_clustering:
        val_data_c = test_data_c
    if do_degree:
        val_data_d = test_data_d

print("DONE")

print("Running Neural Networks: BEGINNING")

common['decay_steps'] = len(train_labels) / N_batch
common['decay_steps'] = len(train_labels) / N_batch
C = max(train_labels) + 1  # number of classes

if do_c_a or do_np or do_sp or do_ap or do_hp:
    model_manager = manager.model_manager()

if do_c_a:
    name = 'c_cheb_a'
    print(" ")
    print("Training model: {}".format(name))
    print(" ")
    params = common.copy()
    params['S'] = S_c
    params['V'] = len(fields)

    params['archit'] = 'clustering'
    params['filter'] = 'chebyshev5'
    params['pool'] = 'maxpool'
    params['nonlin'] = 'b1relu'

    params['K'] = [5, 5]
    params['F'] = [32, 32]
    params['a'] = [2, 2]
    params['M'] = [C]

    params['name'] = name
    params['dir_name'] += '_' + name + '/'

    model_manager.test(architecture.cnngs(**params), name, params,
                       train_data_c, train_labels, val_data_c, val_labels,
                       test_data_c, test_labels)

if do_np:
    name = 'np_3'
    print(" ")
    print("Training model: {}".format(name))
    print(" ")
    params = common.copy()
    params['S'] = S
    params['V'] = len(fields)

    params['archit'] = 'no-pooling'
    params['filter'] = 'lsigf'
    params['pool'] = 'maxpool'
    params['nonlin'] = 'b1relu'

    params['K'] = [5, 5]
    params['F'] = [32, 32]
    params['a'] = [2, 2]
    params['M'] = [C]

    params['name'] = name
    params['dir_name'] += '_' + name + '/'

    model_manager.test(architecture.cnngs(**params), name, params,
                       train_data, train_labels, val_data, val_labels,
                       test_data, test_labels)

if do_sp:
    name = 'selection_pooling'
    print(" ")
    print("Training model: {}".format(name))
    print(" ")
    params = common.copy()
    params['S'] = [S_d, [15, 5]]  # the number of nodes at each layer, if fewer layers than elements in list, final parameter specifies downsampling at the end
    params['V'] = len(fields)

    params['archit'] = 'selection'
    params['filter'] = 'lsigf'
    params['pool'] = 'maxpool'
    params['nonlin'] = 'b1relu'

    params['K'] = [5, 5]  # number of filter taps
    params['F'] = [32, 32]  # number of features
    params['a'] = [2, 4]  # size of the pooling, n-hop pooling
    params['M'] = [C]

    params['name'] = name
    params['dir_name'] += '_' + name + '/'

    model_manager.test(architecture.cnngs(**params), name, params,
                       train_data_d, train_labels, val_data_d, val_labels,
                       test_data_d, test_labels)

if do_ap:
    name = 'aggregation_pooling'
    print(" ")
    print("Training model: {}".format(name))
    print(" ")
    params = common.copy()
    params['S'] = [S_d, [1]]  # the node to use for aggregation (nth highest degree)
    params['V'] = len(fields)

    params['archit'] = 'aggregation'
    params['filter'] = 'lsigf'
    params['pool'] = 'maxpool'
    params['nonlin'] = 'b1relu'

    params['K'] = [4, 4]
    params['F'] = [32, 32]
    params['a'] = [2, 2]
    params['M'] = [C]

    params['name'] = name
    params['dir_name'] += '_' + name + '/'

    model_manager.test(architecture.cnngs(**params), name, params,
                       train_data_d, train_labels, val_data_d, val_labels,
                       test_data_d, test_labels)

if do_hp:
    name = 'hybrid_pooling'
    print(" ")
    print("Training model: {}".format(name))
    print(" ")
    params = common.copy()
    params['S'] = [S_d, [25, 10], [15, 10]]  # subset of nodes to look at , number of exchanges you do
    params['V'] = len(fields)

    params['archit'] = 'hybrid'
    params['filter'] = 'lsigf'
    params['pool'] = 'maxpool'
    params['nonlin'] = 'b1relu'

    params['K'] = [[3, 3], [3, 3]]
    params['F'] = [[4, 8], [8, 16]]
    params['a'] = [[2, 2], [2, 2]]
    params['M'] = [C]

    params['name'] = name
    params['dir_name'] += '_' + name + '/'

    model_manager.test(architecture.cnngs(**params), name, params,
                       train_data_d, train_labels, val_data_d, val_labels,
                       test_data_d, test_labels)

print(" ")

print("Showing results...")

print(" ")

print("    {{n = {}, {}, num_epochs = {}, batch_size = {}, ".
      format(train_data.shape[1], common['GSO'],
             common['num_epochs'], common['batch_size']))
print("     reg = {}, dropout = {}, momentum = {}".
      format(common['regularization'], common['dropout'],
             common['momentum']))
if train_method == "SGD":
    print("     {}, learning_rate = {}, decay_rate = {}}}".
          format(common['train_method'], common['learning_rate'],
                 common['decay_rate']))
elif train_method == "ADAM":
    print("     {}, learning_rate = {}}}".
          format(common['train_method'], common['learning_rate']))

print(" ")

if do_c_a or do_np or do_sp or do_ap or do_hp:
    model_manager.show()
    print(" ")

print(" ")
print("Clustering graph sizes:")
for i in range(len(S_c)):
    print('S_c[{}]: {}'.format(i, S_c[i].shape[0]))

if log:
    ff = open(this_filename + '-DONE', 'a');
    ff.write('DONE.')
    ff.write('\r\n')
    ff.close()

if False:
    grid_params = {}
    data = (train_data, train_labels, val_data, val_labels, test_data,
            test_labels)
    manager.grid_search(params, grid_params, *data,
                        model=lambda x: architecture.cnngs(**x))
"""
A=np.array([\
		[0.,0.,1.,0.,1.,1.,0.,0.],\
		[0.,0.,0.,0.,1.,1.,0.,1.],\
		[1.,0.,0.,0.,1.,0.,1.,1.],\
		[0.,0.,0.,0.,0.,0.,0.,1.],\
		[1.,1.,1.,0.,0.,1.,0.,0.],\
		[1.,1.,0.,0.,1.,0.,1.,1.],\
		[0.,0.,1.,0.,0.,1.,0.,1.],\
		[0.,1.,1.,1.,0.,1.,1.,0.]])

  {n = 1000, max2-Laplacian, num_epochs = 40, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {F = [[16, 16], [16, 32]], K = [[4, 4], [4, 4]], M = [20]}
    selection_pooling = {F = [32, 32], K = [5, 5], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    17.79 18.23   13.49 13.67     43904         81       aggregation_pooling
    36.97 41.12   35.25 39.37    294400        742       hybrid_pooling
    20.69 21.76   16.05 17.18     37280        3142       selection_pooling



{n = 1000, max2-Laplacian, num_epochs = 40, batch_size = 100, 
     reg = 0.001, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [5, 5], M = [20]}
    hybrid_pooling = {F = [[16, 16], [16, 32]], K = [[4, 4], [4, 4]], M = [20]}
    selection_pooling = {F = [32, 32], K = [5, 5], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    15.47 16.83    9.82 11.01     44960         81       aggregation_pooling
    34.34 37.91   32.49 35.93    294400        740       hybrid_pooling
    14.62 15.92   10.58 11.43     37280        3159       selection_pooling


{n = 1000, max2-Laplacian, num_epochs = 40, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [5, 5], M = [20]}
    hybrid_pooling = {F = [[16, 32], [32, 32]], K = [[4, 4], [4, 4]], M = [20]}
    selection_pooling = {F = [32, 32], K = [5, 5], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    16.74 18.38   12.37 13.82     44960         73       aggregation_pooling
    37.48 41.66   35.97 39.87    652800        1427       hybrid_pooling
    15.07 15.57   11.22 11.17      8480        590       selection_pooling

 {n = 1000, max2-Laplacian, num_epochs = 80, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}
    selection_pooling = {F = [32, 32], K = [5, 5], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    17.30 19.54   12.95 14.89    164224         74       aggregation_pooling
    40.73 45.40   39.87 44.74    228800        701       hybrid_pooling
    16.79 18.73   12.81 14.84     11680        1163       selection_pooling


{n = 1000, norm-Laplacian, num_epochs = 80, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}
    selection_pooling = {F = [32, 32], K = [5, 5], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    13.34 14.67   10.57 11.80    164224         75       aggregation_pooling
    42.14 67.20   42.28 67.18    228800        717       hybrid_pooling
    17.22 17.83   13.82 14.24     11680        1138       selection_pooling

{n = 3000, norm-Laplacian, num_epochs = 80, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}
    selection_pooling = {F = [32, 32], K = [5, 5], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    11.44 12.58    8.38  9.28    484224        200       aggregation_pooling
    39.63 53.17   39.27 52.85    228800        2065       hybrid_pooling
    16.83 17.72   12.57 13.65     11680        3772       selection_pooling

{n = 3000, norm-Laplacian, num_epochs = 160, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {P = [100,50] , Q = [10,5], F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    12.47 13.57    9.03  9.92    484224        178       aggregation_pooling
    41.26 73.18   40.56 72.75    228800        2033       hybrid_pooling

{n = 3000, norm-Laplacian, num_epochs = 160, batch_size = 100, 
     reg = 0, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {P = [300,150] , Q = [15,10], F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    10.27 11.20    7.40  8.26    484224        179       aggregation_pooling
    45.16 99.93   45.26 99.93    686400        5044       hybrid_pooling

{n = 3000, norm-Laplacian, num_epochs = 100, batch_size = 100, 
     reg = 0.001, dropout = 0.5, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {P = [300,150] , Q = [15,10], F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    11.65 13.07    7.71  8.88    484224        188       aggregation_pooling
    44.14 96.11   44.07 96.11    686400        5159       hybrid_pooling

{n = 3000, norm-Laplacian, num_epochs = 100, batch_size = 100, 
     reg = 0.001, dropout = 0.25, momentum = 0
     ADAM, learning_rate = 0.001}

    aggregation_pooling = {F = [32, 32], K = [4, 4], M = [20]}
    hybrid_pooling = {P = [300,150] , Q = [15,10], F = [[16, 16], [16, 32]], K = [[3, 3], [3, 3]], M = [20]}

    Results:
      accuracy        F1        parameters    time [ms]  name
    test  train   test  train   
    12.20 13.34    7.98  8.92    484224        188       aggregation_pooling
    45.07 97.41   45.23 97.41    686400        5139       hybrid_pooling


"""
