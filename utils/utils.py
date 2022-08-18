import numpy as np
import pickle as pkl
import sys
import logging
import os

import scipy.sparse as sp

import torch as th
import torch.utils.data as Data



def set_up_logging(ckpt_dir, args):

    print("\n Set up ...")

    # Set up logging
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logger = logging.getLogger('training logger')
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    logger.info('Arguments: {}'.format(str(args)))
    logger.info('Checkpoints will be saved in {}'.format(ckpt_dir))

    cpu = th.device('cpu')
    gpu = th.device('cuda:0')

    return logger, cpu, gpu


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, length):
    """
    Create mask.
    """
    mask = np.zeros(length)    # array([ 0.,  0.,  0.,  0., ... , 0.])
    mask[idx] = 1              # Fill with 1 in the indices indicated in the idx list
    return np.array(mask, dtype=np.bool)


def load_corpus(dataset_str, batch_size=None):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return:
    - adj: adj (unchanged)
    - features: allx and tx vertically stacked, in lil format
    - y_train: np.array of length=train+val+test and width=n labels, with 1 if train data y is label x, 0 otherwise
    - y_val: np.array of length=train+val+test and width=n labels, with 1 if val data y is label x, 0 otherwise
    - y_test: np.array of length=train+val+test and width=n labels, with 1 if test data y is label x, 0 otherwise
    - train_mask: vector of length train+val+test, where 0:len(train) = 1, 0 otherwise
    - val_mask: vector of length train+val+test, where len(train):len(train) + len(val) = 1, 0 otherwise
    - test_mask: vector of length train+val+test, where len(train) + len(val):end = 1, 0 otherwise
    - train_size: length of train data (inc eval)
    - test_size: length of test data
    """

    # Todo: probably move all of this data cleaning to the graph creation script so it's all in one place

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()       # Stack sparse matrices vertically (row wise)
    labels = np.vstack((ally, ty))                 # .tolil() converts to list of lists

    ## Lengths of things

    print("******")
    print(allx.shape)
    print("******")
    print(tx.shape)

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # compute number of real train/val/test/word nodes and number of classes
    count = {
        'total nodes': features.shape[0],    # Todo: this is the only point where we use features - can maybe just be removed?
        'train nodes': train_mask.sum(),
        'val nodes': val_mask.sum(),
        'test nodes': test_mask.sum()
    }
    count['word nodes'] = count['total nodes'] - count['train nodes'] - count['val nodes'] - count['test nodes']
    count['classes'] = labels.shape[1]

    print('Data: {}'.format(str(count)))



    # document mask used for update feature
    doc_mask = train_mask + val_mask + test_mask

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # load documents
    corpus_file = './data/corpus/'+dataset_str+'_shuffle.txt'
    with open(corpus_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')

    # transform one-hot label to class ID for pytorch computation
    temp_y = th.LongTensor((y_train + y_val + y_test).argmax(axis=1))
    label_dict = {
        'train': temp_y[:count['train nodes']],
        'val': temp_y[count['train nodes']:count['train nodes'] + count['val nodes']],
        'test': temp_y[-count['test nodes']:]
    }

    # transform one-hot label to class ID for pytorch computation
    y = y_train + y_test + y_val
    y_train = y_train.argmax(axis=1)
    y = y.argmax(axis=1)

    # create index loader
    train_idx = Data.TensorDataset(th.arange(0, count['train nodes'], dtype=th.long))
    val_idx = Data.TensorDataset(th.arange(count['train nodes'], count['train nodes'] + count['val nodes'], dtype=th.long))
    test_idx = Data.TensorDataset(th.arange(count['total nodes'] - count['test nodes'], count['total nodes'], dtype=th.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    if batch_size:
        idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
        idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
        idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
        idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

        return y, y_train, train_mask, val_mask, test_mask, doc_mask, idx_loader_train, idx_loader_val, idx_loader_test, \
               idx_loader, adj, text, count, label_dict

    else:
        return y, y_train, train_mask, val_mask, test_mask, doc_mask, adj, text, count, label_dict


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
