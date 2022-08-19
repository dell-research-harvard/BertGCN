import numpy as np
import pickle as pkl
import sys
import logging
import os
import json

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

    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: ~todo: update
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

    names = ['labels', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    labels, adj = tuple(objects)

    # load documents
    corpus_file = './data/corpus/'+dataset_str+'_shuffle.txt'
    with open(corpus_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')

    # Lengths of things
    with open('data/' + dataset_str + '.count.json') as f:
        count = json.load(f)

    # Masks
    train_mask = sample_mask(range(count['train nodes']), count['total nodes'])
    val_mask = sample_mask(range(count['train nodes'], count['train nodes']+count['val nodes']), count['total nodes'])
    test_mask = sample_mask(range(count['total nodes']-count['test nodes'], count['total nodes']), count['total nodes'])
    doc_mask = train_mask + val_mask + test_mask

    # Reformat labels

    # transform one-hot label to class ID for pytorch computation
    y = labels.argmax(axis=1)

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_train = y_train.argmax(axis=1)

    temp_y = th.LongTensor(y)
    label_dict = {
        'train': temp_y[:count['train nodes']],
        'val': temp_y[count['train nodes']:count['train nodes'] + count['val nodes']],
        'test': temp_y[-count['test nodes']:]
    }

    # create index loader
    if batch_size:
        train_idx = Data.TensorDataset(th.arange(0, count['train nodes'], dtype=th.long))
        val_idx = Data.TensorDataset(th.arange(count['train nodes'], count['train nodes'] + count['val nodes'], dtype=th.long))
        test_idx = Data.TensorDataset(th.arange(count['total nodes'] - count['test nodes'], count['total nodes'], dtype=th.long))
        doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

        idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
        idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
        idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
        idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

        return y, y_train, train_mask, val_mask, test_mask, doc_mask, idx_loader_train, idx_loader_val, idx_loader_test, \
               idx_loader, adj, text, count, label_dict

    else:
        return y, y_train, train_mask, val_mask, test_mask, doc_mask, adj, text, count, label_dict
