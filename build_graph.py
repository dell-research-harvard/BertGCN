import os
import random
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
import inspect


def load_and_shuffle_data(dataset):

    print("Opening and shuffling data...")

    # Check dataset
    if len(sys.argv) != 2:
        sys.exit("Use: python build_graph.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']

    if dataset not in datasets:
        sys.exit("wrong dataset name")

    # Pull text ids, split (test/train) and label
    doc_name_list = []
    splits = {'train':
                  {'names': [], 'ids': []},
              'test':
                  {'names': [], 'ids': []}
              }

    f = open('data/' + dataset + '.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:      # find returns -1 if not found
            splits['test']['names'].append(line.strip())
        elif temp[1].find('train') != -1:
            splits['train']['names'].append(line.strip())
    f.close()

    print(f'{len(splits["train"]["names"])} training examples')
    print(f'{len(splits["test"]["names"])} test examples')

    # Pull texts
    doc_content_list = []
    f = open('data/corpus/' + dataset + '.clean.txt', 'r')    # This is the version with stop words removed
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

    assert len(doc_content_list) == len(doc_name_list)

    # Shuffle data
    for split in ["train", "test"]:

        for split_name in splits[split]['names']:
            split_id = doc_name_list.index(split_name)
            splits[split]['ids'].append(split_id)
        random.shuffle(splits[split]['ids'])
        # Todo: check that this does actually shuffle

        # # partial labeled data # Todo: look more into this
        # #train_ids = train_ids[:int(0.2 * len(train_ids))]

        split_ids_str = '\n'.join(str(index) for index in splits[split]['ids'])
        f = open(f'data/{dataset}.{split}.index', 'w')
        f.write(split_ids_str)
        f.close()

    ids = splits['train']['ids'] + splits['test']['ids']

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('data/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()

    return shuffle_doc_name_list, shuffle_doc_words_list, splits['train']['ids'], splits['test']['ids']


def create_vocab_list(shuffle_doc_words_list, dataset):

    # Build vocab
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)

    vocab = list(word_set)
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")

    vocab_str = '\n'.join(vocab)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    # Dictionary dictionary mapping words to unique ids
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    return vocab, vocab_size, word_id_map


def create_node_vectors(
        shuffle_doc_name_list,
        shuffle_doc_words_list,
        train_ids,
        test_ids,
        word_embeddings_dim,
        vocab_size,
        dataset
):

    print("Creating node vectors...")

    word_vector_map = {} # Todo: might not need this if get rid of all the redundant code

    # Create list of unique labels
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    f = open('data/corpus/' + dataset + '_labels.txt', 'w')
    f.write(label_list_str)
    f.close()

    # Split 10% of the training data off to be the eval set
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    # different training rates

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()

    # Todo: replace with a one line (zeros) -> matrix of real_train_size word_embeddings_dim - can replace up to 264
    # Create feature matrix, x,  for training docs. At the moment, we don't have any
    # features so they're all intialised as zero
    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)]) # Todo: remove
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:  # Todo: remove
            if word in word_vector_map:  # This does not do anything because word_vector_map is an empty dict
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len # Todo: just append 0.0

    # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    # Todo: this should be in a function - do it 3 times
    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)

    # tx: Do the same for the training data
    test_size = len(test_ids)

    # Todo: rewrite, as above
    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)

    # allx: do the same thing, but with the train and eval samples
    # (a superset of x)
    row_allx = []
    col_allx = []
    data_allx = []
    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

    # Also add in the length of the vocab
    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))

    # Todo: This does nothing - delete
    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            print("true")
            vector = word_vector_map[word]
            word_vectors[i] = vector

    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j))) # Todo: this does nothing

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    print("Featurized matrix sizes:", x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    f = open("data/ind.{}.x".format(dataset), 'wb')
    pkl.dump(x, f)
    f.close()

    f = open("data/ind.{}.y".format(dataset), 'wb')
    pkl.dump(y, f)
    f.close()

    f = open("data/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(tx, f)
    f.close()

    f = open("data/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(ty, f)
    f.close()

    f = open("data/ind.{}.allx".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open("data/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()


def create_edges(shuffle_doc_words_list, vocab, vocab_size, word_id_map, window_size, dataset):

    '''
    Calculate PMI, for word-word edges
    '''

    print("Creating word-word edges...")

    # Create list of sliding windows, moving by one word each time
    windows = []
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

    # Number of windows that a word appears in
    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    # Frequencies of co-occurance of word pairs
    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    # Calcuate PMI
    row = []
    col = []
    weight = []

    num_window = len(windows)
    train_size = len(train_ids)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    '''
    Calculate TF-IDF, for document-word edges 
    '''

    print("Creating document-word edges...")

    # dict of the number of times each word is used in entire corpus
    word_freq = {}
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # dictionary of words to list of ids of all texts that use that word
    word_doc_list = {}
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    # dictionary of words and number of texts that use that word
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # doc word frequency
    doc_word_freq = {}
    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    # Calculate TF_IDF
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    '''
    Put this all together in a graph and save  
    '''

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    print("Weighted edge matrix size:", adj.shape)

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()


if __name__ == '__main__':

    dataset_name = sys.argv[1]

    shuffle_doc_name_list, shuffle_doc_words_list, train_ids, test_ids = load_and_shuffle_data(dataset=dataset_name)

    vocab, vocab_size, word_id_map = create_vocab_list(shuffle_doc_words_list, dataset=dataset_name)

    create_node_vectors(
        shuffle_doc_name_list,
        shuffle_doc_words_list,
        train_ids,
        test_ids,
        word_embeddings_dim=300,
        vocab_size=vocab_size,
        dataset=dataset_name
    )

    create_edges(shuffle_doc_words_list, vocab, vocab_size, word_id_map, window_size=20, dataset=dataset_name)
