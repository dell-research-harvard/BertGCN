import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import sys
from tqdm import tqdm
import json

from utils import *


def load_and_shuffle_data(dataset):

    print("\n Opening and shuffling data...")

    # Check dataset
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    if dataset not in datasets:
        sys.exit("wrong dataset name")

    # Pull text ids, split (test/train) and label
    doc_name_list = []
    splits = {'train': {'names': [], 'ids': []}, 'test': {'names': [], 'ids': []}}

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

    print(f"Vocabulary size: {len(vocab)}")

    vocab_str = '\n'.join(vocab)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
    f.write(vocab_str)
    f.close()

    # Dictionary mapping words to unique ids
    word_id_map = {}
    for i in range(len(vocab)):
        word_id_map[vocab[i]] = i

    return vocab, word_id_map


def create_nodes(
        shuffle_doc_name_list,
        train_ids,
        test_ids,
        word_embeddings_dim,
        vocab,
        dataset
):

    print("\n Creating node vectors...")

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

    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w')
    f.write(real_train_doc_names_str)
    f.close()

    # Dictionary of number of useful things
    count = {
        'total nodes': len(train_ids) + len(test_ids) + len(vocab),
        'train nodes': real_train_size,
        'val nodes': len(train_ids) - real_train_size,
        'test nodes': len(test_ids),
        'word nodes': len(vocab),
        'classes': len(label_list)
    }

    with open('data/' + dataset + '.count.json', 'w') as f:
        json.dump(count, f, indent=4)

    print('Data: {}'.format(str(count)))

    def node_matrix_creation(
            data_length,
            start_index=0,
            add_vocab=False,
            vocab_length=0,
            save_prefix=""
    ):

        # Create feature matrix, x,  for training docs. At the moment, we don't have any
        row_x = list(range(data_length)) * word_embeddings_dim
        tmp_col_x = [[dim] * data_length for dim in range(word_embeddings_dim)]
        col_x = [x for l in tmp_col_x for x in l]
        data_x = [0.0] * (data_length * word_embeddings_dim)

        if add_vocab:

            row_x = [int(i) for i in row_x]

            word_vectors = np.random.uniform(-0.01, 0.01, (vocab_length, word_embeddings_dim))

            for i in range(vocab_length):
                for j in range(word_embeddings_dim):
                    row_x.append(int(i + data_length))
                    col_x.append(j)
                    data_x.append(word_vectors.item((i, j)))

            row_x = np.array(row_x)
            col_x = np.array(col_x)
            data_x = np.array(data_x)

            x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(data_length + vocab_length, word_embeddings_dim))

        else:
            x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(data_length, word_embeddings_dim))

        # Create y, a sparse matrix of labels
        y = []
        for i in range(data_length):
            doc_meta = shuffle_doc_name_list[i + start_index]
            temp = doc_meta.split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)

        if add_vocab:
            for i in range(vocab_length):
                one_hot = [0 for l in range(len(label_list))]
                y.append(one_hot)

        y = np.array(y)

        print("Featurized matrix sizes:", x.shape, y.shape)

        f = open(f"data/ind.{dataset}.{save_prefix}x", 'wb')
        pkl.dump(x, f)
        f.close()

        f = open(f"data/ind.{dataset}.{save_prefix}y", 'wb')
        pkl.dump(y, f)
        f.close()

    # Create and save node matrices for training data
    node_matrix_creation(
        data_length=real_train_size,
        save_prefix=""
    )

    # Create and save node matrices for test data
    node_matrix_creation(
        data_length=len(test_ids),
        start_index=len(train_ids),
        save_prefix="t"
    )

    # Create and save node matrices for train + eval data + vocab together
    node_matrix_creation(
        data_length=len(train_ids),
        add_vocab=True,
        vocab_length=len(vocab),
        save_prefix="all"
    )


def create_edges(shuffle_doc_words_list, vocab, word_id_map, window_size, dataset):

    '''
    Calculate PMI, for word-word edges
    '''

    print("\n Creating word-word edges...")

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
    for window in tqdm(windows):
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
    test_size = len(test_ids)

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

    print("\n Creating document-word edges...")

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
                row.append(i + len(vocab))
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    '''
    Put this all together and save  
    '''

    node_size = train_size + len(vocab) + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    # Make symmetric across main diagonal, by taking largest value
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    print("Weighted edge matrix size:", adj.shape)

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj_norm, f)
    f.close()


if __name__ == '__main__':

    if len(sys.argv) != 2: sys.exit("Use: python build_graph.py <dataset>")
    dataset_name = sys.argv[1]

    shuffle_doc_name_list, shuffle_doc_words_list, train_ids, test_ids = load_and_shuffle_data(dataset=dataset_name)

    vocab, word_id_map = create_vocab_list(shuffle_doc_words_list, dataset=dataset_name)

    create_nodes(
        shuffle_doc_name_list,
        train_ids,
        test_ids,
        word_embeddings_dim=300,
        vocab=vocab,
        dataset=dataset_name
    )

    create_edges(shuffle_doc_words_list, vocab, word_id_map, window_size=20, dataset=dataset_name)
