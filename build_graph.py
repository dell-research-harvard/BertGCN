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

    """
    Open data
    """

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

        # # partial labeled data
        # #train_ids = train_ids[:int(0.2 * len(train_ids))]

    ids = splits['train']['ids'] + splits['test']['ids']

    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])

    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_str)
    f.close()

    # Split 10% of the training data off to be the eval set
    train_size = len(splits['train']['ids'])
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)

    # Dictionary of counts of useful things
    count = {
        'train nodes': real_train_size,
        'val nodes': val_size,
        'test nodes': len(splits['test']['ids']),
    }

    """
    Create list of all words in data
    """

    # Build vocab
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)

    vocab = list(word_set)

    print(f"Vocabulary size: {len(vocab)}")

    # Dictionary mapping words to unique ids
    word_id_map = {}
    for i in range(len(vocab)):
        word_id_map[vocab[i]] = i

    # Add counts to count dict
    count['total nodes'] = train_size + len(splits['test']['ids']) + len(vocab)
    count['word nodes'] = len(vocab)


    """
    Create sparse label matrix 
    """

    print("\n Creating label matrix...")

    # Create list of unique labels
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    # Add to count dict
    count['classes'] = len(label_list)

    # Create a sparse matrix of labels
    labels = []
    for i in range(count['train nodes'] + count['val nodes'] + count['test nodes']):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(count['classes'])]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        labels.append(one_hot)

    labels = np.array(labels)

    print("Label matrix size:", labels.shape)

    f = open(f"data/ind.{dataset}.labels", 'wb')
    pkl.dump(labels, f)
    f.close()

    # Save count dict
    with open('data/' + dataset + '.count.json', 'w') as f:
        json.dump(count, f, indent=4)

    print('Data: {}'.format(str(count)))

    return shuffle_doc_words_list, vocab, word_id_map, count


def create_edges(shuffle_doc_words_list, vocab, word_id_map, count, window_size, dataset):

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
    train_size = count['train nodes']
    test_size = count['test nodes']

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

    if len(sys.argv) != 2:
        sys.exit("Use: python build_graph.py <dataset>")
    dataset_name = sys.argv[1]

    shuffle_doc_words_list, vocab, word_id_map, count = load_and_shuffle_data(dataset=dataset_name)

    create_edges(shuffle_doc_words_list, vocab, word_id_map, count, window_size=20, dataset=dataset_name)
