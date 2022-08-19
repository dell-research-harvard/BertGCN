import random
import numpy as np
import pandas as pd
import pickle as pkl
from math import log
import sys
from tqdm import tqdm
import json
import re

import scipy.sparse as sp

from nltk.corpus import stopwords
import nltk


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def text_clean(list_of_articles):

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.add("nan")
    stop_words.add(",")
    print(stop_words)

    # Calculate word frequencies (to remove rare words)
    word_freq = {}
    for doc_content in list_of_articles:
        temp = clean_str(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in list_of_articles:
        temp = clean_str(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            # word not in stop_words and word_freq[word] >= 5
            if word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)

    return clean_docs


def custom_open_data(dataset):

    print("\n Opening data...")

    datasets = {
        'train': {'orig': pd.read_csv('/mnt/data01/editorials/train_sets/fifth_set/clean/train.csv')},
        'val': {'orig': pd.read_csv('/mnt/data01/editorials/train_sets/fifth_set/clean/eval.csv')},
        'test': {'orig': pd.read_csv('/mnt/data01/editorials/train_sets/fifth_set/clean/test.csv')}
    }

    shuffle_doc_words_list_orig = []
    corpus_label_list = []
    for split in ["train", "val", "test"]:
        datasets[split]['size'] = len(datasets[split]['orig'])

        shuffle_doc_words_list_orig.extend(datasets[split]['orig']['article'])
        corpus_label_list.extend(datasets[split]['orig']['label'])

    shuffle_doc_words_list_clean = text_clean(shuffle_doc_words_list_orig)

    for i in range(len(shuffle_doc_words_list_orig)):
        print(shuffle_doc_words_list_orig[i])
        print("***")
        print(shuffle_doc_words_list_clean[i])
        print("*************************")

    shuffle_doc_words_orig_str = '\n'.join(shuffle_doc_words_list_orig)
    f = open('data/corpus/' + dataset + '_shuffle_orig.txt', 'w')
    f.write(shuffle_doc_words_orig_str)
    f.close()

    shuffle_doc_words_clean_str = '\n'.join(shuffle_doc_words_list_clean)
    f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
    f.write(shuffle_doc_words_clean_str)
    f.close()

    return shuffle_doc_words_list_clean, corpus_label_list, datasets['train']['size'], datasets['val']['size'], datasets['test']['size']


def orig_open_and_shuffle_data(dataset):

    print("\n Opening and shuffling data...")

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

    # Create a list of labels
    corpus_label_list = []
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label = temp[2]
        corpus_label_list.append(label)

    # Split 10% of the training data off to be the eval set
    train_size = len(splits['train']['ids'])
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    test_size = len(splits['test']['ids'])

    return shuffle_doc_words_list, corpus_label_list, real_train_size, val_size, test_size


def build_vocab(shuffle_doc_words_list):

    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)

    vocab = list(word_set)

    # Dictionary mapping words to unique ids
    word_id_map = {}
    for i in range(len(vocab)):
        word_id_map[vocab[i]] = i

    return vocab, word_id_map


def create_label_matrix(corpus_label_list, dataset, vocab, real_train_size, val_size, test_size):

    print("\n Creating label matrix...")

    # Create list of unique labels
    label_list = list(set(corpus_label_list))

    # Create a sparse matrix of labels
    labels = []
    for i in range(real_train_size + val_size):
        label = corpus_label_list[i]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        labels.append(one_hot)

    # Add vocab
    for i in range(len(vocab)):
        one_hot = [0 for l in range(len(label_list))]
        labels.append(one_hot)

    for i in range(test_size):
        label = corpus_label_list[i + real_train_size + val_size]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        labels.append(one_hot)

    labels = np.array(labels)

    print("Label matrix size:", labels.shape)

    f = open(f"data/ind.{dataset}.labels", 'wb')
    pkl.dump(labels, f)
    f.close()

    return len(label_list)


def reformat_data(dataset):

    # Check dataset
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    if dataset in datasets:
        shuffle_doc_words_list, corpus_label_list, real_train_size, val_size, test_size = orig_open_and_shuffle_data(dataset)
    else:
        shuffle_doc_words_list, corpus_label_list, real_train_size, val_size, test_size = custom_open_data(dataset)

    vocab, word_id_map = build_vocab(shuffle_doc_words_list)

    nb_labels = create_label_matrix(corpus_label_list, dataset, vocab, real_train_size, val_size, test_size)

    # Dictionary of counts of useful things
    count = {
        'train nodes': real_train_size,
        'val nodes': val_size,
        'test nodes': test_size,
        'total nodes': real_train_size + val_size + test_size + len(vocab),
        'word nodes': len(vocab),
        'classes': nb_labels
    }

    with open('data/' + dataset + '.count.json', 'w') as f:
        json.dump(count, f, indent=4)

    print('Data: {}'.format(str(count)))

    return shuffle_doc_words_list, vocab, word_id_map, count


def calc_pmi(doc_list, window_size, count, row=[], col=[], weight=[]):

    print("\n Creating word-word edges...")

    # Create list of sliding windows, moving by one word each time
    windows = []
    for doc_words in doc_list:
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
    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        word_count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * word_count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(count['train nodes'] + i)
        col.append(count['train nodes'] + j)
        weight.append(pmi)

    return row, col, weight


def calc_tfidf(doc_words_list, word_id_map, vocab, count, row, col, weight):

    print("\n Creating document-word edges...")

    # dict of the number of times each word is used in entire corpus
    word_freq = {}
    for doc_words in doc_words_list:
        words = doc_words.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # dictionary of words to list of ids of all texts that use that word
    word_doc_list = {}
    for i in range(len(doc_words_list)):
        doc_words = doc_words_list[i]
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
    for doc_id in range(len(doc_words_list)):
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
    for i in range(len(doc_words_list)):
        doc_words = doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < count['train nodes']:
                row.append(i)
            else:
                row.append(i + count['word nodes'])
            col.append(count['train nodes'] + j)
            idf = log(1.0 * len(doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    return row, col, weight


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def create_edges(shuffle_doc_words_list, vocab, word_id_map, count, window_size, dataset):

    row = []
    col = []
    weight = []

    # Calculate PMI, for word-word edges
    row, col, weight = calc_pmi(shuffle_doc_words_list, window_size, count, row, col, weight)

    # Calculate TF-IDF, for document-word edges
    row, col, weight = calc_tfidf(shuffle_doc_words_list, word_id_map, vocab, count, row, col, weight)

    # Combine into sparse matrix
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(count['total nodes'], count['total nodes']))

    # Make symmetric across main diagonal, by taking largest value
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalise
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    print("Weighted edge matrix size:", adj.shape)

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj_norm, f)
    f.close()


if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("Use: python build_graph.py <dataset>")

    dataset_name = sys.argv[1]

    shuffle_doc_words_list, vocab, word_id_map, count = reformat_data(dataset=dataset_name)

    create_edges(shuffle_doc_words_list, vocab, word_id_map, count, window_size=20, dataset=dataset_name)
