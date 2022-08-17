import os
import shutil
import argparse
import logging
import dgl
from sklearn.metrics import accuracy_score

import torch.nn.functional as F
import torch.utils.data as Data
from torch.optim import lr_scheduler

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss

from model import BertGCN, BertGAT
from utils import *


def load_parameters():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--bert_init', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--pretrained_bert_ckpt', default=None)
    parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
    parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
    parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
    parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gcn_lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=1e-5)

    args = parser.parse_args()
    max_length = args.max_length
    batch_size = args.batch_size
    m = args.m
    nb_epochs = args.nb_epochs
    bert_init = args.bert_init
    pretrained_bert_ckpt = args.pretrained_bert_ckpt
    dataset = args.dataset
    checkpoint_dir = args.checkpoint_dir
    gcn_model = args.gcn_model
    gcn_layers = args.gcn_layers
    n_hidden = args.n_hidden
    heads = args.heads
    dropout = args.dropout
    gcn_lr = args.gcn_lr
    bert_lr = args.bert_lr

    if checkpoint_dir is None:
        ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
    else:
        ckpt_dir = checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(os.path.basename(__file__), ckpt_dir)

    return args, max_length, batch_size, m, nb_epochs, bert_init, pretrained_bert_ckpt, dataset, \
           gcn_model, gcn_layers, n_hidden, heads, dropout, gcn_lr, bert_lr, ckpt_dir


def load_data(dataset):

    # Todo: might want to simplify some of this into the load_corpus function

    print("\n Loading dataset ... ")

    adj_norm, y_train, y_val, y_test, train_mask, val_mask, test_mask, text, count = load_corpus(dataset)
    '''
    adj: n*n sparse adjacency matrix
    y_train, y_val, y_test: n*c matrices 
    train_mask, val_mask, test_mask: n-d bool array
    '''

    # transform one-hot label to class ID for pytorch computation
    y = y_train + y_test + y_val
    y_train = y_train.argmax(axis=1)
    y = y.argmax(axis=1)

    # document mask used for update feature
    doc_mask = train_mask + val_mask + test_mask

    # create index loader
    train_idx = Data.TensorDataset(th.arange(0, count['train nodes'], dtype=th.long))
    val_idx = Data.TensorDataset(th.arange(count['train nodes'], count['train nodes'] + count['val nodes'], dtype=th.long))
    test_idx = Data.TensorDataset(th.arange(count['total nodes'] - count['test nodes'], count['total nodes'], dtype=th.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
    idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

    return y, y_train, train_mask, val_mask, test_mask, doc_mask, idx_loader_train, idx_loader_val, idx_loader_test, idx_loader, adj_norm, text, count


def tokenize_data(text, model):

    print("\n Tokenizing data ...")

    def encode_input(text, tokenizer):
        input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        #     print(input.keys())
        return input.input_ids, input.attention_mask

    input_ids, attention_mask = encode_input(text, model.tokenizer)
    input_ids = th.cat(
        [input_ids[:-count['test nodes']],
         th.zeros((count['word nodes'], max_length), dtype=th.long),
         input_ids[-count['test nodes']:]]
    )
    attention_mask = th.cat(
        [attention_mask[:-count['test nodes']],
         th.zeros((count['word nodes'], max_length), dtype=th.long),
         attention_mask[-count['nb_test']:]])

    return input_ids, attention_mask


def build_graph(adj_norm, input_ids, attention_mask, y, train_mask, val_mask, test_mask, y_train, count):

    print("\n Building graph ...")

    # build DGL Graph
    # edges
    g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')

    # nodes
    g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
    g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
        th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
    g.ndata['label_train'] = th.LongTensor(y_train)
    g.ndata['cls_feats'] = th.zeros((count['total nodes'], model.feat_dim))

    # test
    h = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
    h.ndata['input_ids'] = input_ids
    h.ndata['attention_mask'] = attention_mask
    h.ndata['label'] = th.LongTensor(y)
    h.ndata['train'] = th.FloatTensor(train_mask)
    h.ndata['val'] = th.FloatTensor(val_mask)
    h.ndata['test'] = th.FloatTensor(test_mask)
    h.ndata['label_train'] = th.LongTensor(y_train)
    h.ndata['cls_feats'] = th.zeros((count['total nodes'], model.feat_dim))

    assert g == h

    logger.info('graph information:')
    logger.info(str(g))

    return g


if __name__ == '__main__':

    # Set up
    args, max_length, batch_size, m, nb_epochs, bert_init, pretrained_bert_ckpt, dataset, \
        gcn_model, gcn_layers, n_hidden, heads, dropout, gcn_lr, bert_lr, ckpt_dir = load_parameters()

    logger, cpu, gpu = set_up_logging(ckpt_dir, args)

    # Load and format data
    y, y_train, train_mask, val_mask, test_mask, doc_mask, idx_loader_train, idx_loader_val, idx_loader_test, \
        idx_loader, adj_norm, text, count = load_data(dataset)

    # Instantiate model
    if gcn_model == 'gcn':
        model = BertGCN(nb_class=count['classes'], pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                        n_hidden=n_hidden, dropout=dropout)
    else:
        model = BertGAT(nb_class=count['classes'], pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                        heads=heads, n_hidden=n_hidden, dropout=dropout)

    if pretrained_bert_ckpt is not None:
        ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
        model.bert_model.load_state_dict(ckpt['bert_model'])
        model.classifier.load_state_dict(ckpt['classifier'])

    # Tokenize data
    input_ids, attention_mask = tokenize_data(text, model)

    # Build graph
    g = build_graph(adj_norm, input_ids, attention_mask, y, train_mask, val_mask, test_mask, y_train, count)

    # Training
    def update_feature():
        global model, g, doc_mask
        # no gradient needed, uses a large batchsize to speed up the process
        dataloader = Data.DataLoader(
            Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
            batch_size=1024
        )
        with th.no_grad():
            model = model.to(gpu)
            model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(gpu) for x in batch]
                output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
                cls_list.append(output.cpu())
            cls_feat = th.cat(cls_list, axis=0)
        g = g.to(cpu)
        g.ndata['cls_feats'][doc_mask] = cls_feat
        return g


    optimizer = th.optim.Adam([
            {'params': model.bert_model.parameters(), 'lr': bert_lr},
            {'params': model.classifier.parameters(), 'lr': bert_lr},
            {'params': model.gcn.parameters(), 'lr': gcn_lr},
        ], lr=1e-3
    )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


    def train_step(engine, batch):
        global model, g, optimizer
        model.train()
        model = model.to(gpu)
        g = g.to(gpu)
        optimizer.zero_grad()
        (idx, ) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        train_mask = g.ndata['train'][idx].type(th.BoolTensor)
        y_pred = model(g, idx)[train_mask]
        y_true = g.ndata['label_train'][idx][train_mask]
        loss = F.nll_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        g.ndata['cls_feats'].detach_()
        train_loss = loss.item()
        with th.no_grad():
            if train_mask.sum() > 0:
                y_true = y_true.detach().cpu()
                y_pred = y_pred.argmax(axis=1).detach().cpu()
                train_acc = accuracy_score(y_true, y_pred)
            else:
                train_acc = 1
        return train_loss, train_acc


    trainer = Engine(train_step)


    @trainer.on(Events.EPOCH_COMPLETED)
    def reset_graph(trainer):
        scheduler.step()
        update_feature()
        th.cuda.empty_cache()


    def test_step(engine, batch):
        global model, g
        with th.no_grad():
            model.eval()
            model = model.to(gpu)
            g = g.to(gpu)
            (idx, ) = [x.to(gpu) for x in batch]
            y_pred = model(g, idx)
            y_true = g.ndata['label'][idx]
            return y_pred, y_true


    evaluator = Engine(test_step)
    metrics={
        'acc': Accuracy(),
        'nll': Loss(th.nn.NLLLoss())
    }
    for n, f in metrics.items():
        f.attach(evaluator, n)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(idx_loader_train)
        metrics = evaluator.state.metrics
        train_acc, train_nll = metrics["acc"], metrics["nll"]
        evaluator.run(idx_loader_val)
        metrics = evaluator.state.metrics
        val_acc, val_nll = metrics["acc"], metrics["nll"]
        evaluator.run(idx_loader_test)
        metrics = evaluator.state.metrics
        test_acc, test_nll = metrics["acc"], metrics["nll"]
        logger.info(
            "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
            .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
        )
        if val_acc > log_training_results.best_val_acc:
            logger.info("New checkpoint")
            th.save(
                {
                    'bert_model': model.bert_model.state_dict(),
                    'classifier': model.classifier.state_dict(),
                    'gcn': model.gcn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': trainer.state.epoch,
                },
                os.path.join(
                    ckpt_dir, 'checkpoint.pth'
                )
            )
            log_training_results.best_val_acc = val_acc


    log_training_results.best_val_acc = 0
    g = update_feature()
    trainer.run(idx_loader, max_epochs=nb_epochs)
