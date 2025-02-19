import argparse
import shutil

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn.functional as F
from torch.optim import lr_scheduler

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Precision, Recall, Fbeta, Loss

from utils import *
from model import BertClassifier


def f1(prec, rec):
    if prec + rec > 0:
        return (prec * rec * 2)/(prec + rec)
    else:
        return 0

def load_parameters():

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=60)
    parser.add_argument('--bert_lr', type=float, default=1e-4)
    parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'edit'])
    parser.add_argument('--bert_init', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
    parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[dataset] if not specified')

    args = parser.parse_args()

    max_length = args.max_length
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    bert_lr = args.bert_lr
    dataset = args.dataset
    bert_init = args.bert_init

    # Save directory
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
    else:
        ckpt_dir = checkpoint_dir

    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(os.path.basename(__file__), ckpt_dir)

    return max_length, batch_size, nb_epochs, bert_lr, dataset, bert_init, ckpt_dir, args


def tokenize_data(text, count, label_dict, model, max_length):

    print("\n Tokenizing data ...")

    # Tokenize documents
    def encode_input(text, tokenizer):
        input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
        return input.input_ids, input.attention_mask

    input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

    # create train/test/val datasets and dataloaders
    input_ids, attention_mask = {}, {}

    input_ids['train'], input_ids['val'], input_ids['test'] = \
        input_ids_[:count['train nodes']], \
        input_ids_[count['train nodes']:count['train nodes']+count['val nodes']], \
        input_ids_[-count['test nodes']:]
    attention_mask['train'], attention_mask['val'], attention_mask['test'] = \
        attention_mask_[:count['train nodes']], \
        attention_mask_[count['train nodes']:count['train nodes']+count['val nodes']], \
        attention_mask_[-count['test nodes']:]

    datasets = {}
    loader = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = Data.TensorDataset(input_ids[split], attention_mask[split], label_dict[split])
        loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

    return loader


def train_step(engine, batch):

    global model, optimizer

    model.train()

    model = model.to(gpu)

    optimizer.zero_grad()

    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]

    optimizer.zero_grad()

    y_pred = model(input_ids, attention_mask)

    if model.nb_class == 1:
        y_true = label.type(th.float32)
        y_pred = th.sigmoid(y_pred)
        y_pred = th.squeeze(y_pred)

        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        y_true = label.type(th.long)
        loss = F.cross_entropy(y_pred, y_true)

    loss.backward()

    optimizer.step()

    train_loss = loss.item()

    with th.no_grad():
        y_true = y_true.detach().cpu()

        if model.nb_class == 1:
            y_pred = (y_pred > 0.5).int()
            y_pred = y_pred.detach().cpu()
        else:
            y_pred = y_pred.argmax(axis=1).detach().cpu()

        train_acc = accuracy_score(y_true, y_pred)
        train_prec = precision_score(y_true, y_pred, zero_division=0)
        train_rec = recall_score(y_true, y_pred, zero_division=0)
        train_f1 = f1_score(y_true, y_pred, zero_division=0)

    return train_loss, train_acc, train_prec, train_rec, train_f1


def test_step(engine, batch):

    global model

    with th.no_grad():

        model.eval()
        model = model.to(gpu)

        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]

        optimizer.zero_grad()

        y_pred = model(input_ids, attention_mask)

        if model.nb_class == 1:
            y_pred = th.sigmoid(y_pred)
            y_pred = th.squeeze(y_pred)
            y_pred = (y_pred > 0.5).type(th.float32)
            y_true = label.type(th.float32)

        else:
            y_true = label

        return y_pred, y_true


def train(data_loader, model, bert_lr, ckpt_dir, nb_epochs, nb_class):

    print("\n Training ...")

    # Training
    trainer = Engine(train_step)

    evaluator = Engine(test_step)

    if nb_class == 1:
        metrics = {
            'acc': Accuracy(),
            'prec': Precision(average=True),
            'rec': Recall(average=True),
            'f1': Fbeta(beta=1),
            'nll': Loss(th.nn.BCELoss())
        }
    else:
        metrics = {
            'acc': Accuracy(),
            'prec': Precision(average=True),
            'rec': Recall(average=True),
            'f1': Fbeta(beta=1),
            'nll': Loss(th.nn.CrossEntropyLoss())
        }

    for n, f in metrics.items():
        f.attach(evaluator, n)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):

        evaluator.run(data_loader['train'])
        metrics = evaluator.state.metrics
        train_acc, train_prec, train_rec, train_f1, train_nll = metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["nll"]

        evaluator.run(data_loader['val'])
        metrics = evaluator.state.metrics
        val_acc, val_prec, val_rec, val_f1, val_nll = metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["nll"]

        evaluator.run(data_loader['test'])
        metrics = evaluator.state.metrics
        test_acc, test_prec, test_rec, test_f1, test_nll = metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["nll"]

        logger.info("\rEpoch: {}".format(trainer.state.epoch))
        logger.info(" TRAIN acc: {:.3f} prec: {:.3f} rec: {:.3f} f1:{:.3f} loss: {:.3f} "
                    "VAL acc: {:.3f} prec: {:.3f} rec: {:.3f} f1:{:.3f} loss: {:.3f} "
                    "TEST acc: {:.3f} prec: {:.3f} rec: {:.3f} f1:{:.3f} loss: {:.3f}"
                    .format(train_acc, train_prec, train_rec, train_f1, train_nll,
                            val_acc, val_prec, val_rec, val_f1, val_nll,
                            test_acc, test_prec, test_rec, test_f1, test_nll))

        if val_f1 > log_training_results.best_val_f1:
            logger.info("New checkpoint")
            th.save(
                {
                    'bert_model': model.bert_model.state_dict(),
                    'classifier': model.classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': trainer.state.epoch,
                },
                os.path.join(
                    ckpt_dir, 'checkpoint.pth'
                )
            )
            log_training_results.best_val_f1 = val_f1
        scheduler.step()

    log_training_results.best_val_f1 = 0

    trainer.run(data_loader['train'], max_epochs=nb_epochs)


if __name__ == '__main__':

    max_length, batch_size, nb_epochs, bert_lr, dataset, bert_init, ckpt_dir, args = load_parameters()

    logger, cpu, gpu = set_up_logging(ckpt_dir, args)

    _, _, _, _, _, _, _, text, count_dict, label_dict = load_corpus(dataset)

    # # if count_dict['classes'] == 2:
    # #     print("Binary classification")
    # #     nb_class = 1
    # else:
    print("Multiclass classification")
    nb_class = count_dict['classes']

    model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)

    optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    data_loader = tokenize_data(text, count_dict, label_dict, model, max_length)

    train(data_loader, model, bert_lr, ckpt_dir, nb_epochs, nb_class)
