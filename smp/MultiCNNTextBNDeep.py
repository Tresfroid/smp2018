import copy
import logging
from collections import Counter
from itertools import chain
import random
import ipdb
import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from utils import *
import visdom
from optim.nadam import Nadam

kernel_sizes = [1, 2, 3, 4]

class MultiCNNTextBNDeep(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size,
                 content_dim,
                 num_layers,
                 linear_hidden_size,
                 content_seq_len,
                 num_label,
                 pretrained_embedding=None):
        super(MultiCNNTextBNDeep, self).__init__()
        self.model_name = 'CNN'
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.content_dim = content_dim
        self.num_layers = num_layers
        self.linear_hidden_size = linear_hidden_size
        self.num_label = num_label
        self.pretrained_embedding = pretrained_embedding
        self.content_seq_len = content_seq_len

        self.embedding = nn.Embedding(vocab_size, emb_size)

        # convs = [
        #     nn.Sequential(
        #         nn.Conv1d(in_channels=emb_size,
        #                   out_channels=content_dim,
        #                   kernel_size=kernel_size),
        #         nn.BatchNorm1d(content_dim),
        #         nn.ReLU(inplace=True),
        #
        #         nn.Conv1d(in_channels=content_dim,
        #                   out_channels=content_dim,
        #                   kernel_size=kernel_size),
        #         nn.BatchNorm1d(content_dim),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool1d(kernel_size=(content_seq_len - kernel_size * 2 + 2))
        #     )
        #     for kernel_size in kernel_sizes
        # ]
        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=emb_size,
                          out_channels=content_dim,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(content_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=content_dim,
                          out_channels=content_dim,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(content_dim),
                nn.ReLU(inplace=True),
                #                 nn.MaxPool1d(kernel_size=(100))
            )
            for kernel_size in kernel_sizes
        ]

        self.convs = nn.ModuleList(convs)

        linear_input_size = len(kernel_sizes) * content_dim

        self.classifier = nn.Sequential(
            nn.Linear(linear_input_size, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, num_label)
        )

        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(
                pretrained_embedding)

    def forward(self, content):
        content = self.embedding(content)
        # output = [conv(content.permute(0, 2, 1)) for conv in self.convs]
        i = 1
        output = []
        seq_len = content.size(1)
        for conv in self.convs:
            max_pool1d = nn.MaxPool1d(kernel_size=seq_len - i * 2 + 2)
            output.append(max_pool1d(conv(content.permute(0, 2, 1))))
            i += 1

        output = torch.cat((output), dim=1)
        conv_out = output.view(output.size(0), -1)
        logits = self.classifier((conv_out))
        return logits, conv_out


def evaluate(model, loss_function, batch_generator, cuda=None):
    model.train(False)
    total_loss = 0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    no_hit_index_list = []
    true_label_list = []
    predicted_label_list = []
    for batch in batch_generator:
        data, target, original_index = batch[0], batch[1], batch[
            2]
        if cuda is None:
            data_var = Variable(torch.LongTensor(data))
            target_var = Variable(torch.LongTensor(target))
        else:
            data_var = Variable(torch.LongTensor(data).cuda(cuda))
            target_var = Variable(torch.LongTensor(target).cuda(cuda))
        predicted_target, _ = model(data_var)
        loss = loss_function(predicted_target, target_var)
        _, predicted_label = torch.max(predicted_target, dim=1)
        no_hit_mask = predicted_label.data != target_var.data
        no_hit_mask = no_hit_mask.cpu()
        no_hit_index = torch.masked_select(
            torch.arange(predicted_label.data.size(0)), no_hit_mask).tolist()
        no_hit_index = np.asarray(no_hit_index, dtype=np.int32)
        total_hit += torch.sum(predicted_label.data == target_var.data)
        total_loss += loss.data[0]
        total_sample += data.shape[0]
        batch_i += 1
        no_hit_index_list.append(original_index[no_hit_index])
        true_label_list.append(target)
        predicted_label_array = np.asarray(predicted_label.cpu().data.tolist())
        predicted_label_list.append(predicted_label_array)
    not_hit_index_array = np.concatenate(no_hit_index_list)
    true_label_array = np.asarray(
        list(chain(*true_label_list)), dtype=np.int32)
    predicted_label_array = np.asarray(
        list(chain(*predicted_label_list)), dtype=np.int32)

    acc = float(total_hit) / float(total_sample)
    returned_document_list = [
        batch_generator.sample_document(index) for index in original_index
    ]
    return total_loss / (
            batch_i + 1
    ), acc, not_hit_index_array, true_label_array, predicted_label_array, returned_document_list


def train_model(model,
                optimizer,
                loss_function,
                num_epoch,
                train_batch_generator,
                dev_batch_generator,
                vocab,
                plot_every=1000,
                best_dev_f1=0.,
                lr=0.1,
                lr2=0,
                lr_decay=0.5,
                decay_tol=5,
                optim='sgd',
                model_name='CNN',
                cuda=None):
    logging.info("Start Training")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]: key for key in vocab.keys()}
    best_model_loss = 1e7
    temp_batch_index = 0
    batch_list = []
    loss_list = []
    f1_list = []
    lr_list = []
    vis = visdom.Visdom()
    win = vis.line(
        Y=np.random.rand(10, 3),
        X=np.arange(10),
        opts=dict(
            showlegend=True,
            legend=['loss', 'f1', 'lr'],
            markers=True,
            markersize=5,
            title=model_name
        ),
        name=model_name,
    )

    decay_count = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        model.train(True)
        for train_batch in train_batch_generator:
            train_data, train_target, original_index = train_batch[0], train_batch[1], train_batch[2]
            #             train_data_var_list = [Variable(torch.LongTensor(chunk).cuda(cuda)) for chunk in train_data]
            #             train_target_var = Variable(torch.LongTensor(train_target).cuda(cuda))
            #             length_var = Variable(torch.LongTensor(length_data))
            if cuda is None:
                train_data_var = Variable(torch.LongTensor(train_data))
                train_target_var = Variable(torch.LongTensor(train_target))
            else:
                train_data_var = Variable(torch.LongTensor(train_data).cuda(cuda))
                train_target_var = Variable(
                    torch.LongTensor(train_target).cuda(cuda))
            predicted_train_target, _ = model(train_data_var)
            optimizer.zero_grad()
            loss = loss_function(predicted_train_target, train_target_var)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            optimizer.step()
            temp_batch_index += 1
            if temp_batch_index % plot_every == 0:
                # train_loss,train_acc = evaluate(model,loss_function,train_batch_generator,cuda=cuda)
                train_loss, train_acc = 0, 0
                dev_loss, dev_acc, wrong_index, true_label_array, predicted_label_array, document_list = evaluate(
                    model, loss_function, dev_batch_generator, cuda)
                dev_acc, dev_rec, dev_f1, _ = precision_recall_fscore_support(true_label_array, predicted_label_array,
                                                                              average='macro')
                logging.info(
                    "\nBatch :{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ndev_loss:{3:0.6f}\tdev_f1:{4:0.6f}".
                        format(temp_batch_index, train_loss, train_acc, dev_loss,
                               dev_f1))
                batch_list.append(temp_batch_index)
                loss_list.append(dev_loss)
                f1_list.append(dev_f1)
                lr_list.append(lr)
                vis.line(
                    Y=np.array([np.array(loss_list), np.array(f1_list), np.array(lr_list)]).T,
                    X=np.array(batch_list),
                    win=win,
                    opts=dict(
                        showlegend=True,
                        legend=['loss', 'f1', 'lr'],
                        markers=True,
                        markersize=5,
                        title=model_name
                    ),
                    name=model_name
                )
                print(
                    classification_report(
                        true_label_array,
                        predicted_label_array,
                        target_names=['人类作者', '机器作者', '机器翻译', '自动摘要'],
                        digits=4))
                reverse_vocab = {vocab[key]: key for key in vocab.keys()}
                logging.info("True : {} \t Predicted : {}".format(
                    true_label_array[0], predicted_label_array[0]))
                logging.info("\n" + " ".join([reverse_vocab[word] for word in random.choice(document_list)]))
                #                 error_analysis(test_batch_generator, wrong_index,
                #                                predicted_label_array, true_label_array)
                if dev_f1 > best_dev_f1:
                    save_path = 'best_model_new/' + model_name  + '_%.4ff1'%dev_f1 + '_epoch%d'%epoch_i+ '_params.pkl'
                    torch.save(model.state_dict(), save_path)
                    best_dev_f1 = dev_f1
                    logging.info('\n' + 'model saved\n')
                    decay_count = 0
                else:
                    decay_count += 1

            if decay_count == decay_tol:
                if lr < 1e-5:
                    print('Training End %s'%best_dev_f1)
                    return
                model.load_state_dict(torch.load(save_path))
                lr = lr * lr_decay
                if epoch_i > 2:
                    lr2 = 2e-4
                optimizer, lr = adjust_optimizer(model, lr=lr, lr2=lr2, momentum=.9, optim=optim)
                decay_count = 0


class DataIter(object):
    def __init__(self, document_list, label_list, batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = range(0, len(document_list), batch_size)

    def sample_document(self, index):
        return self.document_list[index]

    def __iter__(self):
        self.current_batch_starting_point_list = copy.copy(
            self.batch_starting_point_list)
        self.current_batch_starting_point_list = np.random.permutation(
            self.current_batch_starting_point_list)
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.current_batch_starting_point_list):
            raise StopIteration
        batch_starting = self.current_batch_starting_point_list[
            self.batch_index]
        batch_end = batch_starting + self.batch_size
        raw_batch = self.document_list[batch_starting:batch_end]
        label_batch = self.label_list[batch_starting:batch_end]
        padded_batch = []
        max_length = max([len(doc) for doc in raw_batch])
        for raw_doc in raw_batch:
            new_doc = raw_doc + [self.padded_value] * (max_length - len(raw_doc))
            padded_batch.append(np.asarray(new_doc, dtype=np.int32))
        padded_batch = np.asarray(padded_batch, dtype=np.int32)
        padded_label = np.asarray(label_batch, dtype=np.int32)
        original_index = np.arange(batch_starting, batch_end)
        self.batch_index += 1
        return padded_batch, padded_label, original_index


class TestDataIter(object):
    def __init__(self, document_list, label_list, batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = range(0, len(document_list), batch_size)

    def sample_document(self, index):
        return self.document_list[index]

    def __iter__(self):
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= len(self.batch_starting_point_list):
            raise StopIteration
        batch_starting = self.batch_starting_point_list[
            self.batch_index]
        if self.batch_index < len(self.batch_starting_point_list) - 1:
            batch_end = self.batch_starting_point_list[self.batch_index + 1]
        else:
            batch_end = len(self.document_list)
        raw_batch = self.document_list[batch_starting:batch_end]
        label_batch = self.label_list[batch_starting:batch_end]
        padded_batch = []
        max_length = max([len(doc) for doc in raw_batch])
        for raw_doc in raw_batch:
            new_doc = raw_doc + [self.padded_value] * (max_length - len(raw_doc))
            padded_batch.append(np.asarray(new_doc, dtype=np.int32))
        padded_batch = np.asarray(padded_batch, dtype=np.int32)
        padded_label = np.asarray(label_batch, dtype=np.int32)
        original_index = np.arange(batch_starting, batch_end)
        self.batch_index += 1
        return padded_batch, padded_label, original_index


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')
    train_data_file = 'data_new/train_set.txt'
    dev_data_file = 'data_new/dev_set.txt'
    vocab = get_vocabulary(train_data_file, vocabsize=20000)
    pretrained_embedding = load_embedding(
        vocab, 'model_new/embedding_word2vec300.txt', embedding_size=300)
    trainset_data, trainset_label, random_index, id_list = load(
        train_data_file, vocab, max_value=300)
    dev_set_data, dev_set_label, random_index, id_list = load(
        dev_data_file, vocab, max_value=300)
    train_batch = DataIter(trainset_data, trainset_label, 50, 0)
    dev_batch = TestDataIter(dev_set_data, dev_set_label, 50, 0)
    model = MultiCNNTextBNDeep(
        vocab_size=20000,
        emb_size=300,
        content_dim=150,
        num_layers=1,
        linear_hidden_size=600,
        content_seq_len=300,
        num_label=4,
        pretrained_embedding=pretrained_embedding)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    train_model(model, optimizer, loss_function, 50, train_batch, dev_batch, vocab, plot_every=100, lr=1e-2,
                lr_decay=0.5, decay_tol=-1, best_dev_f1=0., optim='sgd', model_name='MultiCNNTextBNDeep300_1layer_sgd',
                cuda=3)  # cuda 0


if __name__ == '__main__':
    main()
