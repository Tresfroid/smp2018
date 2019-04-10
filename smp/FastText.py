import copy
import logging
from itertools import chain

import ipdb
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.autograd import Variable
from utils import *

class FastText(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size,
                 linear_hidden_size,
                 num_label,
                 pretrained_embedding):
        super(FastText, self).__init__()
        self.model_name = 'FastText'
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.linear_hidden_size = linear_hidden_size
        self.num_label = num_label

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.seq = nn.Sequential(
            nn.Linear(emb_size, emb_size*2),
            nn.BatchNorm1d(emb_size*2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_size*2, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, num_label)
        )

        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(
                pretrained_embedding)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded_size = embedded.size()

        output = self.seq(embedded.contiguous().view(-1, 300)).view(embedded_size[0], embedded_size[1], -1)
        classifier_input = torch.mean(output, dim=1)
        output = self.classifier(classifier_input)

        return output

def evaluate(model, loss_function, batch_generator, cuda=None):
    model.train(False)
    total_loss = 0
    total_hit = 0
    total_sample = 0
    batch_i = 0
    no_hit_index_list = []
    true_label_list = []
    predicted_label_list = []
    original_index_list = []
    for batch in batch_generator:
        data, target, original_index = batch[0], batch[1], batch[
            2]
        if cuda is None:
            data_var = Variable(torch.LongTensor(data))
            target_var = Variable(torch.LongTensor(target))
        else:
            data_var = Variable(torch.LongTensor(data).cuda(cuda))
            target_var = Variable(torch.LongTensor(target).cuda(cuda))
        predicted_target = model(data_var)
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
        original_index_list.append(original_index)
    not_hit_index_array = np.concatenate(no_hit_index_list)
    true_label_array = np.asarray(
        list(chain(*true_label_list)), dtype=np.int32)
    predicted_label_array = np.asarray(
        list(chain(*predicted_label_list)), dtype=np.int32)
    original_index_array = np.asarray(
        list(chain(*original_index_list)), dtype=np.int32
    )

    acc = float(total_hit) / float(total_sample)
    returned_document_list = [
        batch_generator.sample_document(index) for index in original_index
    ]
    return total_loss / (
            batch_i + 1
    ), acc, not_hit_index_array, true_label_array, predicted_label_array, returned_document_list, original_index_array

def train_model(model,
                optimizer,
                loss_function,
                num_epoch,
                train_batch_generator,
                dev_batch_generator,
                vocab,
                best_dev_acc=0.,
                cuda=None):
    logging.info("Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]: key for key in vocab.keys()}
    best_model_loss = 1e7
    temp_batch_index = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        model.train(True)
        for train_batch in train_batch_generator:
            temp_batch_index += 1
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
            predicted_train_target = model(train_data_var)
            optimizer.zero_grad()
            loss = loss_function(predicted_train_target, train_target_var)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            optimizer.step()
            temp_batch_index += 1
            if temp_batch_index % 1000 == 0:
                # train_loss,train_acc = evaluate(model,loss_function,train_batch_generator,cuda=cuda)
                train_loss, train_acc = 0, 0
                dev_loss, dev_acc, wrong_index, true_label_array, predicted_label_array, document_list, original_index = evaluate(
                    model, loss_function, dev_batch_generator, cuda)
                logging.info(
                    "\nBatch :{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ndev_loss:{3:0.6f}\tdev_acc:{4:0.6f}".
                        format(temp_batch_index, train_loss, train_acc, dev_loss,
                               dev_acc))
                print(
                    classification_report(
                        true_label_array,
                        predicted_label_array,
                        target_names=['人类作者', '机器作者', '机器翻译', '自动摘要'],
                        digits=4))
                reverse_vocab = {vocab[key]: key for key in vocab.keys()}
                logging.info("True : {} \t Predicted : {}".format(
                    true_label_array[0], predicted_label_array[0]))
                logging.info("\n" + " ".join([reverse_vocab[word] for word in document_list[0]]))
                #                 error_analysis(test_batch_generator, wrong_index,
                #                                predicted_label_array, true_label_array)
                if dev_acc > best_dev_acc:
                    torch.save(model.state_dict(), 'models/FastText_params.pkl')
                    best_dev_acc = dev_acc
                    logging.info('\n' + 'model saved\n')

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
    train_data_file = 'data/train_set.txt'
    dev_data_file = 'data/dev_set.txt'
    vocab = get_vocabulary(train_data_file, vocabsize=20000)
    pretrained_embedding = load_embedding(
        vocab, 'models/train_set_word2vec300.txt', embedding_size=300)
    trainset_data, trainset_label = load(
        train_data_file, vocab, max_value=500)
    dev_set_data, dev_set_label = load(
        dev_data_file, vocab, max_value=500)
    train_batch = DataIter(trainset_data, trainset_label, 50, 0)
    dev_batch = DataIter(dev_set_data, dev_set_label, 50, 0)
    model = FastText(
        vocab_size=20000,
        emb_size=300,
        linear_hidden_size=2000,
        num_label=4,
        pretrained_embedding=pretrained_embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    train_model(model, optimizer, loss_function, 50, train_batch, dev_batch, vocab, cuda=2)

def test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')
    train_data_file = 'data/train_set.txt'
    test_data_file = 'data/test_set.txt'
    vocab = get_vocabulary(train_data_file, vocabsize=20000)
    pretrained_embedding = load_embedding(
        vocab, 'models/train_set_word2vec300.txt', embedding_size=300)
    test_set_data, test_set_label, random_index = load(
        test_data_file, vocab, max_value=500)
    test_batch = TestDataIter(test_set_data, test_set_label, 50, 0)
    model = FastText(
        vocab_size=20000,
        emb_size=300,
        linear_hidden_size=2000,
        num_label=4,
        pretrained_embedding=pretrained_embedding)
    model.cuda(0)
    model.load_state_dict(torch.load('models/FastText_params.pkl'))
    loss_function = nn.CrossEntropyLoss()
    test_loss, test_acc, wrong_index, true_label_array, predicted_label_array, attention_weight, original_index = evaluate(
        model, loss_function, test_batch, cuda=0)
    print(
        classification_report(
            true_label_array,
            predicted_label_array,
            target_names=['人类作者', '机器作者', '机器翻译', '自动摘要'],
            digits=4))

if __name__ == '__main__':
    test()