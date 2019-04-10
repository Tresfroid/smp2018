import copy
import logging
from collections import Counter
from itertools import chain

import ipdb
import numpy as np
import torch
# from sklearn.metrics import classification_report
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

import re
import json
import codecs
from spacy.lang.en import English
from collections import Counter
from progressbar import ProgressBar

def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    return lines

def log(logfile, text, write_to_log=True):
    if write_to_log:
        with codecs.open(logfile, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
            
def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class RCNN(nn.Module):
    def __init__(self,
                 kernel_size,
                 vocab_size,
                 emb_size,
                 hidden_size,
                 content_dim,
                 num_layers,
                 dropout,
                 bidirectional,
                 k_max_pooling,
                 linear_hidden_size,
                 num_label,
                 pretrained_embedding=None):
        super(RCNN, self).__init__()
        self.model_name = 'RCNN'
        self.kernel_size = kernel_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.content_dim = content_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.k_max_pooling = k_max_pooling
        self.linear_hidden_size = linear_hidden_size
        self.num_label = num_label
        self.pretrained_embedding = pretrained_embedding

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.gru = nn.GRU(input_size=emb_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional
                          )

        self.conv_size = hidden_size*2 if self.bidirectional else hidden_size
        self.conv_size += emb_size

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.conv_size,
                out_channels=self.content_dim,
                kernel_size=self.kernel_size
            ),
            nn.BatchNorm1d(self.content_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=content_dim,
                out_channels=content_dim,
                kernel_size=kernel_size
            ),
            nn.BatchNorm1d(content_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(k_max_pooling*content_dim, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size, num_label)
        )

        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(
                pretrained_embedding)

    def init_rnn_hidden(self, batch_size):
        param_data = next(self.parameters()).data
        bidirectional_multipier = 2 if self.bidirectional else 1
        layer_size = self.num_layers * bidirectional_multipier
        context_rnn_init_hidden = Variable(
            param_data.new(layer_size, batch_size,
                           self.hidden_size*2).zero_())
        return context_rnn_init_hidden

    def forward(self, content):
        batch_size = len(content)
        rnn_hidden = self.init_rnn_hidden(batch_size)
        content = self.embedding(content)

        output, hidden = self.gru(content.permute(1, 0, 2))
        output = output.permute(1, 2, 0)
        embedding = content.permute(0, 2, 1)
        output = torch.cat((output, embedding), dim=1)

        conv_out = kmax_pooling(self.conv(output), 2, self.k_max_pooling)
        conv_out = conv_out.view(conv_out.size(0), -1)
        logits = self.classifier((conv_out))
        return logits

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

def train_model(log_file,model,
                optimizer,
                loss_function,
                num_epoch,
                train_batch_generator,
                dev_batch_generator,
                vocab,
                best_dev_acc=0.,
                cuda=None,
                 ):
    logging.info("Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]: key for key in vocab.keys()}
    best_model_loss = 1e7
    temp_batch_index = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        log(log_file, "Epoch %d" % epoch_i)
        log(log_file, "Batch\t train_loss\t train_acc\t test_loss\t test_acc")
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
            if temp_batch_index % 100 == 0:
                # train_loss,train_acc = evaluate(model,loss_function,train_batch_generator,cuda=cuda)
                train_loss, train_acc = 0, 0
                dev_loss, dev_acc, wrong_index, true_label_array, predicted_label_array, document_list = evaluate(
                    model, loss_function, dev_batch_generator, cuda)
                logging.info(
                    "\nBatch :{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ndev_loss:{3:0.6f}\tdev_acc:{4:0.6f}".
                        format(temp_batch_index, train_loss, train_acc, dev_loss,
                               dev_acc))
                log(log_file, "%d\t %0.6f\t %0.6f\t %0.6f\t %0.6f" % (temp_batch_index, train_loss,train_acc,dev_loss,dev_acc ))
#                 print(
#                     classification_report(
#                         true_label_array,
#                         predicted_label_array,
#                         target_names=['人类作者', '机器作者', '机器翻译', '自动摘要'],
#                         digits=5))
#                 reverse_vocab = {vocab[key]: key for key in vocab.keys()}
#                 logging.info("True : {} \t Predicted : {}".format(
#                     true_label_array[0], predicted_label_array[0]))
#                 logging.info("\n" + " ".join([reverse_vocab[word] for word in document_list[0]]))
                #                 error_analysis(test_batch_generator, wrong_index,
                #                                predicted_label_array, true_label_array)
#                 if dev_acc > best_dev_acc:
#                     torch.save(model.state_dict(), 'models/RCNN_params.pkl')
#                     best_dev_acc = dev_acc
#                     logging.info('\n' + 'model saved\n')

def convert_words2ids(words, vocab, unk, max_value, sos=None, eos=None):
    """ convert string sequence into word id sequence
        Args:
            words (list): word sequence
            vocab (dict): word-id mapping
            unk (int): id of unkown word "<unk>"
            sos (int): id of start-of-sentence symbol "<sos>"
            eos (int): id of end-of-sentence symbol "eos"
        Returns:
            numpy array of word ids sequence
    """
    id_list = [vocab[w] if w in vocab else unk for w in words]
    if sos is not None:
        id_list.insert(0, sos)
    if eos is not None:
        id_list.append(eos)

    return id_list[:max_value]

def get_vocabulary(textfile,
                   initial_vocab={'<unk>':0},
                   vocabsize=0):
    """ acquire vocabulary from text corpus
        Args:
            textfile (str): filename of a dialog corpus
            initial_vocab (dict): initial word-id mapping
            vocabsize (int): upper bound of vocabulary size (0 means no limitation)
        Return:
            dict of word-id mapping
    """
    mallet_stopwords = read_text('/data/lengjia/topic_model/mallet_stopwords.txt')  #得到文件每一行组成的list
    mallet_stopwords = { s.strip() for s in mallet_stopwords}
    parser = English()
    
    word_counts = Counter()
    with codecs.open(textfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()    
    for line in tqdm(lines):
        label,text = line.strip().split("\t")
#         line = json.loads(line)
#         label = line['stars']
#         text = line['text']
        text = re.sub('<[^>]+>', '', text)
        parse = parser(text)
        words = [re.sub('\s', '', token.orth_) for token in parse]
        words = [word.lower() for word in words if len(word) >= 1]
        words = [word for word in words if word not in mallet_stopwords]
        words = [word for word in words if re.match('^[a-zA-A]*$', word) is not None]
        words = [word for word in words if re.match('[a-zA-A0-9]', word) is not None]
        words = ['<NUM>' if re.match('[0-9]', word) is not None else word for word in words]
        word_counts.update(words)

    # if vocabulary size is specified, most common words are selected
    if vocabsize > 0:
        vocab = copy.copy(initial_vocab)
        for w in word_counts.most_common(vocabsize):
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocabsize:
                    break
    else: # all observed words are stored
        for w in word_counts:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab  

def load_embedding(word_id_dict,
                   embedding_file_name='/data/lengjia/topic_model/yelp/yelp_embedding_300.txt',
                   embedding_size=300):
    embedding_length = len(word_id_dict)
    embedding_matrix = np.random.uniform(
        -1e-2, 1e-2, size=(embedding_length, embedding_size))
    embedding_matrix[0] = 0
    hit = 0
    with open(embedding_file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            splited_line = line.strip().split(' ')
            word, embeddings = splited_line[0], splited_line[1:]
            if word in word_id_dict:
                word_index = word_id_dict[word]
                embedding_array = np.fromstring(
                    '\n'.join(embeddings), dtype=np.float32, sep='\n')
                embedding_matrix[word_index] = embedding_array
                hit += 1

    hit_rate = float(hit) / embedding_length
    print(('The hit rate is {}'.format(hit_rate)))
    return embedding_matrix

def load(textfile, vocab, max_value):
    """ Load a text corpus as word Id sequences
        Args:
            textfile (str): filename of a dialog corpus
            vocab (dict): word-id mapping
        Return:
            list of dialogue : dialogue is (input_id_list,output_id_list)
    """
    document_list = []
    label_list = []

    def filter_key(sent):
        unk_count = sent.count(vocab['<unk>'])
        return unk_count / len(sent) < 0.3

    with open(textfile, 'r', encoding='utf-8') as f:
        line_list = f.readlines()
        line_len = len(line_list)
#         random_index = np.random.permutation(line_len)
#         line_list = [line_list[index] for index in random_index]

        word_list_buffer = []
        for line in tqdm(line_list):
            label, text = line.strip().split('\t')
            text_list = text.strip().split(' ')
            text_list = convert_words2ids(text_list, vocab, max_value=max_value, unk=1)

            document_list.append(text_list)
            label_list.append(label)

    document_with_label_list = list(zip(*[document_list, label_list]))
    document_with_label_list = sorted(document_with_label_list, key=len)
    document_list, label_list = list(zip(*document_with_label_list))
    return document_list, label_list

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
    train_data_file = '/data/lengjia/topic_model/yelp_2013_39923/yelp/baseline/yelp-2013-train.txt.ss'
    dev_data_file = '/data/lengjia/topic_model/yelp_2013_39923/yelp/baseline/yelp-2013-test.txt.ss'
    log_file = '/data/lengjia/topic_model/yelp_2013_39923/yelp/baseline/RCNN.log'
    vocab = get_vocabulary(train_data_file, vocabsize=20000)
    pretrained_embedding = load_embedding(
        vocab, "/data/lengjia/topic_model/yelp/yelp_embedding_300.txt", embedding_size=300)
    trainset_data, trainset_label = load(
        train_data_file, vocab, max_value=50)
    dev_set_data, dev_set_label = load(
        dev_data_file, vocab, max_value=50)
    train_batch = DataIter(trainset_data, trainset_label, 50, 0)
    dev_batch = DataIter(dev_set_data, dev_set_label, 50, 0)
    model = RCNN(
        kernel_size=1,
        vocab_size=20000,
        emb_size=300,
        hidden_size=300,
        content_dim=150,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        k_max_pooling=2,
        linear_hidden_size=2000,
        num_label=5,
        pretrained_embedding=pretrained_embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    train_model(log_file,model, optimizer, loss_function, 20, train_batch, dev_batch, vocab, best_dev_acc=0.559336, cuda=0)    # cuda 0

if __name__ == '__main__':
    main()
