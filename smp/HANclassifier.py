
import copy
from collections import Counter
import torch
from torch import nn
from torch.autograd import Variable
from itertools import chain
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import logging
labels = {'人类作者': 0, '机器作者': 1, '机器翻译': 2, '自动摘要': 3}


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
                   initial_vocab={'<unk>': 0, '<sssss>': 1},
                   vocabsize=0):
    """ acquire vocabulary from text corpus
        Args:
            textfile (str): filename of a dialog corpus
            initial_vocab (dict): initial word-id mapping
            vocabsize (int): upper bound of vocabulary size (0 means no limitation)
        Return:
            dict of word-id mapping
    """
    vocab = copy.copy(initial_vocab)
    word_count = Counter()
    for line in open(textfile, 'r', encoding='utf-8').readlines():
        _, label, text, _ = line.strip().split('\t\t')
        for w in text.split():
            word_count[w] += 1

    # if vocabulary size is specified, most common words are selected
    if vocabsize > 0:
        for w in word_count.most_common(vocabsize):
            if w[0] not in vocab:
                vocab[w[0]] = len(vocab)
                if len(vocab) >= vocabsize:
                    break
    else:  # all observed words are stored
        for w in word_count:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def load_embedding(word_id_dict,
                   embedding_file_name='models/trainset_word2vec300.txt',
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


def load(textfile, vocab, max_value, max_utterance):
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
        random_index = np.random.permutation(line_len)
        line_list = [line_list[index] for index in random_index]

        word_list_buffer = []
        for line in tqdm(line_list):
            _, label, text, _ = line.strip().split('\t\t')
            sent_list = text.strip().split('<sssss>')
            sent_list = [sent.strip().split(' ') for sent in sent_list]
            sent_id_list = [
                convert_words2ids(
                    sent, vocab, max_value=max_value, unk=vocab['<unk>'])
                for sent in sent_list
            ]
            sent_id_list = list(filter(filter_key, sent_id_list))
            new_sent_id_list = []
            previous_sent = []
            for sent in sent_id_list:
                if len(previous_sent) != 0:
                    new_sent = previous_sent + sent
                else:
                    new_sent = sent
                if len(new_sent) < 3:
                    previous_sent = new_sent
                else:
                    new_sent_id_list.append(new_sent)
                    previous_sent = []
            if len(previous_sent) > 0:
                new_sent_id_list.append(previous_sent)
            if len(new_sent_id_list) > 0:
                document_list.append(new_sent_id_list[:max_utterance])
                label_list.append(labels[label])

    def sort_key(document_with_label):
        document = document_with_label[0]
        first_key = len(
            document)  # The first key is the number of utterance of input
        second_key = np.max([
            len(utterance) for utterance in document
        ])  # The second key is the max number of word in input
        third_key = np.mean([
            len(utterance) for utterance in document
        ])  # The third key is the mean number of word in input
        return first_key, second_key, third_key

    document_with_label_list = list(zip(*[document_list, label_list]))
    document_with_label_list = sorted(document_with_label_list, key=sort_key)
    document_list, label_list = list(zip(*document_with_label_list))
    return document_list, label_list


class DataIter(object):
    def __init__(self, document_list, label_list, batch_size, padded_value):
        self.document_list = document_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.padded_value = padded_value
        self.batch_starting_point_list = self._batch_starting_point_list()

    def _batch_starting_point_list(self):
        num_turn_list = [len(document) for document in self.document_list]
        batch_starting_list = []
        previous_turn_index = -1
        previous_num_turn = -1
        for index, num_turn in enumerate(num_turn_list):
            if num_turn != previous_num_turn:
                if index != 0:
                    assert num_turn == previous_num_turn + 1
                    num_batch = (
                                        index - previous_turn_index) // self.batch_size
                    for i in range(num_batch):
                        batch_starting_list.append(
                            previous_turn_index + i * self.batch_size)
                previous_turn_index = index
                previous_num_turn = num_turn
        if previous_num_turn != len(self.document_list):
            num_batch = (index - previous_turn_index) // self.batch_size
            for i in range(num_batch):
                batch_starting_list.append(
                    previous_turn_index + i * self.batch_size)
        return batch_starting_list

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
        transeposed_batch = map(list, zip(*raw_batch))
        padded_batch = []
        length_batch = []
        for transeposed_doc in transeposed_batch:
            length_list = [len(sent) for sent in transeposed_doc]
            max_length = max(length_list)
            new_doc = [
                sent + [self.padded_value] * (max_length - len(sent))
                for sent in transeposed_doc
            ]
            padded_batch.append(
                np.asarray(new_doc, dtype=np.int32).transpose(1, 0))
            length_batch.append(length_list)
        padded_length = np.asarray(length_batch)
        padded_label = np.asarray(label_batch, dtype=np.int32)
        original_index = np.arange(batch_starting, batch_end)
        self.batch_index += 1
        return padded_batch, padded_label, padded_batch, original_index


class HierachicalClassifier(nn.Module):
    def __init__(self,
                 num_word,
                 emb_size,
                 word_rnn_size,
                 word_rnn_num_layer,
                 word_rnn_dropout,
                 word_rnn_bidirectional,
                 word_attention_size,
                 context_rnn_size,
                 context_rnn_num_layer,
                 context_rnn_dropout,
                 context_rnn_bidirectional,
                 context_attention_size,
                 mlp_size,
                 num_label,
                 pretrained_embedding=None):
        self.emb_size = emb_size
        self.word_rnn_size = word_rnn_size
        self.word_rnn_num_layer = word_rnn_num_layer
        self.word_rnn_bidirectional = word_rnn_bidirectional
        self.context_rnn_size = context_rnn_size
        self.context_rnn_num_layer = context_rnn_num_layer
        self.context_rnn_bidirectional = context_rnn_bidirectional
        self.num_label = num_label
        super(HierachicalClassifier, self).__init__()
        self.embedding = nn.Embedding(num_word, emb_size)
        self.word_rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=word_rnn_size,
            dropout=word_rnn_dropout,
            num_layers=word_rnn_num_layer,
            bidirectional=word_rnn_bidirectional)
        word_rnn_output_size = word_rnn_size * 2 if word_rnn_bidirectional else word_rnn_size
        self.word_conv_attention_layer = nn.Conv1d(
            word_rnn_output_size, word_attention_size, 3, padding=2, stride=1)
        self.word_conv_attention_linear = nn.Linear(
            word_attention_size, 1, bias=False)
        self.context_rnn = nn.GRU(
            input_size=word_rnn_output_size,
            hidden_size=context_rnn_size,
            dropout=context_rnn_dropout,
            num_layers=context_rnn_num_layer,
            bidirectional=context_rnn_bidirectional)
        context_rnn_output_size = context_rnn_size * 2 if context_rnn_bidirectional else context_rnn_size
        self.context_conv_attention_layer = nn.Conv1d(
            context_rnn_output_size,
            context_attention_size,
            kernel_size=1,
            stride=1)
        self.context_conv_attention_linear = nn.Linear(
            context_attention_size, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(context_rnn_output_size, mlp_size), nn.LeakyReLU(),
            nn.Linear(mlp_size, num_label), nn.Tanh())
        if pretrained_embedding is not None:
            self.embedding.weight.data = self.embedding.weight.data.new(
                pretrained_embedding)

    def init_rnn_hidden(self, batch_size, level):
        param_data = next(self.parameters()).data
        if level == 'word':
            bidirectional_multipier = 2 if self.word_rnn_bidirectional else 1
            layer_size = self.word_rnn_num_layer * bidirectional_multipier
            word_rnn_init_hidden = Variable(
                param_data.new(layer_size, batch_size,
                               self.word_rnn_size).zero_())
            return word_rnn_init_hidden
        elif level == 'context':
            bidirectional_multipier = 2 if self.context_rnn_bidirectional else 1
            layer_size = self.context_rnn_num_layer * bidirectional_multipier
            context_rnn_init_hidden = Variable(
                param_data.new(layer_size, batch_size,
                               self.context_rnn_size).zero_())
            return context_rnn_init_hidden
        else:
            raise Exception("level must be 'word' or 'context'")

    def forward(self, input_list, length_list):
        """ 
        Arguments: 
        input_list (list) : list of quote utterances, the item is Variable of FloatTensor (word_length * batch_size)
                                 the length of list is number of utterance
        length_list (list): list of length utterances
        Returns:
        word_rnn_output (Variable of FloatTensor): (word_length_of_last_utterance * batch_size)
        context_rnn_ouput (Variable of FloatTensor): (num_utterance * batch_size)
        """
        num_utterance = len(input_list)
        _, batch_size = input_list[0].size()
        # word-level rnn
        word_rnn_hidden = self.init_rnn_hidden(batch_size, level='word')
        word_rnn_output_list = []
        for utterance_index in range(num_utterance):
            word_rnn_input = self.embedding(input_list[utterance_index])
            word_rnn_output, word_rnn_hidden = self.word_rnn(
                word_rnn_input, word_rnn_hidden)
            word_attention_weight = self.word_conv_attention_layer(
                word_rnn_output.permute(1, 2, 0))
            word_attention_weight = word_attention_weight[:, :, 1:-1]
            word_attention_weight = word_attention_weight.permute(2, 0, 1)
            word_attention_weight = self.word_conv_attention_linear(
                word_attention_weight)
            word_attention_weight = nn.functional.relu(word_attention_weight)
            word_attention_weight = nn.functional.softmax(
                word_attention_weight, dim=0)
            word_rnn_last_output = torch.mul(word_rnn_output,
                                             word_attention_weight).sum(dim=0)
            word_rnn_output_list.append(word_rnn_last_output)
            word_rnn_hidden = word_rnn_hidden.detach()
        # context-level rnn
        context_rnn_hidden = self.init_rnn_hidden(batch_size, level='context')
        context_rnn_input = torch.stack(word_rnn_output_list, dim=0)
        context_rnn_output, context_rnn_hidden = self.context_rnn(
            context_rnn_input, context_rnn_hidden)
        context_attention_weight = self.context_conv_attention_layer(
            context_rnn_output.permute(1, 2, 0))
        context_attention_weight = context_attention_weight.permute(2, 0, 1)
        context_attention_weight = nn.functional.relu(context_attention_weight)
        context_attention_weight = self.context_conv_attention_linear(
            context_attention_weight)
        context_attention_weight = nn.functional.relu(context_attention_weight)
        context_attention_weight = nn.functional.softmax(
            context_attention_weight, dim=0)
        context_rnn_last_output = torch.mul(
            context_rnn_output, context_attention_weight).sum(dim=0)
        classifier_input = context_rnn_last_output
        logit = self.classifier(classifier_input)
        attention_weight_array = np.array(context_attention_weight.data.cpu()
                                          .squeeze(-1).tolist()).transpose(
            1, 0)
        context_rnn_last_output_array = np.array(context_rnn_last_output.data.cpu().squeeze(-1).tolist()).transpose(1, 0)
        return logit, attention_weight_array, context_rnn_last_output_array


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
        data, target, length, original_index = batch[0], batch[1], batch[
            2], batch[3]
        if cuda is None:
            data_var_list = [
                Variable(torch.LongTensor(chunk)) for chunk in data
            ]
            target_var = Variable(torch.LongTensor(target))
            length_var = Variable(torch.LongTensor(length))
        else:
            data_var_list = [
                Variable(torch.LongTensor(chunk).cuda(cuda)) for chunk in data
            ]
            target_var = Variable(torch.LongTensor(target).cuda(cuda))
            length_var = Variable(torch.LongTensor(length))
        predicted_target, attention_weight, output_array = model(data_var_list, length)
        loss = loss_function(predicted_target, target_var)
        _, predicted_label = torch.max(predicted_target, dim=1)
        no_hit_mask = predicted_label.data != target_var.data
        no_hit_mask = no_hit_mask.cpu()
        no_hit_index = torch.masked_select(
            torch.arange(predicted_label.data.size(0)), no_hit_mask).tolist()
        no_hit_index = np.asarray(no_hit_index, dtype=np.int32)
        total_hit += torch.sum(predicted_label.data == target_var.data)
        total_loss += loss.data[0]
        total_sample += data[0].shape[1]
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
    ), acc, not_hit_index_array, true_label_array, predicted_label_array, attention_weight, returned_document_list, output_array


def train_model(model,
                optimizer,
                loss_function,
                num_epoch,
                train_batch_generator,
                test_batch_generator,
                vocab,
                cuda=None):
    logging.info("Start Tranining")
    if cuda != None:
        model.cuda(cuda)
    reverse_vocab = {vocab[key]: key for key in vocab.keys()}
    best_model_loss = 1e7
    temp_batch_index = 0
    best_test_acc = 0
    for epoch_i in range(num_epoch):
        logging.info("Epoch {}".format(epoch_i))
        model.train(True)
        for train_batch in train_batch_generator:
            temp_batch_index += 1
            train_data, train_target, length_data = train_batch[
                                                        0], train_batch[1], train_batch[2]

            #             train_data_var_list = [Variable(torch.LongTensor(chunk).cuda(cuda)) for chunk in train_data]
            #             train_target_var = Variable(torch.LongTensor(train_target).cuda(cuda))
            #             length_var = Variable(torch.LongTensor(length_data))

            if cuda is None:
                train_data_var_list = [
                    Variable(torch.LongTensor(chunk)) for chunk in train_data
                ]
                train_target_var = Variable(torch.LongTensor(train_target))
                length_var = Variable(torch.LongTensor(length_data))
            else:
                train_data_var_list = [
                    Variable(torch.LongTensor(chunk).cuda(cuda))
                    for chunk in train_data
                ]
                train_target_var = Variable(
                    torch.LongTensor(train_target).cuda(cuda))
                length_var = Variable(torch.LongTensor(length_data))
            predicted_train_target, _ = model(train_data_var_list, length_var)
            optimizer.zero_grad()
            loss = loss_function(predicted_train_target, train_target_var)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            optimizer.step()
            temp_batch_index += 1
            if temp_batch_index % 1000 == 0:
                # train_loss,train_acc = evaluate(model,loss_function,train_batch_generator,cuda=cuda)
                train_loss, train_acc = 0, 0
                test_loss, test_acc, wrong_index, true_label_array, predicted_label_array, attention_weight, document_list = evaluate(
                    model, loss_function, test_batch_generator, cuda)
                logging.info(
                    "\nBatch :{0:8d}\ntrain_loss:{1:0.6f}\ttrain_acc:{2:0.6f}\ntest_loss:{3:0.6f}\ttest_acc:{4:0.6f}".
                        format(temp_batch_index, train_loss, train_acc, test_loss,
                               test_acc))
                print(
                    classification_report(
                        true_label_array,
                        predicted_label_array,
                        target_names=['人类作者', '机器作者', '机器翻译', '自动摘要'],
                        digits=4))
                reverse_vocab = {vocab[key]: key for key in vocab.keys()}
                logging.info("True : {} \t Predicted : {}".format(
                    true_label_array[0], predicted_label_array[0]))
                logging.info(attention_weight[0])
                sent_str_list = []
                for sent in document_list[0]:
                    sent_str_list.append(" ".join(
                        [reverse_vocab[word] for word in sent]))
                logging.info("\n" + "\n".join(sent_str_list))
                #                 error_analysis(test_batch_generator, wrong_index,
                #                                predicted_label_array, true_label_array)
                if test_acc > best_test_acc:
                    torch.save(model.state_dict(),
                               'models/HAN_classifier_params_0.9819.pkl')
                    best_test_acc = test_acc
                    logging.info('\n' + 'model saved\n')


def error_analysis(batch_generator, wrong_index, predicted_label_array,
                   true_label_array):
    wrong_document_list = [
        batch_generator.sample_document(index) for index in wrong_index
    ]
    wrong_length_counter = Counter()
    total_length_counter = batch_generator.length_count()
    for doc in wrong_document_list:
        wrong_length_counter[len(doc)] += 1
    for length in sorted(wrong_length_counter.keys()):
        print(
            'Length : {0} \t ACC: {1:6f} \t total_num : {2:6d} \t wrong_num: {3:6d}'.
                format(length, 1 -
                       wrong_length_counter[length] / total_length_counter[length],
                       total_length_counter[length], wrong_length_counter[length]))
    fusion_array = np.zeros((5, 5))
    assert predicted_label_array.shape == true_label_array.shape
    for predicted_label, true_label in zip(predicted_label_array, true_label_array):
        fusion_array[predicted_label, true_label] += 1
    fusion_array = fusion_array / np.sum(fusion_array, axis=1, keepdims=True)
    print('\t{0:6d}\t\t{1:6d}\t\t{2:6d}\t\t{3:6d}\t\t{4:6d}'.format(1, 2, 3, 4, 5))
    for true_label, row in enumerate(fusion_array):
        print(true_label + 1, end='\t')
        for predicted_label in row:
            print('{0:6f}'.format(predicted_label), end='\t')
        print()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')
    train_data_file = 'data/train_set.txt'
    test_data_file = 'data/test_set.txt'
    vocab = get_vocabulary(train_data_file, vocabsize=20000)
    pretrained_embedding = load_embedding(
        vocab, 'models/trainset_word2vec300.txt', embedding_size=300)
    trainset_data, trainset_label = load(
        train_data_file, vocab, max_value=50, max_utterance=25)
    testset_data, testset_label = load(
        test_data_file, vocab, max_value=50, max_utterance=25)
    train_batch = DataIter(trainset_data, trainset_label, 50, 0)
    test_batch = DataIter(testset_data, testset_label, 50, 0)
    model = HierachicalClassifier(
        num_word=20000,
        emb_size=300,
        word_rnn_size=300,
        word_rnn_num_layer=2,
        word_rnn_dropout=0.3,
        word_rnn_bidirectional=True,
        word_attention_size=150,
        context_rnn_size=150,
        context_rnn_dropout=0.3,
        context_rnn_bidirectional=True,
        context_attention_size=150,
        mlp_size=100,
        num_label=4,
        context_rnn_num_layer=1,
        pretrained_embedding=pretrained_embedding)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    test_loss, test_acc, wrong_index, true_label_array, predicted_label_array, attention_weight, document_list, output_array = evaluate(
        model, loss_function, test_batch, 0)
    print(output_array)
