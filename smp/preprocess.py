import json
import time
import pandas as pd
import numpy as np
import os
import re
from pyltp import SentenceSplitter
import csv
import random
from tqdm import tqdm
from progressbar import ProgressBar
from pyltp import Segmentor
import jieba
import logging
import copy
from collections import Counter
labels = {'人类作者': 0, '机器作者': 1, '机器翻译': 2, '自动摘要': 3}
idx2labels = ['人类作者', '机器作者', '机器翻译', '自动摘要']


# 读取原始数据training.txt，保存为DataFrame格式
def read_data(file):
    start_time = time.time()
    with open(file, 'r', encoding='utf-8') as f:
        print('Reading...')
        lines = f.readlines()

        ids = []
        labels = []
        contents = []

        for line in lines:
            text = json.loads(line)
            ids.append(text['id'])
            labels.append(text['标签'])
            contents.append(text['内容'])

    data = pd.DataFrame({
        'id': ids,
        'label': labels,
        'content': contents,
    })
    end_time = time.time()
    print('Time: %.2fs' % (end_time - start_time))
    return data


# 读取验证集或者测试集数据(无标签)
def read_train_data(file):
    start_time = time.time()
    with open(file, 'r', encoding='utf-8') as f:
        print('Reading...')
        lines = f.readlines()

        ids = []
        contents = []
        for line in lines:
            text = json.loads(line)
            ids.append(text['id'])
            contents.append(text['内容'])
    data = pd.DataFrame({'id': ids, 'content': contents})
    end_time = time.time()
    print('Time: %.2fs' % (end_time - start_time))
    return data


# 打标签 eng, num, date, time, 特殊字符special, 分段符sssss
def pad_alltags(content):
    content = re.sub(r'<|>', '', content)  # clean
    content = re.sub(r'[a-zA-Z]+', '<eng>', content)
    content = re.sub(r'\d+[年月日号]|(周|星期|礼拜)[一二三四五六七日天]', '<date>', content)
    content = re.sub(
        r'[\d零一二三四五六七八九十百千万亿]+天|\d+[时分秒|点半?钟?]|[1-2]\d{3}\D|\d{2}:\d{2}(:\d{2})?',
        '<time>', content)
    content = re.sub(r'[\d\.]+%?％?[十百千万亿]*|[零一二三四五六七八九十几]+[十百千万亿]+', '<num>',
                     content)
    content = re.sub(
        r'[^\u4e00-\u9fa5a-zA-Z ,\.?!%\*\(\)-_\+=`~#\|\{\}:;\"\'<>\^/\\\[\]，。、？！…·（）￥【】：；“”‘’《》—]',
        '<special>', content)
    content = '<sssss>'.join(SentenceSplitter.split(content))
    return content


# 统计tag信息
def count_tag(contents, tag):
    return [len(re.findall(tag, content)) for content in contents]


# 处理分词后的标签
def process_tags(words):
    pos_words = []
    tags = ['num', 'eng', 'date', 'time', 'special', 'sssss']
    for word in words:
        word = re.sub(r'<|>', '', word)
        if word == '':
            continue
        if word in tags:
            word = '<' + word + '>'
        if word.endswith('。') and len(word) > 1:
            pos_words.extend([word[:-1], '。'])
            continue
        pos_words.append(word)
    return pos_words


# 统计顿号
def count_comma(contents):
    div_re = re.compile(r'[、]+')
    return [len(re.findall(div_re, content)) for content in contents]


# 统计段落数
def count_para(contents):
    return [max(1, len(re.findall(r'<sssss>', content))) for content in contents]


# 统计句数
def count_sent(contents):
    div_re = re.compile(r"[,:;?!。；，：？！]+")  #unicode标点符号，。
    return [max(1, len(re.findall(div_re, content))) for content in contents]


# 统计文本长度(字符)
def get_text_len(contents):
    return [len(content) for content in contents]


# 统计文本长度(词)
def get_text_word_len(contents):
    return [len(re.sub(r'<sssss>', '', content).split(' ')) for content in contents]


# 第一段的长度(字符级别)
def first_para_length(contents):
    return [len(content.split('<sssss>')[0]) for content in contents]


# 第一段的长度(词级别)
def first_para_word_len(contents):
    return [len(content.split('<sssss>')[0].split(' ')) for content in contents]


# 最大段落长度(字符级别)
def max_para_length(contents):
    return [max([len(sent) for sent in content.split('<sssss>')]) for content in contents]


# 最小段落长度(字符级别)
def min_para_length(contents):
    return [min([len(sent) for sent in content.split('<sssss>') if sent.strip(' ') != '']) for content in contents]


# 最大段落长度(词级别)
def max_para_word_len(contents):
    return [max([len(sent.split(' ')) for sent in content.split('<sssss>')]) for content in contents]


# 最小段落长度(词级别)
def min_para_word_len(contents):
    return [min([len(sent.split(' ')) for sent in content.split('<sssss>')]) for content in contents]


# 第一句长度(字符级别)
def first_sent_length(contents):
    punc = r"[,\.:;?!。；，：？！…]+"  #unicode标点符号，。
    text = [re.sub(punc, '<sent>', content) for content in contents]
    return [len(t.split('<sent>')[0]) for t in text]


# 最大句长度(字符级别)
def max_sent_length(contents):
    punc = r"[,\.:;?!。；，：？！…]+"  #unicode标点符号，。
    text = [re.sub(punc, '<sent>', content) for content in contents]
    return [max([len(s) for s in t.split('<sent>')]) for t in text]


# 最大句长度(词级别)
def max_sent_word_len(contents):
    punc = r"[,\.:;?!。；，：？！…]+"  #unicode标点符号，。
    text = [re.sub(punc, '<sent>', content) for content in contents]
    return [max([len(s.split(' ')) for s in t.split('<sent>')]) for t in text]


# 最小句长度(字符级别)
def min_sent_length(contents):
    punc = r"[,\.:;?!。；，：？！…]+"  #unicode标点符号，。
    text = [re.sub(punc, '<sent>', content) for content in contents]
    return [min([len(s) for s in t.split('<sent>') if s.strip(' ') != '']) for t in text]


# 最小句长度(词级别)
def min_sent_word_len(contents):
    punc = r"[,\.:;?!。；，：？！…]+"  #unicode标点符号，。
    text = [re.sub(punc, '<sent>', content) for content in contents]
    return [min([len(s.split(' ')) for s in t.split('<sent>') if s.strip(' ') != '']) for t in text]


# 最后一个字符
def get_last_char(contents):
    vocab = get_vocabulary([content[-1] for content in contents], initial_vocab={}, vocabsize=18)
    last_chars = []
    for content in contents:
        last_char = content[-1]
        if vocab.get(last_char) is not None:
            last_chars.append(vocab[last_char])
        else:
            if re.match(r'[^\u3400-\u9fff]', content[-1]):
                last_chars.append(18)
            else:
                last_chars.append(19)
    return last_chars


# 第一个字符
def get_first_char(contents):
    vocab = get_vocabulary([content[0] for content in contents], initial_vocab={}, vocabsize=10)
    first_chars = []
    for content in contents:
        first_char = content[0]
        if vocab.get(first_char) is not None:
            first_chars.append(vocab[first_char])
        else:
            if re.match(r'[^\u3400-\u9fff]', content[-1]):
                first_chars.append(8)
            else:
                first_chars.append(9)
    return first_chars


# 最后一个词(用tfidf)
def get_last_word(words, vocab, tfidf):
    last_words = []
    for word in words:
        last_word = word[-1]
        if vocab.get(last_word) is not None:
            last_words.append(vocab[last_word])
        else:
            last_words.append(vocab['<unk>'])
    return last_words


# 第一个词(用tfidf)
def get_first_word(words, vocab, tfidf):
    first_words = []
    for word in words:
        first_word = word[0]
        if vocab.get(first_word) is not None:
            first_words.append(vocab[first_word])
        else:
            first_words.append(vocab['<unk>'])
    return first_words


# 获取标签的第一个位置
def first_tag_position(contents, tag):
    positions = []
    for content in contents:
        i = 0
        content = re.sub(' <sssss>', '', content)
        pos = -1
        for word in content.split(' '):
            if word in tag:
                pos = i
                break
            i += 1
        positions.append(pos)
    return positions


# 获取词典，按词频排序
def get_vocabulary(corpus,
                   initial_vocab={
                       '<unk>': 0,
                       '<sssss>': 1
                   },
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
    for text in corpus:
        for w in text.split(' '):
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


def main():
    train_file = 'data/training.txt'
    train_data = read_data(train_file)
    train_data.loc[:, 'text_len'] = get_text_len(train_data.loc[:, 'content'])
    train_data.loc[:, 'pad_content'] = [pad_alltags(content) for content in tqdm(train_data.loc[:, 'content'])]
    train_data['eng_count'] = count_tag(train_data['pad_content'], '<eng>')
    train_data['num_count'] = count_tag(train_data['pad_content'], '<num>')
    train_data['date_count'] = count_tag(train_data['pad_content'], '<date>')
    train_data['time_count'] = count_tag(train_data['pad_content'], '<time>')
    train_data[
        'datetime_count'] = train_data['date_count'] + train_data['time_count']
    train_data['special_count'] = count_tag(train_data['pad_content'], '<special>')
    train_data = train_data.drop('date_count', axis=1)
    train_data = train_data.drop('time_count', axis=1)
    train_data.loc[:, 'count_comma'] = count_comma(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'para_num'] = count_para(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'sent_num'] = count_sent(train_data.loc[:, 'pad_content'])



    # 分词
    LTP_DATA_DIR = 'ltp_data_v3.4.0'
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

    segmentor = Segmentor()
    segmentor.load_with_lexicon(cws_model_path, 'dict/train_freq.dict')

    train_data.loc[:, 'pad_words'] = [
        ' '.join(process_tags(segmentor.segment(content)))
        for content in tqdm(train_data.loc[:, 'pad_content'])
    ]

    segmentor.release()

    train_data.loc[:, 'text_word_len'] = get_text_word_len(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'first_para_len'] = first_para_length(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'first_para_word_len'] = first_para_word_len(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'max_para_len'] = max_para_length(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'min_para_len'] = min_para_length(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'max_para_word_len'] = max_para_word_len(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'min_para_word_len'] = min_para_word_len(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'first_sent_len'] = first_sent_length(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'max_sent_len'] = max_sent_length(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'max_sent_word_len'] = max_sent_word_len(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'min_sent_len'] = min_sent_length(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'min_sent_word_len'] = min_sent_word_len(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'last_char'] = get_last_char(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'first_char'] = get_first_char(train_data.loc[:, 'pad_content'])
    train_data.loc[:, 'first_word'] = get_first_word(train_data.loc[:, 'pad_words'])
    train_data.loc[:, 'last_word'] = get_last_word(train_data.loc[:, 'pad_words'])
    tags = ['eng', 'num', 'date', 'time']
    for tag in tags:
        train_data.loc[:, 'first_%s' % tag] = first_tag_position(train_data.loc[:, 'pad_words'], '<%s>' % tag)