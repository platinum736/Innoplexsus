import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize


test_info = pd.read_csv('information_test.csv', sep='\t')
train_info = pd.read_csv('information_train.csv', sep='\t')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train_info.shape)
print(train.shape)
print(test_info.shape)
print(test.shape)
print(sum(train_info['full_Text'].isna()))
print(sum(test_info['full_Text'].isna()))


'''def create_edgelist(req_set, train_info, train):
    edgelist = pd.DataFrame()
    edgelist['from']= []
    edgelist['to']= []
    for i in range(0,train.shape[0]):
        from_doc = train.iloc[i]['pmid']
        if int(train_info[train_info['pmid']==from_doc]['set']) == req_set:
            from_author_list = list(train_info[train_info['pmid']==from_doc]['author_str'])[0].split(',')
            to_doc = train.iloc[i]['ref_list']
            to_author_list = list()
            for doc in to_doc:
                to_author_list.append(list(train_info[train_info['pmid']==doc]['author_str'])[0].split(','))
            to_author_list = set(to_author_list)
            for from_author in from_author_list:
                for to_author in to_author_list:
                    edgelist['from'].append(from_author)
                    edgelist['to'].append(to_author)
    return edgelist'''


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def preprocess(df):
    text = df['article_title']
    text = text + ' ' + df['abstract']
    if not df.isnull()[6]:
        text = text+' '+df['full_Text']
    words = nltk.word_tokenize(text)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    return words


def createVocab(train_info,test_info):
    vocab_words = []
    all_words=[]
    for i in range(0, train_info.shape[0]):
        print(i)
        words = preprocess(train_info.iloc[i])
        all_words.append(words)
        #train_info.iloc[i]['words'] = words
        vocab_words.append(set(words))
    train_info['words']=all_words
    all_words = []
    for i in range(0,test_info.shape[0]):
        print(i)
        words = preprocess(train_info.iloc[i])
        all_words.append(words)
        #train_info.iloc[i]['words'] = words
        vocab_words.append(set(words))
    test_info['words']=all_words
    print(len(vocab_words))
    return vocab_words,train_info,test_info


def createRepresentation(train_info, test_info, vocabulary):
    nrow = train_info.shape[0] + test_info.shape[0]
    vocabulary = list(vocabulary)
    vocabulary_map = dict()
    for i in range(0,len(vocabulary)):
        vocabulary_map[vocabulary[i]]=i
    df = np.zeros([nrow,len(vocabulary)])
    for i in range(0,train_info.shape[0]):
        for word in train_info.iloc[i]['words']:
            df[i][vocabulary_map[word]] = 1
    j = i

    for i in range(0,test_info.shape[0]):
        for word in test_info.iloc[i]['words']:
            df[j+i][vocabulary_map[word]] = 1
    return df


def createTrainSet(train_info,vocabulary,train,df):
    trainset=pd.DataFrame()
    for i in range(0,train.shape[0]):
        print('train'+str(i))
        c_pmid = train.iloc[i]['pmid']
        rowindex = train_info[train_info['pmid']==c_pmid].index.values[0]
        row = df[rowindex]
        ref_list = train.iloc[i]['ref_list']
        c_set = train_info[train_info['pmid']==train.iloc[i]['pmid']]['set'].values[0]
        for item in ref_list:
            refIndex = train_info[train_info['pmid']==item].index.values[0]
            full_row = row
            full_row.extend(df[refIndex])
            full_row.extend([1])
            trainset.append([full_row])

        for j in range(0,train_info.shape[0]):
            if train_info.iloc[j]['pmid'] != c_pmid \
                and train_info.iloc[j]['pmid'] not in ref_list \
                    and train_info.iloc[j]['set']==c_set:
                nonrefIndex = train_info.iloc[j].index.values[0]
                full_row = row
                full_row.extend(df[nonrefIndex])
                full_row.extend([0])
                trainset.append([full_row])
    return trainset


def createTestSet(test_info, vocabulary, test, df):
    testset = pd.DataFrame()
    for i in range(0,test.shape[0]):
        print('test'+str(i))
        c_pmid = test.iloc[i]['pmid']
        rowindex = test_info[test_info['pmid']==c_pmid].index.values[0]
        row = df[rowindex]
        c_set = train_info[test_info['pmid']==test.iloc[i]['pmid']]['set'].values[0]

        for j in range(0,test_info.shape[0]):
            if test.iloc[j]['pmid'] != c_pmid \
                    and test_info.iloc[j]['set']!=c_set:
                Index = test_info.iloc[j].index.values[0]
                full_row = row
                full_row.extend(df[Index])
                testset.append([full_row])
    return testset



vocab,train_info,test_info = createVocab(train_info,test_info)
vocabulary = []
for vocabList in vocab:
    vocabulary.extend(vocabList)
vocabulary = set(vocabulary)
print(len(vocabulary))
print(train_info.columns)
df = createRepresentation(train_info,test_info,vocabulary)
train_set = createTrainSet(train_info,vocabulary,train,df)
train_set.to_csv('trainData.csv')
ntrainRows = train_info.shape[0]
df_test = df[ntrainRows:]
test_set = createTrainSet(test_info,vocabulary,test,df_test)
test_set.to_csv('testData.csv')

