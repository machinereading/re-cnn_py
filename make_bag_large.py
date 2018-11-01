import csv
import random
from gensim.models import KeyedVectors
import numpy as np

class readDS():
    def __init__(self):
        self.train_data = "../data/train_data.tsv"
        self.test_data = "../data/test_data.tsv"
        self.sequence_length=80
        self.wv_model = KeyedVectors.load_word2vec_format("../data/w2v_model.vec")
        print("-- complete loading word embedding")
        self.positionVec = np.zeros((122,5),float)
        for i, r in enumerate(open("../data/positionVec.data", 'r').readlines()):
            r = r.strip().split("\t")
            self.positionVec[i] = np.array([float(x) for x in r])
        self.properties = "../data/ds_label_properties.txt"
        self.relation2id = open(self.properties, 'r', encoding='utf-8').read().rstrip().split("\n")
        self.num_classes = len(self.relation2id)
        self.train_data_size = len(open(self.train_data,'r',encoding='utf-8').readlines())
        self.test_data_size = len(open(self.test_data,'r',encoding='utf-8').readlines())
        self.embedding_size = self.wv_model.vector_size

    def id2relation(self,ids):
        labels=[]
        for id in ids:
            labels.append(self.relation2id[id])
        return labels
    def pos_embed(self,x):
        if x< -60:  return 0
        if x>=-60 and x<=60:  return x+61
        if x>60:    return 121

    def create_bag(self):
        self.bag_sentIDs={}
        self.bag_label={}
        self.all_labels=[]
        for idx, line in enumerate(csv.reader(open(self.train_data,'r',encoding='utf-8'),delimiter='\t')):
            relation = self.relation2id.index(line[1])
            self.all_labels.append(relation)
            key = line[5]+","+line[6]+","+str(relation)
            if key in self.bag_sentIDs:
                self.bag_sentIDs[key].append(idx)
            else:
                self.bag_sentIDs[key] = [idx]
                self.bag_label[key] = relation

    def sampling_sents(self,key,min_id,max_id):
        keys = self.bag_sentIDs.keys()
        num_sents = len(self.bag_sentIDs[key])
        sampled_sentences = []
        while len(sampled_sentences)<num_sents*10:
            selected_key = random.sample(keys,1)[0]
            if selected_key==key:   continue
            for i in self.bag_sentIDs[selected_key]:
                sampled_sentences.append(i)
        for i in sampled_sentences:
            if i<min_id or i>max_id:    sampled_sentences.remove(i)
        return sampled_sentences

    def read_batch_data_with_id(self,sentIDs,st,en):
        # print("--reading raw data from {} to {} for {}".format(st,en,flag))
        sentence_data = list(csv.reader(open(self.train_data,'r',encoding='utf-8'),delimiter='\t'))[st:en]
        w2v_sentences=[]
        relations = []
        for idx, line in enumerate(sentence_data):
            sent_id = st+idx
            if sent_id in sentIDs:
                sentence = line[0]
                s_tokens = sentence.rstrip().split(" ")
                relations.append(self.relation2id.index(line[1]))
                tmp_s = np.zeros((self.sequence_length, self.embedding_size + 10), dtype=float)
                p1 = s_tokens.index("<<_sbj_>>")
                p2 = s_tokens.index("<<_obj_>>")
                s_tokens[p1] = line[5]
                s_tokens[p2] = line[6]
                for i, word in enumerate(s_tokens):
                    if word not in self.wv_model:   continue
                    word_vec = self.wv_model[word]
                    pE1 = self.positionVec[self.pos_embed(p1 - i)]
                    pE2 = self.positionVec[self.pos_embed(p2 - i)]
                    tmp = np.append(word_vec, pE1)
                    tmp = np.append(tmp, pE2)
                    tmp_s[i] = tmp
                w2v_sentences.append(tmp_s)

        return w2v_sentences, relations

    def read_batch_data(self,flag,st,en):
        # print("--reading raw data from {} to {} for {}".format(st,en,flag))
        if flag=="train":
            sentence_data = list(csv.reader(open(self.train_data,'r',encoding='utf-8'),delimiter='\t'))[st:en]
        elif flag=="test":
            sentence_data = list(csv.reader(open(self.test_data,'r',encoding='utf-8'),delimiter='\t'))[st:en]
        else:   return

        w2v_sentences=[]
        relations=[]

        for idx, line in enumerate(sentence_data):
            sentence = line[0]
            s_tokens = sentence.rstrip().split(" ")
            if len(s_tokens)>self.sequence_length:  continue
            relations.append(self.relation2id.index(line[1]))
            tmp_s = np.zeros((self.sequence_length,self.embedding_size+10),dtype=float)
            p1 = s_tokens.index("<<_sbj_>>")
            p2 = s_tokens.index("<<_obj_>>")
            s_tokens[p1] = line[5]
            s_tokens[p2] = line[6]
            for i, word in enumerate(s_tokens):
                if word not in self.wv_model:   continue
                word_vec = self.wv_model[word]
                pE1 = self.positionVec[self.pos_embed(p1-i)]
                pE2 = self.positionVec[self.pos_embed(p2-i)]
                tmp = np.append(word_vec,pE1)
                tmp = np.append(tmp,pE2)
                tmp_s[i] = tmp
            w2v_sentences.append(tmp_s)

        return w2v_sentences, relations

if __name__ =="__main__":
    f = readDS()
