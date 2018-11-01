import pycnn
import readFile
import tensorflow as tf
import csv
import os
import re
import itertools
from konlpy.tag import Twitter
import numpy as np
import datetime

def spTagger(s):
    tagged_sentence=[]
    s = s.replace(" [", " << ")
    s = s.replace("] ", " >> ")
    entities = re.findall("<<.*?>>",s)
    pairs = itertools.permutations(entities,2)
    for pair in pairs:
        e1 = "<e1>%s</e1>" % pair[0][3:-3]
        e2 = "<e2>%s</e2>" % pair[1][3:-3]
        tmp_s = s.replace(pair[0],e1)
        tmp_s = tmp_s.replace(pair[1],e2)
        tagged_sentence.append(tmp_s)
    return tagged_sentence

def posTagger(s,tw):
    e1 = s[s.find("<e1>"):s.find("</e1>") + 5]
    e2 = s[s.find("<e2>"):s.find("</e2>") + 5]
    s = s.replace(e1, " <(_sbj_)> ", 1)
    s = s.replace(e2, " <(_obj_)> ", 1)
    tokens = tw.pos(s, norm=True, stem=True)
    s = ""
    for token in tokens:
        token = list(token)
        s += token[0] + "/" + token[1] + " "
    entities = re.findall("<</Punctuation.*?>>/Punctuation", s)
    for i in range(len(entities)):
        r = ""
        t_list = entities[i].split(" ")
        for e in t_list:
            if e == "<</Punctuation" or e == ">>/Punctuation":  continue
            r += e.split("/")[0]
        r += "/Entity"
        s = s.replace(entities[i], r, 1)
    s = s.replace("<(_/Punctuation obj/Alpha _)>/Punctuation", "<<_obj_>>", 1)
    s = s.replace("<(_/Punctuation sbj/Alpha _)>/Punctuation", "<<_sbj_>>", 1)
    e1 = e1[4:e1.find("</e1>")] + "/Entity"
    e2 = e2[4:e2.find("</e2>")] + "/Entity"
    return e1, e2, s


def pos_embed(x):
    if x < -60:  return 0
    if x >= -60 and x <= 60:  return x + 61
    if x > 60:    return 121

def sen2vec(e1,e2,s,wv,posVec,emb_size):
    s_tokens = s.rstrip().split(" ")
    senVec = np.zeros((300,emb_size+10),dtype=float)
    p1 = s_tokens.index("<<_sbj_>>")
    p2 = s_tokens.index("<<_obj_>>")
    s_tokens[p1] = e1
    s_tokens[p2] = e2
    for i,word in enumerate(s_tokens):
        if word not in wv:  continue
        word_vec = wv[word]
        pE1 = posVec[pos_embed(p1-i)]
        pE2 = posVec[pos_embed(p2-i)]
        tmp = np.append(word_vec,pE1)
        tmp = np.append(tmp,pE2)
        senVec[i] = tmp
    return [senVec]

def extractor(flag,input_file,out_one,out_all,batch_size,model,dev,memRatio,threshold):
    fw_one= csv.writer(open(out_one,'w',encoding='utf-8',newline=''),delimiter='\t')
    fw_all = csv.writer(open(out_all,'w',encoding='utf-8',newline=''),delimiter='\t')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memRatio)
    f = readFile.readDS()
    file_size = len(open(input_file, 'r', encoding='utf-8').readlines())
    total_batch = int(file_size / batch_size) + 1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    tf.logging.set_verbosity(tf.logging.WARN)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    print("-- [START] Extract relations: ",input_file)
    print("-- file size: {}, total batch: {}".format(file_size, total_batch))
    if flag=="CNN":
        vars_scope = "cnn_main"
    elif flag=="RL":
        vars_scope = "cnn_target"
    else:
        return
    cnn = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,name=vars_scope)
    cnn.build_model()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=vars_scope))
    saver.restore(sess,model)
    print("-- [COMPLETE] Loading CNN model")
    tw = Twitter()
    print("-- [START] Extract File: ",input_file)
    for i in range(total_batch):
        s_idx = i*batch_size
        e_idx = min((i+1)*batch_size,file_size)
        sentence = list(csv.reader(open(input_file,'r',encoding='utf-8'),delimiter='\t'))[s_idx:e_idx]
        for line in sentence:
            tagged_sentences = spTagger(line[0])
            for s in tagged_sentences:
                e1,e2,s = posTagger(s,tw)
                x = sen2vec(e1,e2,s,f.wv_model,f.positionVec,f.embedding_size)
                prediction, cs = sess.run([cnn.prediction,cnn.probabilities],feed_dict={cnn.raw_input:x})
                if cs[0][prediction[0]]<threshold:  continue
                relation = f.relation2id[prediction[0]]
                score = cs[0][prediction[0]]
                e1 = e1.replace("/Entity","")
                e2 = e2.replace("/Entity","")
                out_form = [e1,relation,e2,".",score,line[0],line[1],line[2],line[3]]
                fw_one.writerow(out_form)
                for i in range(len(f.relation2id)):
                    out_form = [e1,f.relation2id[i],e2,".",cs[0][i],line[0],line[1],line[2],line[3]]
                    fw_all.writerow(out_form)
    print("[END] ",datetime.datetime.now())


                


