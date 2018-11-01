import pycnn
import readFile
import tensorflow as tf
import csv
import os
def evaluator(flag,input_file,output_file,batch_size,model,dev,memRatio):
    f = readFile.readDS()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    tf.logging.set_verbosity(tf.logging.WARN)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memRatio)
    if flag=="CNN":
        vars_scope = "cnn_main"
    elif flag=="RL":
        vars_scope = "cnn_target"
    else:
        return
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("-- [START] Extract Evaluate File: ",input_file)
        cnn = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,name=vars_scope)
        cnn.build_model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=vars_scope))
        saver.restore(sess,model)
        print("-- [COMPLETE] Loading CNN model")
        file_size = len(open(input_file,'r',encoding='utf-8').readlines())
        print("file size: {}".format(file_size))
        total_batch = int(file_size/batch_size)+1
        fw = csv.writer(open(output_file,'w',encoding='utf-8',newline=''))
        for i in range(total_batch):
            s_idx = i*batch_size
            e_idx = min((i+1)*batch_size,file_size)
            input_x, origin_sentence = f.readBatch(input_file,s_idx,e_idx)
            prediction, score = sess.run([cnn.prediction,cnn.probabilities],feed_dict={cnn.raw_input:input_x})
            for i in range(len(origin_sentence)):
                origin_sentence[i][1] = f.relation2id[prediction[i]]
                origin_sentence[i].append(score[i][prediction[i]])
                fw.writerow(origin_sentence[i])
