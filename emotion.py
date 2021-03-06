#-*- coding: utf-8 -*-
import math
import os
import time
import ipdb
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
import dataprovider
from collections import OrderedDict

#from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence

class Emotion_Recognizer():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, n_emotions,  dim_ctx, dim_hidden, n_lstm_steps, batch_size=200, ctx_shape=[196,512], bias_init_vector=None):
        self.n_emotions = n_emotions
        self.dim_ctx = dim_ctx
        self.dim_hidden = dim_hidden
        self.ctx_shape = ctx_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size

        self.init_hidden_W = self.init_weight(dim_ctx, dim_hidden, name='init_hidden_W')
        self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')

        self.init_memory_W = self.init_weight(dim_ctx, dim_hidden, name='init_memory_W')
        self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')

        self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U')
        self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b')

        self.image_encode_W = self.init_weight(dim_ctx, dim_hidden*4, name='image_encode_W')

        self.image_att_W = self.init_weight(dim_ctx, dim_ctx, name='image_att_W')
        self.hidden_att_W = self.init_weight(dim_hidden, dim_ctx, name='hidden_att_W')
        self.pre_att_b = self.init_bias(dim_ctx, name='pre_att_b')

        self.att_W = self.init_weight(dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        self.decode_lstm_W = self.init_weight(dim_hidden, n_emotions, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(n_emotions, name='decode_lstm_b')

        self.decode_emotion_W = self.init_weight(n_emotions, n_emotions, name='decode_emotion_W')
        if bias_init_vector is not None:
            self.decode_emotion_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_emotion_b')
        else:
            self.decode_emotion_b = self.init_bias(n_emotions, name='decode_emotion_b')


    def get_initial_lstm(self, mean_context):
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    def build_model(self):
        context = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps, self.ctx_shape[0], self.ctx_shape[1]])  #change
        emotion = tf.placeholder("int32", [self.batch_size])
        mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])


        # TODO: need to modify
        h, c = self.get_initial_lstm(tf.reduce_mean(context[:, 0, :, :], 1))  #[batch_size, dim_hidden]


        # for labels
        labels = tf.expand_dims(emotion, 1)  # [batch_size, 1]
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)  # [batch_size, 1]
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_emotions]), 1.0, 0.0) #[batch_size, n_emotions]

        loss = 0.0
        hh=[]
        for ind in range(self.n_lstm_steps):
            #add for context
            one_step_context = context[:, ind, :, :]  #[batch_size, 196, 512]
            context_flat = tf.reshape(one_step_context, [-1, self.dim_ctx]) #[batch_size * 196, 512]
            context_encode = tf.matmul(context_flat, self.image_att_W)    #[batch_size * 196, 512]
            context_encode = tf.reshape(context_encode, [-1, ctx_shape[0], ctx_shape[1]]) #[batch_size, 196, 512]

            #for att
            context_encode = context_encode + tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + self.pre_att_b
            #[batch_size, 196, dim_ctx]

            context_encode = tf.nn.tanh(context_encode)

            # context_encode: 3D -> flat required
            context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx]) # (batch_size*196, 512)
            alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b # (batch_size*196, 1)
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]]) #[batch_size, 196]
            alpha = tf.nn.softmax( alpha )

            weighted_context = tf.reduce_sum(context[:, ind, :, :] * tf.expand_dims(alpha, 2), 1) #[batch_size, dim_ctx]

            lstm_preactive = tf.matmul(h, self.lstm_U) + tf.matmul(weighted_context, self.image_encode_W)
            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(new_c) #[batch_size, dim_hidden]

            #add
            hh.append(tf.expand_dims(h, 1))  #[batch_size, 1, dim_hidden]

        #hc = tf.pack(hh)  #[batch_size, n_lstm_steps, dim_hidden]
        hc = tf.concat(1, [hh[i] for i in range(self.n_lstm_steps)])  # [batch_size, n_lstm_steps, dim_hidden]

        #compute mean hm
        hm = tf.reduce_sum(hc * tf.expand_dims(mask, 2), 1)/tf.expand_dims(tf.reduce_sum(mask, 1), 1)  #[batch_size, dim_hidden]

        logits = tf.matmul(hm, self.decode_lstm_W) + self.decode_lstm_b                      #[batch_size, n_emotions]
        logits = tf.nn.relu(logits)
        logits = tf.nn.dropout(logits, 0.5)

        pred_emotions = tf.matmul(logits, self.decode_emotion_W) + self.decode_emotion_b    #[batch_size, n_emotions]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred_emotions, onehot_labels) #[batch_size, 1]

        loss = tf.reduce_mean(cross_entropy)


        return loss, pred_emotions, context, emotion, mask


###### 학습 관련 Parameters ######
n_epochs=500
batch_size=30
n_emotions=7
max_Frame = 30
feat_size = 196
dim_ctx=512
dim_hidden=256
ctx_shape=[feat_size,dim_ctx]
valid_portion = 0.0
pretrained_model_path = './model/model-8'
model_path = './models/'
#############################

###### 잡다한 Parameters #####
trainVal_annotation_path = '/home/lidian/models/emotion/datas/train_val_emotion_annotations.pickle'
trainVal_feat_dir        = '/home/lidian/models/emotion/datas/Train_Val_face_qiyi_0.8_con_16_resize_256_conv5_3_196_512'

test_sub_annotation_path     = '/home/lidian/models/emotion/datas/test_emotion_annotations.pickle'
test_sub_feat_dir            = '/home/lidian/models/emotion/datas/Test_sub_face_qiyi_0.8_con_16_resize_256_conv5_3_196_512'

test_full_annotation_path    = '/home/lidian/models/emotion/datas/test_full_emotion_annotations.pickle'
test_full_feat_dir           = '/home/lidian/models/emotion/datas/Test_full_face_qiyi_0.8_con_16_resize_256_conv5_3_196_512'




#############################

def train(pretrained_model_path=pretrained_model_path):
    print('Parameters: begin ###########################################################')

    print("n_epoches = %d"  %n_emotions)
    print('batch_size = %d' %batch_size)
    print('n_emotion = %d'  %n_emotions)
    print('max_Frame = %d'  %max_Frame)
    print('feat_size = %d'  %feat_size)
    print('dim_ctx = %d'    %dim_ctx)
    print('dim_hidden = %d' %dim_hidden)
    print('ctx_shape  = （%d，%d）' %(feat_size, dim_ctx))
    print('valid_portion = %f'     %valid_portion)
    print('load_model_file = %s'   %pretrained_model_path)
    print('save_model_dir = %s'    %model_path)

    print('trainVal label file = %s'  %trainVal_annotation_path)
    print('trainVal feat dir = %s'    %trainVal_feat_dir)
    print('test_sub label file = %s'  %test_sub_annotation_path)
    print('test_sub feat dir = %s'    %test_sub_feat_dir)
    print('test_full label file = %s' %test_full_annotation_path)
    print('test_full feat dir = %s'   %test_full_feat_dir)

    print('Parameters: end ###########################################################')

    dp = dataprovider.DataProvider(maxFrame = max_Frame, valid_portion = valid_portion,
                                   trainValfeat_dir = trainVal_feat_dir, trainAnnotation_path= trainVal_annotation_path,
                                   testfeat_dir = test_sub_feat_dir, testAnnotation_path = test_sub_annotation_path)
    display_step = 5
    learning_rate = 0.001

    sess = tf.InteractiveSession()

    emotion_recognizer = Emotion_Recognizer(
            n_emotions=n_emotions,        #m
            dim_ctx=dim_ctx,              #D
            dim_hidden=dim_hidden,        #n
            n_lstm_steps=max_Frame,        #
            batch_size=batch_size,
            ctx_shape=ctx_shape,
            bias_init_vector=None)

    loss, pred_emotions, context, emotion, mask = emotion_recognizer.build_model()

    correct_pred = tf.equal(tf.cast(tf.argmax(pred_emotions, 1), tf.int32), emotion)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.initialize_all_variables().run()

    if pretrained_model_path is not None:
        print "Starting with pretrained model"
        saver.restore(sess, pretrained_model_path)

    num_test_batch = dp.initTestEpoch(batch_size = 1, shuffle= True)

    stop_time = 0

    uidx = 0

    print("traning begin!")
    for epoch in range(n_epochs):
        num_batch = dp.initTrainEpoch(batch_size, shuffle = True)

        start_time = time.time()
        for batchi in range(num_batch):

            # Select the random examples for this minibatch
            feats, masks, emotions = dp.getTrainBatch()

            #print("getTrainBatch time： %f sec" %(stop_time - start_time) )

            sess.run(train_op, feed_dict={ context:feats, emotion:emotions, mask:masks})

            if uidx % display_step == 0 :
                stop_time = time.time()

                cost = sess.run(loss, feed_dict={context: feats, emotion: emotions, mask: masks})

                print("epoch %d, batch %d, cost %f, ave batch time %f") % (epoch, batchi, cost, (stop_time - start_time)/display_step)

                start_time = time.time()

            uidx += 1

        acc_mean = 0.0
        for test_batchi in range(num_test_batch):
            tfeats, tmasks, temotions = dp.getTestBatch()
            acc = sess.run(accuracy, feed_dict={context: tfeats, emotion: temotions, mask: tmasks})
            acc_mean += acc

        acc_mean = acc_mean / num_test_batch

        print("epoch %d, accurracy %f ") % (epoch, acc_mean)

        saver.save(sess, os.path.join(model_path, 'model-' + str(acc_mean)), global_step=epoch)



def test(maxFrame = max_Frame, model_path = 'model/model-0.290196103208-123',
         testfeat_dir = test_full_feat_dir, testAnnotation_path = test_full_annotation_path):


    dp = dataprovider.DataProvider(maxFrame=maxFrame, feat_size=ctx_shape[0], ctx_dim=ctx_shape[1], \
                                   testfeat_dir=testfeat_dir, testAnnotation_path=testAnnotation_path)

    testAnnotation = pd.read_pickle(testAnnotation_path)

    sess = tf.InteractiveSession()

    emotion_recognizer = Emotion_Recognizer(
            n_emotions=n_emotions,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxFrame,
            batch_size=1,
            ctx_shape=ctx_shape)

    loss, pred_emotions, context, emotion, mask = emotion_recognizer.build_model()

    pred_softmax = tf.nn.softmax(pred_emotions)

    saver = tf.train.Saver()

    saver.restore(sess, model_path)

    num_test_batch = dp.initTestEpoch(1, shuffle=False) #must set to (1, False） !!!

    result = np.zeros((num_test_batch, 7))
    i = 0
    for batchi, vid in zip(range(num_test_batch), testAnnotation['videoid'].values):

        feats, masks, emotions = dp.getTestBatch()

        pred = sess.run(pred_softmax, feed_dict={context: feats, emotion: emotions, mask: masks})

        em = np.argmax(pred)

        print("vid %s, emotion %d") % (vid, em)
        result[i, : ] = pred
        #result.apnd(zip(vid, pred))
        #result[vid] = pred
        i+=1

#    ipdb.set_trace()

    np.save('test_result', result)

    print('done! total %d test videos' %num_test_batch)



#test()
train(pretrained_model_path=None)