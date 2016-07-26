import pandas as pd
import numpy as np
import os
import cPickle
#from cnn_util import *
import struct
from collections import OrderedDict

def bytes2int( tb, order='big'):
    if order == 'big': seq=[0,1,2,3]
    elif order == 'little': seq=[3,2,1,0]
    i = 0
    for j in seq: i = (i<<8)+ int(tb[j].encode('hex'), 16)
    return i


class DataProvider():
    def __init__(self, maxFrame, valid_portion, trainValfeat_dir, trainAnnotation_path, testfeat_dir, testAnnotation_path):
        self.maxFrame = maxFrame

        self.trainValAnnotation = pd.read_pickle(trainAnnotation_path)
        self.trainValfeat_dir = trainValfeat_dir

        self.testAnnotation = pd.read_pickle(testAnnotation_path)
        self.testfeat_dir = testfeat_dir

        self.tkf = None
        self.vkf = None
        self.ekf = None

        self.tind = -1
        self.vind = -1
        self.eind = -1

        self.valid_portion = valid_portion

        n = len(self.trainValAnnotation)
        self.n_train = n * (1 - self.valid_portion)
        self.n_val = n - self.n_train

        sidx = np.random.permutation(n)
        self.train_set = sidx[:self.n_train]
        self.valid_set = sidx[self.n_train:]

        self.test_set = range(len(self.testAnnotation))

        self.mem_caches = OrderedDict()
        #print(self.train_set)

    def getFeature(self, ann_list, feat_dir, cachable = False):

        #print(ann_list)

        #emotions
        emotions = np.array(ann_list['label'])

        fn = ann_list['framenum']

        #mask
        masks = np.zeros((len(ann_list), self.maxFrame))

        i = 0
        for fni in fn:
            if fni > self.maxFrame:
                for j in range(self.maxFrame):
                    masks[i, j] = 1
            else:
                for j in range(fni):
                    masks[i, j] = 1
            i += 1

        #feat
        feats = np.zeros((len(ann_list), self.maxFrame, 49, 512))

        vi = 0
        for vid in ann_list['videoid']:
            '''
            filename = os.path.join(self.feat_dir, vid + ".dat")

            video_feats = np.zeros((self.maxFrame, 196, 512), dtype=np.float32)

            fi = 0
            with open(filename, 'rb') as f:
                while True:
                    data = f.read(4)
                    if not data:
                        break

                    ind = struct.unpack("i", data)  # index = bytes2int(fi, 'little')

                    fea = f.read(512 * 14 * 14 * 4)

                    video_feats[fi] = np.fromstring(fea, dtype=np.float32).reshape((512, 196)).swapaxes(0, 1)

                    fi += 1

                    if fi >= self.maxFrame:
                        break
            '''

            #disk_cache_filename = os.path.join(feat_dir, vid + "-" + str(self.maxFrame) + "-cache.npy")
            #if False: #os.path.isfile(cache_filename):
            #    feats[vi] = np.load(disk_cache_filename)
            #else:
            fea = None
            if self.mem_caches.has_key(vid):
                #print('use cache')
                fea = self.mem_caches[vid]
            else:
                filename = os.path.join(feat_dir, vid + ".npy")

                fea = np.load(filename)
                if cachable:
                    self.mem_caches[vid] = fea

            video_feats = np.zeros((self.maxFrame, 49, 512), dtype=np.float32)

            fn = fea.shape[0]
            if fn > self.maxFrame:
                fn = self.maxFrame
            #print(fn)
            for i in range(fn):
                video_feats[i] = fea[i]

            #np.save(cache_filename, video_feats)

            feats[vi] = video_feats

            ######################################################

            vi += 1

        return feats, masks, emotions

    def getTrainBatch(self):
        idx_list = self.tkf[self.tind]

        self.tind = (self.tind + 1) % len(self.tkf)  # update tind

        ann_list = self.trainValAnnotation.iloc[idx_list[1]]

        return self.getFeature(ann_list, self.trainValfeat_dir, True)


    def initTrainEpoch(self, batch_size, shuffle = True):
        if shuffle:
            np.random.shuffle(self.train_set)

        batches = []
        batch_start = 0

        for i in range(len(self.train_set) // batch_size):
            batches.append(self.train_set[batch_start:
            batch_start + batch_size])
            batch_start += batch_size

        #if (batch_start != len(self.train_set)):
            # Make a minibatch out of what is left
        #    batches.append(self.train_set[batch_start:])

        self.tkf = zip(range(len(batches)), batches)
        self.tind = 0
        return len(batches)

    def getValidBatch(self):
        idx_list = self.tkf[self.vind]

        self.vind = (self.vind + 1) % len(self.vkf)  # update vind

        ann_list = self.trainValAnnotation.iloc[idx_list[1]]

        return self.getFeature(ann_list, self.trainValfeat_dir)

    def initValidEpoch(self, batch_size, shuffle=False):
        if shuffle:
            np.random.shuffle(self.valid_set)

        batches = []
        batch_start = 0

        for i in range(len(self.valid_set) // batch_size):
            batches.append(self.valid_set[batch_start:
            batch_start + batch_size])
            batch_start += batch_size

            # if (batch_start != n):
            # Make a minibatch out of what is left
            # batches.append(idx_list[batch_start:])

        self.vkf = zip(range(len(batches)), batches)
        self.vind = 0
        return len(batches)

    def getTestBatch(self):
        idx_list = self.ekf[self.eind]

        self.eind = (self.eind + 1) % len(self.ekf)  # update eind

        ann_list = self.testAnnotation.iloc[idx_list[1]]

        return self.getFeature(ann_list, self.testfeat_dir, True)

    def initTestEpoch(self, batch_size, shuffle=False):
        if shuffle:
            np.random.shuffle(self.test_set)

        batches = []
        batch_start = 0

        for i in range(len(self.test_set) // batch_size):
            batches.append(self.test_set[batch_start:
            batch_start + batch_size])
            batch_start += batch_size

        self.ekf = zip(range(len(batches)), batches)
        self.eind = 0
        return len(batches)

if __name__ == '__main__':
    feat_dir = '/home/lidian/models/emotion/Train_Val_face_qiyi_0.8_con_16_resize_256_conv5_3'
    annotation_path = '/home/lidian/models/emotion/datas/emotion_annotations.pickle'
    batch_size = 80

    dp = DataProvider(maxFrame = 100, feat_dir = feat_dir, annotation_path= annotation_path)
    num_batch = dp.initEpoch(batch_size=batch_size, shuffle= True)
    #print(num_batch)

    dp.getTrainBatch()


