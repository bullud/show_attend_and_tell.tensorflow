import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *
import struct

def bytes2int( tb, order='big'):
    if order == 'big': seq=[0,1,2,3]
    elif order == 'little': seq=[3,2,1,0]
    i = 0
    for j in seq: i = (i<<8)+ int(tb[j].encode('hex'), 16)
    return i


class DataProvider():
    def __init__(self, maxFrame, feat_dir, annotation_path):
        self.maxFrame = maxFrame

        self.annotation = pd.read_pickle(annotation_path)
        self.feat_dir = feat_dir

        self.tkf = None
        self.vkf = None
        self.tkf = None

        self.tind = -1
        self.vind = -1
        self.tind = -1


    def getTrainBatch(self):
        idx_list = self.tkf[self.tind]

        self.tind = (self.tind + 1) % len(self.tkf) #update tind

        ann_list = self.annotation.iloc[idx_list[1]]

        print(ann_list)

        #emotions
        emotions = ann_list['label']

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
        feats = np.zeros((len(ann_list), self.maxFrame, 512, 14, 14))

        vi = 0
        for vid in ann_list['videoid']:
            #print(vid, fni)
            filename = os.path.join(self.feat_dir, vid + ".dat")

            video_feats = np.zeros((self.maxFrame, 512, 14, 14), dtype=np.float32)

            fi = 0
            with open(filename, 'rb') as f:
                while True:
                    data = f.read(4)
                    if not data:
                        break

                    ind = struct.unpack("i", data)  # index = bytes2int(fi, 'little')

                    fea = f.read(512 * 14 * 14 * 4)

                    video_feats[fi] = np.fromstring(fea, dtype=np.float32).reshape((512, 14, 14))

                    fi += 1

                    if fi >= self.maxFrame:
                        break

            feats[vi] = video_feats
            vi += 1


        return feats, masks, emotions


    def initEpoch(self, batch_size, shuffle = True):
        n = len(self.annotation)
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        batches = []
        batch_start = 0

        for i in range(n // batch_size):
            batches.append(idx_list[batch_start:
            batch_start + batch_size])
            batch_start += batch_size

        if (batch_start != n):
            # Make a minibatch out of what is left
            batches.append(idx_list[batch_start:])

        self.tkf = zip(range(len(batches)), batches)
        self.tind = 0
        return len(batches)

if __name__ == '__main__':
    feat_dir = '/home/lidian/models/emotion/Train_Val_face_qiyi_0.8_con_16_resize_256_conv5_3'
    annotation_path = '/home/lidian/models/emotion/datas/emotion_annotations.pickle'
    batch_size = 80

    dp = DataProvider(maxFrame = 100, feat_dir = feat_dir, annotation_path= annotation_path)
    num_batch = dp.initEpoch(batch_size=batch_size, shuffle= True)
    #print(num_batch)

    dp.getTrainBatch()


