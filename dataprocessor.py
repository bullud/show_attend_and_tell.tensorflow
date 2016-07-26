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

phrase = 'train'

#input data
in_annotation_dir = ''
in_feat_dir       = ''
in_filename_list  = ''
in_framenum_list  = ''
in_labels_list    = ''
#output data
out_feat_dir       = ''
out_annotation_path = ''

fea_map_size = 49
kernel_num   = 512

#for train_val
if phrase == 'train':
    in_annotation_dir = '/home/lidian/models/emotion/labels'
    in_feat_dir       = '/home/lidian/models/emotion/origin/Train_Val_face_qiyi_0.8_con_16_resize_256_pool5'
    in_filename_list  = 'train_val_filename.txt'
    in_framenum_list  = 'train_val_framenum.txt'
    in_labels_list    = 'train_val_labels.txt'

    out_feat_dir        = '/home/lidian/models/emotion/datas/Train_Val_face_qiyi_0.8_con_16_resize_256_conv5_3_49_512'
    out_annotation_path = '/home/lidian/models/emotion/datas/train_val_emotion_annotations.pickle'

#for test
elif phrase == 'test':
    in_annotation_dir = '/home/lidian/models/emotion/labels'
    in_feat_dir       = '/home/lidian/models/emotion/origin/Test_sub_face_qiyi_0.8_con_16_resize_256_pool5'
    in_filename_list  = 'test_filename.txt'
    in_framenum_list  = 'test_framenum.txt'
    in_labels_list    = 'test_labels.txt'

    out_feat_dir        = '/home/lidian/models/emotion/datas/Test_sub_face_qiyi_0.8_con_16_resize_256_conv5_3_49_512'
    out_annotation_path = '/home/lidian/models/emotion/datas/test_emotion_annotations.pickle'

else:
    print("phrase unsupported")
    exit()

############################################
videonames = pd.read_table(os.path.join(in_annotation_dir, in_filename_list), sep='\n', names=['videoid'])
videoids = videonames['videoid'].map(lambda x: x[1:10])

framenums  = pd.read_table(os.path.join(in_annotation_dir, in_framenum_list), sep='\n', names=['framenum'])
labels     = pd.read_table(os.path.join(in_annotation_dir, in_labels_list),   sep='\n', names=['label'])

annotations = pd.concat([videoids, framenums, labels], axis =1)

ann = pd.merge(annotations, annotations)

#prepare annotation
ann.to_pickle(out_annotation_path)


#prepare feature
for vid, fn in zip(annotations['videoid'], annotations['framenum']):
    print(vid, fn)
    filename = os.path.join(in_feat_dir, vid + ".dat")

    video_feats = np.zeros((fn, fea_map_size, kernel_num), dtype=np.float32)

    fi = 0
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4)
            if not data:
                break

            ind = struct.unpack("i", data)  # index = bytes2int(fi, 'little')

            fea = f.read(kernel_num * fea_map_size * 4)

            video_feats[fi] = np.fromstring(fea, dtype=np.float32).reshape((kernel_num, fea_map_size)).swapaxes(0, 1) #switch feature the vector

            fi += 1

    if fi != fn:
        print('%d framenum miss match !!!!!!!' % (vid))
        break

    filename = os.path.join(out_feat_dir, vid)
    np.save(filename, arr = video_feats)