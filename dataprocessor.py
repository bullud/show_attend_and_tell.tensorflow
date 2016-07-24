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

maxframe = 100

annotation_dir = '/home/lidian/models/emotion/labels'
feat_dir       = '/home/lidian/models/emotion/Train_Val_face_qiyi_0.8_con_16_resize_256_conv5_3'

#output data
feat_result_path = '/home/lidian/models/emotion/datas/emotion_feats.npy'
mask_result_path = '/home/lidian/models/emotion/datas/mask.npy'
annotation_path = '/home/lidian/models/emotion/datas/emotion_annotations.pickle'


videonames = pd.read_table(os.path.join(annotation_dir, 'train_val_filename.txt'), sep='\n', names=['videoid'])
videoids = videonames['videoid'].map(lambda x: x[1:10])

framenums  = pd.read_table(os.path.join(annotation_dir, 'train_val_framenum.txt'), sep='\n', names=['framenum'])
labels     = pd.read_table(os.path.join(annotation_dir, 'train_val_labels.txt'),   sep='\n', names=['label'])

annotations = pd.concat([videoids, framenums, labels], axis =1)

#annotations.to_pickle(annotation_result_path)

ann = pd.merge(annotations, annotations)

ann.to_pickle(annotation_path)

exit()

print(len(videoids))
print(framenums['framenum'].min())

videonum = len(annotations)

total_feas = np.zeros((videonum, maxframe, 512, 14, 14))
total_mask = np.zeros((videonum, maxframe,))

vi = 0
for vid, fn in zip(annotations['videoid'], annotations['framenum']):
    print(vid, fn)
    filename = os.path.join(feat_dir, vid + ".dat")
    video_feas = np.zeros((maxframe, 512, 14, 14), dtype=np.float32)
    video_mask = np.zeros((maxframe))

    if fn > maxframe:
        video_mask[0: maxframe] = 1
    else:
        video_mask[0: fn] = 1

    fi = 0
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4)
            if not data:
                break

            ind = struct.unpack("i", data) # index = bytes2int(fi, 'little')

            fea = f.read(512*14*14*4)

            video_feas[fi] = np.fromstring(fea, dtype=np.float32).reshape((512, 14, 14))

            fi += 1

            if fi >= maxframe:
                break

    total_feas[vi] = video_feas
    vi += 1

np.save(feat_result_path, total_feas)
np.save(mask_result_path, total_mask)

exit()

'''
annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

unique_images = annotations['image'].unique()
image_df = pd.DataFrame({'image':unique_images, 'image_id':range(len(unique_images))})

annotations = pd.merge(annotations, image_df)
annotations.to_pickle(annotation_result_path)

if not os.path.exists(feat_path):
    feats = cnn.get_features(unique_images, layers='conv5_3', layer_sizes=[512,14,14])
    np.save(feat_path, feats)
'''