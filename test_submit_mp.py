import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from tqdm import tqdm

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []


q_size = 10
from multiprocessing import Pool

params = cv2.SimpleBlobDetector_Params()
#params.minThreshold = 100
#params.maxThreshold = 800

params.filterByArea = True
params.maxArea = 800
params.minArea = 1

params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

def fill(ax, img, cnt=0):
    if cnt > 1000:
        raise Exception('too many')
    try:
        if img[ax] == False:
            return
    except IndexError:
        return
    img[ax] = False
    fill((ax[0] + 1, ax[1]), img, cnt + 1)
    fill((ax[0] - 1, ax[1]), img, cnt + 1)
    fill((ax[0], ax[1] + 1), img, cnt + 1)
    fill((ax[0], ax[1] - 1), img, cnt + 1)


def exe(id):
    pred3 = cv2.imread('data/test_0824/{}.png'.format(id), cv2.IMREAD_GRAYSCALE).astype(float)    
    pred = cv2.imread('data/test_0822/{}.png'.format(id), cv2.IMREAD_GRAYSCALE).astype(float)
    pred1 = cv2.imread('data/test/{}.png'.format(id), cv2.IMREAD_GRAYSCALE).astype(float)
    pred2 = cv2.imread('data/test2/{}.png'.format(id), cv2.IMREAD_GRAYSCALE).astype(float)
    pred = (pred + pred1 + pred2 + pred3) / 4
    cv2.imwrite('data/test_ens_0825/{}.png'.format(id), pred.astype(np.uint8))
        
    prob = cv2.resize(pred, (orig_width, orig_height))
        
    mask = prob > 127 #threshold
    """
    img = ((prob > 127) * 255).astype(np.uint8)
    keypoints = detector.detect((255 - img).astype(np.uint8))
    try:
        for pt in keypoints:
            p = pt.pt
            fill((int(p[1]), int(p[0])), mask)
    except Exception:
        print(id)
        mask = prob > 127 #threshold
    """        
    rle = run_length_encode(mask)
    return rle

if __name__ == '__main__':

    p = Pool()
    rles = list(p.map(exe, tqdm(ids_test)))
    p.close()
    p.join()

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
