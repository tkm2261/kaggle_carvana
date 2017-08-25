import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

ids_test = ids_valid_split

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


rles = []

model.load_weights(filepath='weights/best_weights.hdf5')

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))

gamma = 1.2
look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0

for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
k = 1.3
shape_operator = np.array([[0,        -k, 0],
                  [-k, 1 + 4 * k, -k],
                  [0,         -k, 0]])


for start in tqdm(range(0, len(ids_test), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread('input/train/{}.jpg'.format(id))
        img = cv2.LUT(img, look_up_table)
        #img = cv2.filter2D(img, -1, shape_operator)
        img = cv2.resize(img, (input_size, input_size))
        
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for i, id in enumerate(ids_test_batch.values):
        with open('data/valid_bright/{}.pkl'.format(id), 'wb') as f:
            pickle.dump(preds[i], f, -1)
