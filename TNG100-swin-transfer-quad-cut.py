from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import os
import sys
import math
import numpy as np

import glob

#import pandas as pd
#import matplotlib.pyplot as plt

import pickle

from tf_fits.image import image_decode_fits
from tensorflow_addons.image import rotate as tfa_image_rotate
from math import pi
AUTOTUNE = tf.data.experimental.AUTOTUNE


#Check if GPUs. If there are, some code to fix cuDNN bugs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for i in range(0, len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print('No GPU')
strategy = tf.distribute.experimental.CentralStorageStrategy()


path = '/mnt/home/wpearson/merger_challenge/TNG/data_tng/train/cut_128/'

train_path = path+'train/'
valid_path = path+'valid/'

extn = '.fits'

train_non_images = glob.glob(train_path+'*.NM.*'+extn)
train_non_image_count = len(train_non_images)
train_pre_images = glob.glob(train_path+'*.PR.*'+extn)
train_pre_image_count = len(train_pre_images)
train_ong_images = glob.glob(train_path+'*.OG.*'+extn)
train_ong_image_count = len(train_ong_images)
train_pst_images = glob.glob(train_path+'*.PO.*'+extn)
train_pst_image_count = len(train_pst_images)
train_image_count = np.min((train_non_image_count, train_pre_image_count, train_ong_image_count, train_pst_image_count))*4

valid_non_images = glob.glob(valid_path+'*.NM.*'+extn)
valid_non_image_count = len(valid_non_images)
valid_pre_images = glob.glob(valid_path+'*.PR.*'+extn)
valid_pre_image_count = len(valid_pre_images)
valid_ong_images = glob.glob(valid_path+'*.OG.*'+extn)
valid_ong_image_count = len(valid_ong_images)
valid_pst_images = glob.glob(valid_path+'*.PO.*'+extn)
valid_pst_image_count = len(valid_pst_images)
valid_image_count = np.min((valid_non_image_count, valid_pre_image_count, valid_ong_image_count, valid_pst_image_count))*4

EPOCHS = 30
BATCH_SIZE = 64
VALID_BATCH_SIZE = 64 #160

STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE).astype(int)
STEPS_PER_VALID_EPOCH = np.ceil(valid_image_count/VALID_BATCH_SIZE).astype(int)

IMG_HEIGHT = 112 #112
IMG_WIDTH = 112
IMAGE_SIZE = [2*IMG_HEIGHT, 2*IMG_WIDTH]
edge_cut = (128 - IMG_HEIGHT)//2
CROP_FRAC = IMG_HEIGHT/(edge_cut+edge_cut+IMG_HEIGHT)
CH = [0,1,2] #ugri

OFSET = 500.
SCALE = 1500.

print(train_image_count, STEPS_PER_EPOCH)
print(valid_image_count, STEPS_PER_VALID_EPOCH)
print()
print(train_non_image_count, train_pre_image_count, train_ong_image_count, train_pst_image_count)
print(valid_non_image_count, valid_pre_image_count, valid_ong_image_count, valid_pst_image_count)


MODEL_PATH = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224_fe/1"

WARMUP_STEPS = 10
INIT_LR = 0.03
WAMRUP_LR = 0.006

TOTAL_STEPS = int((train_image_count / BATCH_SIZE) * EPOCHS)


#@tf.function
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The last is the file name, split the name
    sub_name = tf.strings.split(parts[-1], '.')

    if sub_name[7] == 'PR':
        value = 1
    elif sub_name[7] == 'OG':
        value = 2
    elif sub_name[7] == 'PO':
        value = 3
    else:
        value = 0
    return value

#@tf.function
def decode_image(byte_data):
    #Get the image from the byte string
    img = image_decode_fits(byte_data, 0) 
    img = tf.reshape(img, (1,128,128))
    img = tf.transpose(img,[1,2,0])
    return img

def process_path(file_path):
    label = get_label(file_path)
    byte_data = tf.io.read_file(file_path)
    img = decode_image(byte_data)
    return img, label


from time import time
g = tf.random.Generator.from_seed(int(time()))

#@tf.function
def augment_img(img, label):
    img = tf.image.rot90(img, k=g.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    
    return img, label

#@tf.function
def crop_img(img, label):
    img = tf.slice(img, [edge_cut,edge_cut,0], [IMG_HEIGHT,IMG_HEIGHT,1])
    
    img = tf.math.asinh(img)
    
    chans = []
    for i in CH:
        chan = tf.slice(img, [0,0,0], [IMG_HEIGHT,IMG_HEIGHT,1])
        chan = tf.reshape(chan, [IMG_HEIGHT,IMG_HEIGHT])
        mini = tf.reduce_min(chan)
        maxi = tf.reduce_max(chan)
        numerator = tf.math.subtract(chan, mini)
        denominator = tf.math.subtract(maxi, mini)
        chan = tf.math.divide(numerator, denominator)
        #chan = tf.subtract(chan, i_mean[i])
        #chan = tf.math.divide(chan, i_std[i])
        chans.append(chan)
    img = tf.convert_to_tensor(chans)
    img = tf.transpose(img,[1,2,0])
    
    img = tf.image.resize(img, (IMG_HEIGHT*2, IMG_WIDTH*2), method='nearest')
    
    return img, label


#@tf.function
def prepare_dataset(ds, batch_size, shuffle_buffer_size=1000, training=False):
    #Load images and labels
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    #cache result
    ds = ds.cache()
    #shuffle images
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    
    #Augment Image
    if training:
        ds = ds.map(augment_img, num_parallel_calls=AUTOTUNE)
    ds = ds.map(crop_img, num_parallel_calls=AUTOTUNE)
    
    #Set batches and repeat forever
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


list_train_non_ds = tf.data.Dataset.list_files(train_path+'*.NM.*'+extn)
train_non_ds = prepare_dataset(list_train_non_ds, BATCH_SIZE//4, train_non_image_count, True)
dist_train_non_ds = strategy.experimental_distribute_dataset(train_non_ds)
train_non_iter = iter(dist_train_non_ds)
list_train_pre_ds = tf.data.Dataset.list_files(train_path+'*.PR.*'+extn)
train_pre_ds = prepare_dataset(list_train_pre_ds, BATCH_SIZE//4, train_pre_image_count, True)
dist_train_pre_ds = strategy.experimental_distribute_dataset(train_pre_ds)
train_pre_iter = iter(dist_train_pre_ds)
list_train_ong_ds = tf.data.Dataset.list_files(train_path+'*.OG.*'+extn)
train_ong_ds = prepare_dataset(list_train_ong_ds, BATCH_SIZE//4, train_ong_image_count, True)
dist_train_ong_ds = strategy.experimental_distribute_dataset(train_ong_ds)
train_ong_iter = iter(dist_train_ong_ds)
list_train_pst_ds = tf.data.Dataset.list_files(train_path+'*.PO.*'+extn)
train_pst_ds = prepare_dataset(list_train_pst_ds, BATCH_SIZE//4, train_pst_image_count, True)
dist_train_pst_ds = strategy.experimental_distribute_dataset(train_pst_ds)
train_pst_iter = iter(dist_train_pst_ds)

list_valid_non_ds = tf.data.Dataset.list_files(valid_path+'*.NM.*'+extn)
valid_non_ds = prepare_dataset(list_valid_non_ds, VALID_BATCH_SIZE//4, valid_non_image_count)
dist_valid_non_ds = strategy.experimental_distribute_dataset(valid_non_ds)
valid_non_iter = iter(dist_valid_non_ds)
list_valid_pre_ds = tf.data.Dataset.list_files(valid_path+'*.PR.*'+extn)
valid_pre_ds = prepare_dataset(list_valid_pre_ds, VALID_BATCH_SIZE//4, valid_pre_image_count)
dist_valid_pre_ds = strategy.experimental_distribute_dataset(valid_pre_ds)
valid_pre_iter = iter(dist_valid_pre_ds)
list_valid_ong_ds = tf.data.Dataset.list_files(valid_path+'*.OG.*'+extn)
valid_ong_ds = prepare_dataset(list_valid_ong_ds, VALID_BATCH_SIZE//4, valid_ong_image_count)
dist_valid_ong_ds = strategy.experimental_distribute_dataset(valid_ong_ds)
valid_ong_iter = iter(dist_valid_ong_ds)
list_valid_pst_ds = tf.data.Dataset.list_files(valid_path+'*.PO.*'+extn)
valid_pst_ds = prepare_dataset(list_valid_pst_ds, VALID_BATCH_SIZE//4, valid_pst_image_count)
dist_valid_pst_ds = strategy.experimental_distribute_dataset(valid_pst_ds)
valid_pst_iter = iter(dist_valid_pst_ds)


# Reference:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def get_model(model_url: str, res: int = IMAGE_SIZE[0], num_classes: int = 4) -> tf.keras.Model:
    inputs = tf.keras.Input((res, res, 3))
    hub_module = hub.KerasLayer(model_url, trainable=True)

    x = hub_module(inputs, training=False)
    y = keras.layers.Dense(256, activation="relu", name='y')(x)
    outputs = keras.layers.Dense(4, activation="softmax", name='out')(y)

    return tf.keras.Model(inputs, outputs)


get_model(MODEL_PATH).summary()


scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

with strategy.scope():
    model = get_model(MODEL_PATH)
    optimizer = keras.optimizers.SGD(scheduled_lrs)
    total_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

#@tf.function
def train_step(non, pre, ong, pst):
    tnx, tny = non
    tpx, tpy = pre
    tox, toy = ong
    tsx, tsy = pst
    images = tf.concat((tnx, tpx, tox, tsx), axis=0)
    labels = tf.concat((tny, tpy, toy, tsy), axis=0)

    #'''labels shoule be one_hot'''
    with tf.GradientTape() as tape:
        pred = model(images)
        loss = total_loss(labels, pred)
        mean_loss = tf.reduce_mean(loss)

    #Update gradients and optimize
    grads = tape.gradient(mean_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    #tf statistics tracking
    train_loss(mean_loss)
    train_acc(labels, pred)

#@tf.function
def val_step(non, pre, ong, pst):
    vnx, vny = non 
    vpx, vpy = pre 
    vox, voy = ong 
    vsx, vsy = pst 
    images = tf.concat((vnx, vpx, vox, vsx), axis=0)
    labels = tf.concat((vny, vpy, voy, vsy), axis=0)

    #'''labels should be one_hot'''
    pred = model(images, training=False)
    v_loss = total_loss(labels, pred)
    mean_v_loss = tf.reduce_mean(v_loss)

    #tf statistics tracking
    val_loss(mean_v_loss)
    val_acc(labels, pred)
    return pred

@tf.function
def dist_train_step(dist_non, dist_pre, dist_ong, dist_pst):
    per_replica_losses = strategy.run(train_step, args=(dist_non, dist_pre, dist_ong, dist_pst,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

@tf.function
def dist_val_step(dist_non, dist_pre, dist_ong, dist_pst):
    per_replica_losses = strategy.run(val_step, args=(dist_non, dist_pre, dist_ong, dist_pst,))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)


t_los = []
v_los = []
t_acc = []
v_acc = []
template = 'Epoch {}\nTrain Loss: {:.3g}\tTrain Accuracy: {:.3g}\nValid Loss: {:.3g}\tValid Accuracy: {:.3g}'

peak = [-1, 0.0, 10.0]

for epoch in range(0, EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    
    for step in range(0, STEPS_PER_EPOCH):
        dist_train_step(next(train_non_iter), next(train_pre_iter), next(train_ong_iter), next(train_pst_iter))
        
    for step in range(0, STEPS_PER_VALID_EPOCH):
        dist_val_step(next(valid_non_iter), next(valid_pre_iter), next(valid_ong_iter), next(valid_pst_iter))
        
    print(template.format(epoch, train_loss.result(), train_acc.result(), val_loss.result(), val_acc.result()))
    
    t_los.append(train_loss.result())
    v_los.append(val_loss.result())
    t_acc.append(train_acc.result())
    v_acc.append(val_acc.result())

    if val_loss.result() <= peak[2] and val_acc.result() >= peak[1]:
        peak[0] = epoch+1
        peak[1] = val_acc.result()
        peak[2] = val_loss.result()
        model.save_weights('./saved_model_Q_cut/checkpoint')
        print('Saved')
        

import matplotlib.pyplot as plt

plt.plot(t_los)
plt.plot(v_los, '--')
plt.show()
plt.savefig("swin-transfer-quad-cut-loss.png")
plt.close()


plt.plot(t_acc)
plt.plot(v_acc, '--')
plt.show()
plt.savefig("swin-transfer-quad-cut-acc.png")
