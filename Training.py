import os
import tensorflow as tf
import itertools
import argparse
import sys
import numpy as np

#Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--GPU_num', type=str, default='0')
parser.add_argument('--train_num', type=int)
parser.add_argument('--valid_num', type=int)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--train_dir', type=str)
parser.add_argument('--valid_dir', type=str)
parser.add_argument('--ckpt_name', type=str)
parser.add_argument('--csv_log_name', type=str)
parser.add_argument('--tensorboard_logs', type=str)
parser.add_argument('--MP', type=str, default="Yes")
args = parser.parse_args()

#Which GPU to use
GPU_number = args.GPU_num
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_number
#Size of your batch
batch_size = args.batch_size
#Number of images in train dataset
train_num = args.train_num
#Number of images in valid dataset
valid_num = args.valid_num
#Image size
size = args.size
#Epochs
EPOCHS = args.epochs
#Training directory along with glob file pattern
train_dir = args.train_dir
#Valid directory along with glob file pattern
valid_dir = args.valid_dir
#Chekpoint names with _epoch#.hdf5 in the end
ckpt_name = args.ckpt_name
#CSV log filename
csv_log_name = args.csv_log_name
#Tensorboard logs location
tensorboard_logs = args.tensorboard_logs
#Mixed precision 
MP = args.MP

#Multi-GPU
strategy = tf.distribute.MirroredStrategy()
GPUnum = strategy.num_replicas_in_sync
print ('Number of devices: {}'.format(GPUnum))
#Multiple batch size by GPU num
batch_size = batch_size*GPUnum

#Mixed precision
if MP == "Yes":
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('Compute dtype: %s' % tf.keras.mixed_precision.global_policy())


#Getting steps
TRAINING_STEPS_PER_EPOCH = train_num//batch_size
VALIDATION_STEPS_PER_EPOCH = valid_num//batch_size

'''Custom callback'''
#Terminate on NaN callback, works for training as well as validation datasets
class TerminateOnNaN(tf.keras.callbacks.Callback):
 
  #Terminate if NaN encountered in training
  def on_train_batch_end(self, batch, logs=None):
    logs = logs or {}
    loss = logs.get('loss')
    if loss is not None:
      loss = np.array(loss)
      if np.isnan(loss) or np.isinf(loss):
        print('Batch %d: Invalid loss in training, terminating training' % (batch))
        self.model.stop_training = True

  #Terminate if NaN encountered in validation
  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    loss = logs.get('val_loss')
    if loss is not None:
      loss = np.array(loss)
      if np.isnan(loss) or np.isinf(loss):
        print('Epoch %d: Invalid loss in validation, terminating training' % (epoch))
        self.model.stop_training = True


#Extract data from TFRecord
def read_tfrecord(record):
    keys_to_features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/class/label':     tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string)
    }
    sample =  tf.io.parse_single_example(record, keys_to_features)
    image = tf.io.decode_jpeg(sample['image/encoded'], channels=3)
    label = tf.one_hot(sample['image/class/label'], 2)
    image = tf.cast(image, tf.float32) / 255
    return image, label


#Get training dataset
def get_batched_train_dataset(BATCH_SIZE, filenames):
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

#Get validation dataset
def get_batched_valid_dataset(BATCH_SIZE, filenames):
    files = tf.data.Dataset.list_files(filenames)
    dataset = tf.data.TFRecordDataset(filenames=files)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

def get_training_dataset():
  return get_batched_train_dataset(batch_size, train_dir)

def get_validation_dataset():
  return get_batched_valid_dataset(batch_size, valid_dir)

#get_training_dataset()

#Train
with strategy.scope():
    efficientNet = tf.keras.applications.Xception(include_top=False, weights=None, input_shape=(size,size,3))
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(efficientNet.output)
    x = tf.keras.layers.Dense(2)(x)
    end = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)
    print('Outputs dtype: %s' % end.dtype.name)
    model = tf.keras.Model(inputs=efficientNet.input, outputs=end)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    my_callbacks = [
        tf.keras.callbacks.CSVLogger(csv_log_name, separator=',', append=False),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_graph=True, update_freq=500, profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint('%s' % ckpt_name, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
        TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=30, verbose=0, mode='min')
    ]

model.summary()

model.fit(get_training_dataset(), steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
        validation_data=get_validation_dataset(), validation_steps=VALIDATION_STEPS_PER_EPOCH,
        epochs=EPOCHS, verbose=1, callbacks=my_callbacks)
