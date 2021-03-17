#Prediction
import os
import time
import cv2
import csv
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from skimage import color

#Runtime measurements
start_time = time.time()

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--GPU_num', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--model_ckpt', type=str)
parser.add_argument('--test_dir', type=str)
parser.add_argument('--test_num', type=int)
parser.add_argument('--output_csv', type=str)
parser.add_argument('--roc_im_name', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

#Which GPU to use
GPU_number = args.GPU_num
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_number

#Size of your batch
batch_size = args.batch_size

#Your Saved Models path
model_ckpt_Path = args.model_ckpt

#Your testing dataset path 
test_dir = args.test_dir

#Number of test images
test_num = args.test_num

#Output CSV file name
output_csv = args.output_csv

#ROC plot image name
roc_im_name = args.roc_im_name

#Evaluation or prediction mode
mode = args.mode

#Plot ROC
def plot_roc_curve(fpr, tpr, file):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(file+'.jpeg')
    plt.close()


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


#Get validation dataset
def get_batched_test_dir(batch_size, filenames):
    files = tf.data.Dataset.list_files(filenames)
    dataset = tf.data.TFRecordDataset(filenames=files)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_testing_dataset():
  return get_batched_test_dir(batch_size, test_dir)

#Load a model
model = tf.keras.models.load_model(model_ckpt_Path)
#Metrics for evaluate
metrics = ["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
#Compile your model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=metrics)
#Model summary
model.summary()

if mode == 'evaluate':
    #Evaluate your model on test dataset 
    output_dict = model.evaluate(get_testing_dataset(), steps=test_num//batch_size, use_multiprocessing=True, workers=tf.data.experimental.AUTOTUNE, return_dict=True)
    print(output_dict)

    #Write results to csv
    try:
        with open(output_csv, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(output_dict.keys())
            writer.writerow(output_dict.values())
    except IOError:
        print("I/O error")
    
elif mode == 'predict':
    #List of true labels
    labels = []
    #List of predictions
    preds = []
    #Loop over dataset
    for image, label in get_testing_dataset():
        #Append true labels
        labels.extend(np.argmax(label, axis=1))
        #Predict
        y_pred = model(image, training=False)
        #Since it is softmax, take the second element
        #and append it to predictions list
        preds.extend(y_pred.numpy()[:, 1])

    #At this point you have list of true labels and predicted labels, use it however you want, sky is the limit    
    
    #Threshold your prediction values 
    for_class_report = [1 if num_ > threshold else 0 for num_ in np.asarray(preds)]

    #Classification report
    print(classification_report(labels, for_class))

    #AUC score
    print('AUC: %.4f' % roc_auc_score(labels, preds))

    #Plot ROC curve 
    fpr, tpr, thresholds = roc_curve(labels, preds)
    plot_roc_curve(fpr, tpr, roc_im_name)

#Timer
print("--- %s seconds ---" % (time.time() - start_time))
