from datetime import datetime
from glob import glob
import os
import random
import sys
import time
import threading
import argparse
import numpy as np
import tensorflow as tf


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label, text, height, width, channels):

  feature = {
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))
  }

  # Create an example protocol buffer
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example


def process_single_image(filename):
  # Read the image file.
  image_data = open(filename, 'rb').read()

  # Decode the RGB JPEG.
  image = tf.image.decode_jpeg(image_data)
    
  # Check your images
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  channels = image.shape[2]
  assert channels == 3

  return image_data, height, width, channels


def process_by_batch_train(thread_index, ranges, dataset_label, filenames,
                               texts, labels, num_shards, outdir, tile_size):

  num_threads = len(ranges)
  assert not num_shards % num_threads
  #Shards per batch
  num_shards_per_batch = int(num_shards / num_threads)

  #Splitting every shard into individual shards
  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
    
  #Files that will be processed by thread
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
  image_counter = 0

  #Loop over shards per batch
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    #print(shard)
    output_file = os.path.join(outdir, '%s-%.5d-of-%.5d.tfrecord' % (dataset_label, shard, num_shards))
    writer = tf.io.TFRecordWriter(output_file)

    shard_counter = 0
    #Indeces of files that go into every shard
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    #print(files_in_shard)
    for i in files_in_shard:
      #Get your filename
      filename = filenames[i]
      #Numerical label value
      label = labels[i]
      #text label value
      text = texts[i]
      #Image data to write into TFRecord
      image_buffer, height, width, channels = process_single_image(filename)
      #Image dimension checker
      if (height != tile_size or width != tile_size):
        continue
      #Write image to TFRecord
      else:
        example = _convert_to_example(filename, image_buffer, label,
                                      text, height, width, channels)
        writer.write(example.SerializeToString())
        shard_counter += 1
        image_counter += 1

        #if not image_counter % 1000:
        #  print('%s [thread %d]: Processed %d of %d images in thread batch.' %
        #        (datetime.now(), thread_index, image_counter, num_files_in_thread))
        #  sys.stdout.flush()

    writer.close()
    #print('%s [thread %d]: Wrote %d images to %s' %
    #      (datetime.now(), thread_index, shard_counter, output_file))
    #sys.stdout.flush()
    shard_counter = 0

  #print('%s [thread %d]: Wrote %d images to %d shards.' %
  #      (datetime.now(), thread_index, image_counter, num_files_in_thread))
  #sys.stdout.flush()


def process_by_batch_valid_test(thread_index, ranges, dataset_label, filenames,
                               texts, labels, outdir, tile_size):
  
  rootname = ''
  slide_counter = 0
  counter = 0
  
  for i in range(len(filenames)):
    #print(i)
    #Get your filename
    filename = filenames[i]
    #Numerical label value
    label = labels[i]
    #text label value
    text = texts[i]
    #Image data to write into TFRecord
    image_buffer, height, width, channels = process_single_image(filename)
    #Image dimension checker
    if (height != tile_size or width != tile_size):
      continue
    #Write image to TFRecord
    else:
      #Assemble back label and svs filename
      next_file_root = '_'.join(os.path.basename(filename).split('_')[:2])
      
      if rootname == next_file_root:
        # New tile of same slide
        tile_counter += 1
        
      else:
        # New slide
        rootname = next_file_root
        slide_counter += 1
        tile_counter = 0
        counter += 1
        output_file = os.path.join(outdir, '%s_%s.tfrecord' % (rootname, label))
        writer = tf.io.TFRecordWriter(output_file)

      example = _convert_to_example(filename, image_buffer, label,
                                      text, height, width, channels)
      writer.write(example.SerializeToString())
  
      #if not counter % 100:
      #  print('%s [thread %d]: Processed tile %d of slide %d.' %
      #        (datetime.now(), thread_index, tile_counter, slide_counter))
      #  sys.stdout.flush()

  writer.close()
  #print('%s [thread %d]: Wrote %d images to %s' %
  #      (datetime.now(), thread_index, slide_counter, output_file))
  #sys.stdout.flush()


def process_image_files(dataset_label, filenames, texts, labels, num_shards, outdir, tile_size, threads):

  #Check your lengths
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), threads+1).astype(np.int)
    
  ranges = []
  #Loop over split image groups, making ranges between the two points
  for i in range(len(spacing)-1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  #Threading
  threads = []

  #Launch your threads
  print("***************************Started processing %s dataset***************************" % (dataset_label))
  if dataset_label == 'train':
    for thread_index in range(len(ranges)):
      args = (thread_index, ranges, dataset_label, filenames,
              texts, labels, num_shards, outdir, tile_size)
      t = threading.Thread(target=process_by_batch_train, args=args)
      t.start()
      threads.append(t)
  else:
    for thread_index in range(len(ranges)):
      args = (thread_index, ranges, dataset_label, filenames,
              texts, labels, outdir, tile_size)
      t = threading.Thread(target=process_by_batch_valid_test, args=args)
      t.start()
      threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished dealing with all %d images in %s data set.' %
        (datetime.now(), len(filenames), dataset_label))
  print('***************************Processing %s dataset is done***************************' % (dataset_label))

  sys.stdout.flush()

def array_creator(data_dir, label, dataset_label, formatting, amount, label_index, oversampling_rate):
  #Filenames
  filenames = []
  #String names of labels
  texts = []
  #Numerical values of labels
  labels = []

  #Match all .jpeg
  matching_files = glob(os.path.join(data_dir, label, dataset_label + '*.'+formatting))
  print("%s %s images: %d" % (dataset_label, label, amount))
  matching_files.sort()
  #Add filenames
  filenames.extend(matching_files * oversampling_rate)
  #Add string text class labels
  texts.extend([label] * amount * oversampling_rate)
  #Add numeric class labels
  labels.extend([label_index] * amount * oversampling_rate)
  return filenames, texts, labels
  

#Structure your files
def organizer(dataset_label, data_dir, formatting, oversampling):

  #Unique class labels
  unique_labels = []
  #Filenames
  filenames = []
  #String names of labels
  texts = []
  #Numerical values of labels
  labels = []
  tiles_per_class = {}
  # Leave label index 0 empty as a background class.
  label_index = 0

  #Get subdirectories of your directory
  for item in os.listdir(data_dir):
    #Get you labels
    if os.path.isdir(os.path.join(data_dir, item)):
    	unique_labels.append(os.path.join(item))

  unique_labels.sort()
  print("Given labels: %s" % unique_labels)
  
  if len(unique_labels) == 2:
    #Construct the list of JPEG files and labels.
    #Loop over labels
    for label in unique_labels:
        print(label)
        #Match all .jpeg
        tiles_per_class[label] = (len(glob(os.path.join(data_dir, label, dataset_label + '*.'+formatting))))
    if dataset_label == 'train':
      if oversampling == 'Yes':
        #See whats the numeric difference between the class samples and sample with replacement 
        oversampling_rate = max(tiles_per_class.values())//min(tiles_per_class.values())

        for idx, label in enumerate(unique_labels):
          #print(tiles_per_class[label])
          if label == min(tiles_per_class, key=tiles_per_class.get):
            file_tmp, text_tmp, label_tmp = array_creator(data_dir, label, dataset_label, formatting, tiles_per_class[label], idx, oversampling_rate)
            filenames.extend(file_tmp)
            texts.extend(text_tmp)
            labels.extend(label_tmp)
            
          elif label == max(tiles_per_class, key=tiles_per_class.get):
            file_tmp, text_tmp, label_tmp = array_creator(data_dir, label, dataset_label, formatting, tiles_per_class[label], idx, 1)
            filenames.extend(file_tmp)
            texts.extend(text_tmp)
            labels.extend(label_tmp)
      else:
        for idx, label in enumerate(unique_labels):
          file_tmp, text_tmp, label_tmp = array_creator(data_dir, label, dataset_label, formatting, tiles_per_class[label], idx, 1)
          filenames.extend(file_tmp)
          texts.extend(text_tmp)
          labels.extend(label_tmp)

      #Shuffle the ordering of all image files in order to guarantee
      #random ordering of the images with respect to label in the
      #saved TFRecord files. Make the randomization repeatable.
      shuffled_index = list(range(len(filenames)))
      random.seed(12345)
      random.shuffle(shuffled_index)

      #Reshuffle
      filenames = [filenames[i] for i in shuffled_index]
      texts = [texts[i] for i in shuffled_index]
      labels = [labels[i] for i in shuffled_index]

    elif dataset_label == 'valid' or dataset_label == 'test':
      #Construct the list of JPEG files and labels.
      #Loop over labels
      for idx, label in enumerate(unique_labels):
        file_tmp, text_tmp, label_tmp = array_creator(data_dir, label, dataset_label, formatting, tiles_per_class[label], idx, 1)
        filenames.extend(file_tmp)
        texts.extend(text_tmp)
        labels.extend(label_tmp)

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
  
  #Print your contents
  #for i in range(len(filenames)):
  #  print(os.path.basename(filenames[i]), "|", texts[i], "|", labels[i])

  return filenames, texts, labels


def main():

  #Getting arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--sort_dir', type=str)
  parser.add_argument('--outdir', type=str)
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--num_files', type=int, default=1024)
  parser.add_argument('--size', type=int, default=256)
  parser.add_argument('--format', type=str, default="jpeg")
  parser.add_argument('--oversampling', type=str, default="No")
  args = parser.parse_args()

  #Casting arguments
  #Sorted directory
  sort_dir = args.sort_dir
  #Output directory
  outdir = args.outdir
  #Number of threads 
  threads = args.threads
  #Must be divisible by number of threads
  num_files = args.num_files
  #Tile size
  tile_size = args.size
  #Image file extension 
  formatting = args.format
  #Whether to oversample minority class
  oversampling = args.oversampling

  assert not num_files % threads, ('Numer of output files should be divisible by number of threads')

  #Start timer
  start = time.time()

  train_filenames, train_texts, train_labels = organizer('train', sort_dir, formatting, oversampling)
  print('***************************Train organizing is done***************************')
  #Start TFRecord creation
  if len(train_filenames)>0:
    process_image_files('train', train_filenames, train_texts, train_labels, num_files, outdir, tile_size, threads)

  valid_filenames, valid_texts, valid_labels = organizer('valid', sort_dir, formatting, 'No')
  print('***************************Valid organizing is done***************************')
  #Start TFRecord creation
  if len(valid_filenames)>0:
    process_image_files('valid', valid_filenames, valid_texts, valid_labels, num_files, outdir, tile_size, 1)
  
  test_filenames, test_texts, test_labels = organizer('test', sort_dir, formatting, 'No')
  print('***************************Test organizing is done***************************')
  #Start TFRecord creation
  if len(test_filenames)>0:
    process_image_files('test', test_filenames, test_texts, test_labels, num_files, outdir, tile_size, 1)
  
  end = time.time()
  print(f'Execution time {end - start:.2f}s')


main()