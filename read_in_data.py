__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
#DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'

def read_prediction_set(data_dir):
    if not gfile.Exists(data_dir):
        print("Image directory '" + data_dir + "' not found.")
        return None
    file_list = []
    image_list = []
    for ipath in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, ipath)):
            continue
        file_glob = os.path.join(data_dir, ipath, '*.' + 'png')
        print(file_glob)
        file_list.extend(glob.glob(file_glob))

    if not file_list:
        print('No files found')
    else:
        image_list = [{'image': f, 'filename': (os.path.splitext(f.split("/")[-2])[0] + '_' + os.path.splitext(f.split("/")[-1])[0])} for f in file_list]
    print ('No. of files: %d' % len(image_list))
    return image_list

def read_dataset(data_dir, pwc=False, test=False):
    if not pwc:
        pickle_filename = "dataset.pickle"
    else:
        pickle_filename = "dataset_pwc.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    print(pickle_filepath)
    if not os.path.exists(pickle_filepath):
        if not pwc:
            result = create_image_lists(data_dir, test)
        else:
            result = create_image_lists_pwc(data_dir, test)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        if not test:
            validation_records = result['validation']
        else:
            validation_records = None
        del result

    return training_records, validation_records


def create_image_lists(image_dir, test=False):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    if not test:
        directories = ['training', 'validation']
    else:
        directories = ['training']
    image_list = {}

    for directory in directories:
        print('Here')
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
#                 print(f)
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, 'label_' + filename + '.png')
#                 print(annotation_file)
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print(annotation_file)
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list


# Read annotation for patch-wise classification
def create_image_lists_pwc(image_dir, test=False):
    print(image_dir)
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    if not test:
        directories = ['training', 'validation']
    else:
        directories = ['training']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        print(os.path.join(image_dir, "images", directory))
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
#                 print(f)
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations_patch_classify", directory, 'label_' + filename + '.png')
#                 print(annotation_file)
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print(annotation_file)
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
