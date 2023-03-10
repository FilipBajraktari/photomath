#!/usr/bin/env python3

import os
import shutil
import math

old_path = '/home/filip/Desktop/informatika/Petnica_project_2020-21/inverted_images'

new_path = '/home/filip/Desktop/informatika/Petnica_project_2020-21/dataset'
new_path_train = os.path.join(new_path, 'training')
new_path_test = os.path.join(new_path, 'testing')
new_path_valid = os.path.join(new_path, 'validation')


if not os.path.exists(new_path):
    os.mkdir(new_path, mode=0o755)
    os.mkdir(new_path_train, mode=0o755)
    os.mkdir(new_path_test, mode=0o755)
    os.mkdir(new_path_valid, mode=0o755)

sample_index_train = 0
sample_index_test = 0
sample_index_valid = 0
labels_train = []
labels_test = []
labels_valid = []

sample_ratio = (0.7, 0.2, 0.1)


for root, dirs, files in os.walk(old_path):

    for character in dirs:
        print(character)

        for _, _, files in os.walk(os.path.join(root, character)):

            # files = files[:20]

            count_train = math.floor(sample_ratio[0] * len(files))
            count_test = math.floor(sample_ratio[1] * len(files))

            print(len(files), count_train, count_test)

        for sample in range(len(files)):

            # print('sample', sample)

            if sample <= count_train:
                shutil.copyfile(os.path.join(
                    root, character, files[sample]), os.path.join(new_path_train, '.'.join([str(sample_index_train), 'jpg'])))
                sample_index_train += 1
                labels_train.append(character)
                # print('train')

            elif sample < count_train + count_test:
                shutil.copyfile(os.path.join(
                    root, character, files[sample]), os.path.join(new_path_test, '.'.join([str(sample_index_test), 'jpg'])))
                sample_index_test += 1
                labels_test.append(character)
                # print('test')

            else:

                shutil.copyfile(os.path.join(
                    root, character, files[sample]), os.path.join(new_path_valid, '.'.join([str(sample_index_valid), 'jpg'])))
                sample_index_valid += 1
                labels_valid.append(character)
                # print('valid')


with open(os.path.join(new_path, 'labels_training.txt'), 'w') as labels_file:
    labels_file.write('\n'.join(labels_train))

with open(os.path.join(new_path, 'labels_testing.txt'), 'w') as labels_file:
    labels_file.write('\n'.join(labels_test))

with open(os.path.join(new_path, 'labels_validation.txt'), 'w') as labels_file:
    labels_file.write('\n'.join(labels_valid))
