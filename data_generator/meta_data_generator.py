import numpy as np
import os
import random

from tqdm import trange
from tensorflow.python.platform import flags
from utils.misc import get_images

FLAGS = flags.FLAGS

class MetaDataGenerator(object):

    def __init__(self, num_samples_per_class):
        self.num_samples_per_class = num_samples_per_class
        self.num_unlabeled_samples = FLAGS.nb_ul_samples
        self.num_classes = FLAGS.way_num
        metatrain_labeled_folder = FLAGS.data_path + '/data/' + FLAGS.dataset + '/labeled_train'
        metatest_labeled_folder = FLAGS.data_path + '/data/' + FLAGS.dataset + '/labeled_test'
        metaval_labeled_folder = FLAGS.data_path + '/data/' + FLAGS.dataset + '/labeled_val'
        self.metatrain_unlabeled_folder = FLAGS.data_path + '/data/' + FLAGS.dataset + '/unlabeled_train'
        self.metatest_unlabeled_folder = FLAGS.data_path + '/data/' + FLAGS.dataset + '/unlabeled_test'
        self.metaval_unlabeled_folder = FLAGS.data_path + '/data/' + FLAGS.dataset + '/unlabeled_val'

        filename_dir = FLAGS.logdir_base + 'filenames_and_labels/'
        if not os.path.exists(filename_dir):
            os.mkdir(filename_dir)

        if FLAGS.dataset == 'miniImagenet':
            data_name = 'mini_'
        else:
            data_name = 'tiered_'

        self.this_setting_filename_dir = filename_dir + data_name + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.way_num) + 'way/'
        if not os.path.exists(self.this_setting_filename_dir):
            os.mkdir(self.this_setting_filename_dir)

        metatrain_folders = [(os.path.join(metatrain_labeled_folder, label), label) \
                             for label in os.listdir(metatrain_labeled_folder) \
                             if os.path.isdir(os.path.join(metatrain_labeled_folder, label)) \
                             ]
        metatest_folders = [(os.path.join(metatest_labeled_folder, label), label) \
                            for label in os.listdir(metatest_labeled_folder) \
                            if os.path.isdir(os.path.join(metatest_labeled_folder, label)) \
                            ]
        metaval_folders = [(os.path.join(metaval_labeled_folder, label), label) \
                           for label in os.listdir(metaval_labeled_folder) \
                           if os.path.isdir(os.path.join(metaval_labeled_folder, label)) \
                           ]

        self.metatrain_character_folders = metatrain_folders
        self.metatest_character_folders = metatest_folders
        self.metaval_character_folders = metaval_folders


    def make_data_list(self, data_type='train'):
        if data_type == 'train':
            folders = self.metatrain_character_folders
            num_total_batches = 80000
        elif data_type == 'test':
            folders = self.metatest_character_folders
            num_total_batches = 600
        elif data_type == 'val':
            folders = self.metaval_character_folders
            num_total_batches = 600
        else:
            print('Please check data list type')

        if not os.path.exists(self.this_setting_filename_dir + '/' + data_type + '_filenames.npy'):
            print('Generating ' + data_type + ' filenames')
            all_filenames = []
            label_names = []
            for _ in trange(num_total_batches):
                sampled_character_folders = random.sample(folders, self.num_classes)
                random.shuffle(sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                                nb_samples=self.num_samples_per_class, shuffle=False)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1].replace('/home/lxz/python_code','') for li in labels_and_images]
                names = [l[1] for l in sampled_character_folders]
                all_filenames.extend(filenames)
                label_names.append(names)

            np.save(self.this_setting_filename_dir + '/' + data_type + '_labels.npy', labels)
            np.save(self.this_setting_filename_dir + '/' + data_type + '_filenames.npy', all_filenames)
            np.save(self.this_setting_filename_dir + '/' + data_type + '_labelnames.npy', label_names)
            print('The ' + data_type + ' filename and label lists are saved')
        else:
            print('The ' + data_type + ' filename and label lists have already been created')


    def make_unlabeled_data_list(self, data_type='train'):
        if data_type == 'train':
            un_folder = self.metatrain_unlabeled_folder
            num_total_batches = 80000
        elif data_type == 'val':
            un_folder = self.metaval_unlabeled_folder
            num_total_batches = 600
        else:
            print('Please check data list type')

        label_names = np.load(self.this_setting_filename_dir + '/' + data_type + '_labelnames.npy')

        unlabeled_dir = self.this_setting_filename_dir + '/unlabeled_samples_for_' + data_type
        if not os.path.exists(unlabeled_dir):
            os.mkdir(unlabeled_dir)

        if os.path.exists(self.this_setting_filename_dir + '/' + data_type + '_filenames.npy') and not \
                os.path.exists(unlabeled_dir + '/' + data_type + '_' + str(
                    self.num_unlabeled_samples) + '_un_filenames.npy'):
            print('Generating ' + data_type + ' unlabeled filenames')
            all_unlabeled_filenames = []
            all_unlabeled_labels = []

            for i in trange(num_total_batches):
                names_for_task = label_names[i]

                sampled_character_folders = [(os.path.join(un_folder, name), name) for name in names_for_task]

                labels_and_images = get_images(sampled_character_folders, range(self.num_classes),
                                               nb_samples=self.num_unlabeled_samples, shuffle=False)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1].replace('/home/lxz/python_code','') for li in labels_and_images]
                all_unlabeled_filenames.extend(filenames)
                all_unlabeled_labels.extend(labels)

            np.save(unlabeled_dir + '/' + data_type + '_' + str(self.num_unlabeled_samples) + '_un_labels.npy',
                    all_unlabeled_labels)
            np.save(unlabeled_dir + '/' + data_type + '_' + str(self.num_unlabeled_samples) + '_un_filenames.npy',
                all_unlabeled_filenames)

            print('The ' + data_type + ' unlabeded filename and label lists are saved')
        else:
            print('The ' + data_type + ' unlabeled filename and label lists have already been created')


    def make_unlabeled_test_data_list(self):
        un_folder = self.metatest_unlabeled_folder
        num_total_batches = 600

        label_names = np.load(self.this_setting_filename_dir + '/test_labelnames.npy')

        unlabeled_dir = self.this_setting_filename_dir + '/unlabeled_samples_for_test'
        if not os.path.exists(unlabeled_dir):
            os.mkdir(unlabeled_dir)

        if os.path.exists(self.this_setting_filename_dir + '/test_filenames.npy') and not \
                os.path.exists(unlabeled_dir + '/test_0_un_filenames.npy'):
            print('Generating test unlabeled filenames')

            file_dict = {}
            label_dict = {}

            for m in range(10):
                file_dict[m] = []
                label_dict[m] = []

            for i in trange(num_total_batches):

                names_for_task = label_names[i]

                sampled_character_folders = [(os.path.join(un_folder, name), name) for name in names_for_task]

                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=100,
                                               shuffle=False)
                for j in range(10):
                    sub_labels_and_images = []
                    for k in range(len(names_for_task)):
                        sub_labels_and_images.extend(labels_and_images[j * 10 + k * 100: (j + 1) * 10 + k * 100])
                    labels = [li[0] for li in sub_labels_and_images]
                    filenames = [li[1].replace('/home/lxz/python_code','') for li in sub_labels_and_images]
                    file_dict[j].extend(filenames)
                    label_dict[j].extend(labels)

                # make sure the above isn't randomized order
            for n in range(10):
                np.save(unlabeled_dir + '/test_' + str(n) + '_un_labels.npy', label_dict[n])
                np.save(unlabeled_dir + '/test_' + str(n) + '_un_filenames.npy', file_dict[n])

            print('The test unlabeded filename and label lists are saved')
        else:
            print('The test unlabeled filename and label lists have already been created')


    def make_distractors_list(self, data_type='train'):
        if data_type == 'train':
            un_folder = self.metatrain_unlabeled_folder
            num_total_batches = 80000
        elif data_type == 'val':
            un_folder = self.metaval_unlabeled_folder
            num_total_batches = 600
        else:
            print('Please check data list type')

        label_names = np.load(self.this_setting_filename_dir + '/' + data_type + '_labelnames.npy')
        unlabeled_dir = self.this_setting_filename_dir + '/unlabeled_samples_for_' + data_type + '/distracting_file'
        if not os.path.exists(unlabeled_dir):
            os.mkdir(unlabeled_dir)

        if os.path.exists(self.this_setting_filename_dir + '/' + data_type + '_labelnames.npy') and not \
                os.path.exists(unlabeled_dir + '/' + data_type + '_' + str(self.num_unlabeled_samples) + '_dis_filenames.npy'):
            print('Generating ' + data_type + ' distracting unlabeled filenames')

            all_distractors_filenames = []
            all_distractors_labels = []

            base_list = os.listdir(un_folder)

            def return_dis_labels(list_1, list_2, list_3):
                item = random.sample(list_1, 1)[0]
                if (item not in list_2) and (item not in list_3):
                    return item
                else:
                    return return_dis_labels(list_1, list_2, list_3)

            for i in trange(num_total_batches):

                task_label_list = label_names[i]
                task_distractors_list = []

                for j in range(FLAGS.num_dis):
                    distractors_label = return_dis_labels(base_list, task_label_list, task_distractors_list)
                    task_distractors_list.append(distractors_label)

                sampled_distractors_folders = [(os.path.join(un_folder, name), name) for name in task_distractors_list]
                labels_and_images = get_images(sampled_distractors_folders, range(FLAGS.num_dis),
                                               nb_samples=self.num_unlabeled_samples, shuffle=False)
                labels = [-1 for li in labels_and_images]
                filenames = [li[1].replace('/home/lxz/python_code','') for li in labels_and_images]
                all_distractors_filenames.extend(filenames)
                all_distractors_labels.extend(labels)

            np.save(unlabeled_dir + '/' + data_type + '_' + str(self.num_unlabeled_samples) + '_dis_filenames.npy',
                    all_distractors_filenames)

            print('The ' + data_type + ' distractors filename lists are saved')
        else:
            print('The ' + data_type + ' distractors filename lists have already been created')


    def make_test_distractors_list(self, total_samples_per_class=100):
        un_folder = self.metatest_unlabeled_folder
        num_total_batches = 600

        label_names = np.load(self.this_setting_filename_dir + '/test_labelnames.npy')
        unlabeled_dir = self.this_setting_filename_dir + '/unlabeled_samples_for_test/distracting_file'
        if not os.path.exists(unlabeled_dir):
            os.mkdir(unlabeled_dir)

        if os.path.exists(self.this_setting_filename_dir + '/test_labelnames.npy') and not \
                os.path.exists(unlabeled_dir + '/test_0_dis_filenames.npy'):
            print('Generating test distracting unlabeled filenames')

            base_list = os.listdir(un_folder)

            dis_dict = {}

            for n in range(7):
                dis_dict[n] = []

            def return_dis_labels(list_1, list_2, list_3):
                item = random.sample(list_1, 1)[0]
                if (item not in list_2) and (item not in list_3):
                    return item
                else:
                    return return_dis_labels(list_1, list_2, list_3)

            for i in trange(num_total_batches):

                task_label_list = label_names[i]
                task_distractors_list = []

                for j in range(7):
                    distractors_label = return_dis_labels(base_list, task_label_list, task_distractors_list)
                    task_distractors_list.append(distractors_label)
                    sampled_distractors_folders = [(os.path.join(un_folder, distractors_label), distractors_label)]
                    labels_and_images = get_images(sampled_distractors_folders, range(1),
                                                   nb_samples=total_samples_per_class, shuffle=False)
                    filenames = [li[1].replace('/home/lxz/python_code','') for li in labels_and_images]
                    dis_dict[j].extend(filenames)

            for k in range(7):
                np.save(unlabeled_dir + '/test_' + str(k + 1) + '_dis_filenames.npy', dis_dict[k])

            print('The test distractors filename lists are saved')
        else:
            print('The test distractors filename lists have already been created')
