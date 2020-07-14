import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import cv2
import pdb

from tqdm import trange
from model_LTTL.meta_model_test import MetaModel
from tensorflow.python.platform import flags
from misc import process_batch_new, process_un_batch_new, process_dis_batch_recur, process_un_batch_recur

FLAGS = flags.FLAGS

class MetaTrainer:
    def __init__(self):
        print('Building test model')
        self.model = MetaModel()
        if FLAGS.use_distractors:
            self.model.construct_model_test_with_distractors(prefix='metatest_')
        else:
            self.model.construct_model_test(prefix='metatest')
        self.unfiles_num = FLAGS.unfiles_num
        self.model.summ_op = tf.summary.merge_all()
        print('Finish building test model')

        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

        exp_string = FLAGS.exp_string

        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        if FLAGS.test_iter == 0:
            print('Loading pretrain weights')
            if FLAGS.meta_finetune:
                weights_path = FLAGS.finetune_weights_path
                weights = np.load(weights_path + '/weights_15000.npy').tolist()
                ss_weights = np.load(
                    weights_path + '/ss_weights_15000.npy').tolist()
                fc_weights = np.load(
                    weights_path + '/fc_weights_15000.npy').tolist()
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in ss_weights.keys():
                    self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
                for key in fc_weights.keys():
                    self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
                print('Finetune weights loaded')
            else:
                weights_path = FLAGS.pretrain_weights_path
                weights = np.load(
                    os.path.join(weights_path, "weights_{}.npy".format(FLAGS.pretrain_iterations)), allow_pickle=True).tolist()
                bais_list = [bais_item for bais_item in weights.keys() if '_bias' in bais_item]
                fc_weights = np.load(os.path.join(weights_path, "fc_weights_"+str(FLAGS.shot_num)+"shot_15000.npy"), allow_pickle=True).tolist()
                for bais_key in bais_list:
                    self.sess.run(tf.assign(self.model.ss_weights[bais_key], weights[bais_key]))
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in fc_weights.keys():
                    self.sess.run(tf.assign(self.model.fc_weights_pl[key], fc_weights[key]))
                print('Pretrain weights loaded')
        else:
            weights = np.load(FLAGS.logdir + '/' + exp_string + '/weights_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True).tolist()
            weights_path = FLAGS.pretrain_weights_path
            fc_weights_pl = np.load(os.path.join(weights_path, "fc_weights_"+str(FLAGS.shot_num)+"shot.npy"), allow_pickle=True).tolist()
            ss_weights = np.load(
                FLAGS.logdir + '/' + exp_string + '/ss_weights_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True).tolist()
            fc_weights1 = np.load(
                FLAGS.logdir + '/' + exp_string + '/fc_weights1_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True).tolist()
            fc_weights2 = np.load(
                FLAGS.logdir + '/' + exp_string + '/fc_weights2_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True).tolist()
            swn_weights = np.load(
                FLAGS.logdir + '/' + exp_string + '/swn_weights_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True).tolist()

            for key in weights.keys():
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
            for key in ss_weights.keys():
                self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
            for key in fc_weights_pl.keys():
                self.sess.run(tf.assign(self.model.fc_weights_pl[key], fc_weights_pl[key]))
            for key in fc_weights1.keys():
                self.sess.run(tf.assign(self.model.fc_weights1[key], fc_weights1[key]))
            for key in fc_weights2.keys():
                self.sess.run(tf.assign(self.model.fc_weights2[key], fc_weights2[key]))
            for key in swn_weights.keys():
                self.sess.run(tf.assign(self.model.swn_weights[key], swn_weights[key]))

            print('Weights loaded')
            print('Test iter: ' + str(FLAGS.test_iter))

        if FLAGS.use_distractors:
            self.test_with_d()
        else:
            self.test()


    def test(self):
        NUM_TEST_POINTS = 6000
        np.random.seed(1)
        metaval_accuracies = []
        recur_accuracies = []
        pl_accruacies = []
        num_samples_per_class = FLAGS.shot_num * 2
        task_num = FLAGS.way_num * num_samples_per_class
        task_un_num = 10 * FLAGS.way_num
        half_num_samples = FLAGS.shot_num
        dim_input = FLAGS.img_size * FLAGS.img_size * 3

        ######
        filename_dir = FLAGS.logdir_base + '/' + 'filenames_and_labels_6000/'
        this_setting_filename_dir = filename_dir + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.way_num) + 'way/'
        all_filenames = np.load(this_setting_filename_dir + 'test_filenames.npy').tolist()
        labels = np.load(this_setting_filename_dir + 'test_labels.npy').tolist()

        un_file_dict = {}
        un_label_dict = {}
        ######
        un_new_folder = this_setting_filename_dir + 'unlabeled_samples_for_test'

        for f in range(self.unfiles_num):
            un_file_dict[f] = np.load(un_new_folder + '/' + 'test_' + str(f) + '_un_filenames.npy').tolist()
            un_label_dict[f] = np.load(un_new_folder + '/' + 'test_' + str(f) + '_un_labels.npy').tolist()

        for test_idx in trange(NUM_TEST_POINTS):

            this_task_filenames = all_filenames[test_idx * task_num:(test_idx + 1) * task_num]

            this_task_un_filenames = []
            this_task_un_labels = []

            for j in range(self.unfiles_num):
                this_task_un_filenames.extend(un_file_dict[j][test_idx * task_un_num:(test_idx + 1) * task_un_num])
                this_task_un_labels.extend(un_label_dict[j][test_idx * task_un_num:(test_idx + 1) * task_un_num])

            this_task_tr_filenames = []
            this_task_tr_labels = []
            this_task_te_filenames = []
            this_task_te_labels = []
            for class_k in range(FLAGS.way_num):
                this_class_filenames = this_task_filenames[
                                       class_k * num_samples_per_class:(class_k + 1) * num_samples_per_class]
                this_class_label = labels[class_k * num_samples_per_class:(class_k + 1) * num_samples_per_class]
                this_task_tr_filenames += this_class_filenames[0:half_num_samples]
                this_task_tr_labels += this_class_label[0:half_num_samples]
                this_task_te_filenames += this_class_filenames[half_num_samples:]
                this_task_te_labels += this_class_label[half_num_samples:]

            inputa, labela = process_batch_new(this_task_tr_filenames, this_task_tr_labels, dim_input, half_num_samples,
                                               reshape_with_one=True)
            inputb, labelb = process_batch_new(this_task_te_filenames, this_task_te_labels, dim_input, half_num_samples,
                                               reshape_with_one=True)
            inputc, labelc = process_un_batch_new(this_task_un_filenames, this_task_un_labels, dim_input,
                                               reshape_with_one=True)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, self.model.labela: labela,
                         self.model.labelb: labelb, self.model.inputc: inputc, self.model.labelc: labelc,
                         self.model.meta_lr: 0.0}

            result = self.sess.run([self.model.metaval_total_accuracies,self.model.metaval_total_recur_accuracies,self.model.metaval_total_pl_accuracies], feed_dict)
            metaval_accuracies.append(result[0])
            recur_accuracies.append(result[1])
            pl_accruacies.append(result[2])
            if test_idx % 100 == 0:
                tmp_accies = np.array(metaval_accuracies)
                tmp_means = np.mean(tmp_accies, 0)
                max_acc = np.max(tmp_means)
                print('***** Best Acc: ' + str(max_acc))

        metaval_accuracies = np.array(metaval_accuracies)
        recur_accuracies = np.array(recur_accuracies)
        pl_accruacies = np.array(pl_accruacies)
        means = np.mean(metaval_accuracies, 0)
        recur_means = np.mean(recur_accuracies, 0)
        pl_means = np.mean(pl_accruacies,0)
        max_idx = np.argmax(means)
        max_acc = np.max(means)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
        max_ci95 = ci95[max_idx]
        print('Mean validation accuracy and confidence intervals')
        print((means, ci95))

        print('***** Best Acc: ' + str(max_acc) + ' CI95: ' + str(max_ci95))

        ######
        logdir_base = FLAGS.test_output_dir + '/' + FLAGS.exp_string
        if not os.path.exists(logdir_base):
            os.mkdir(logdir_base)

        logdir = logdir_base + '/' + 'logdir_LST_recur'
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        out_filename = logdir + '/' + str(FLAGS.pre_train_epoch_num) + '_' + str(
            FLAGS.hard_selection) + '_hard.csv'

        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update' + str(i) for i in range(len(means))])
            writer.writerow(recur_means)
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)


    def test_with_d(self):
        NUM_TEST_POINTS = 6000
        np.random.seed(1)
        metaval_accuracies = []
        recur_accuracies = []
        pl_accruacies = []
        num_samples_per_class = FLAGS.shot_num * 2
        task_num = FLAGS.way_num * num_samples_per_class
        task_un_num = 10 * FLAGS.way_num
        task_dis_num = FLAGS.nb_ul_samples
        half_num_samples = FLAGS.shot_num
        dim_input = FLAGS.img_size * FLAGS.img_size * 3

        ####
        filename_dir = FLAGS.logdir_base + '/' + 'filenames_and_labels_6000/'
        this_setting_filename_dir = filename_dir + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.way_num) + 'way/'
        all_filenames = np.load(this_setting_filename_dir + 'test_filenames.npy').tolist()
        labels = np.load(this_setting_filename_dir + 'test_labels.npy').tolist()

        un_file_dict = {}
        un_label_dict = {}

        ####
        un_new_folder = this_setting_filename_dir + 'unlabeled_samples_for_test'

        for f in range(self.unfiles_num):
            un_file_dict[f] = np.load(un_new_folder + '/' + 'test_' + str(f) + '_un_filenames.npy').tolist()
            un_label_dict[f] = np.load(un_new_folder + '/' + 'test_' + str(f) + '_un_labels.npy').tolist()


        dis_file_dict = {}
        dis_folder = un_new_folder + '/distracting_file'
        for n in range(FLAGS.num_dis):
            dis_file_dict[n] = np.load(
                dis_folder + '/' + 'test_' + str(n + 1) + '_dis_filenames.npy').tolist()

        for test_idx in trange(NUM_TEST_POINTS):

            this_task_filenames = all_filenames[test_idx * task_num:(test_idx + 1) * task_num]

            this_task_un_filenames = []
            this_task_un_labels = []

            for j in range(self.unfiles_num):
                this_task_un_filenames.extend(un_file_dict[j][test_idx * task_un_num:(test_idx + 1) * task_un_num])
                this_task_un_labels.extend(un_label_dict[j][test_idx * task_un_num:(test_idx + 1) * task_un_num])

            this_task_dis_filenames = []
            for k in range(FLAGS.num_dis):
                current_dis_files = dis_file_dict[k][test_idx * 100:(test_idx + 1) * 100]
                this_task_dis_filenames.extend(
                    current_dis_files[:task_dis_num])

            this_task_tr_filenames = []
            this_task_tr_labels = []
            this_task_te_filenames = []
            this_task_te_labels = []
            for class_k in range(FLAGS.way_num):
                this_class_filenames = this_task_filenames[
                                       class_k * num_samples_per_class:(class_k + 1) * num_samples_per_class]
                this_class_label = labels[class_k * num_samples_per_class:(class_k + 1) * num_samples_per_class]
                this_task_tr_filenames += this_class_filenames[0:half_num_samples]
                this_task_tr_labels += this_class_label[0:half_num_samples]
                this_task_te_filenames += this_class_filenames[half_num_samples:]
                this_task_te_labels += this_class_label[half_num_samples:]

            inputa, labela = process_batch_new(this_task_tr_filenames, this_task_tr_labels, dim_input, half_num_samples,
                                               reshape_with_one=True)
            inputb, labelb = process_batch_new(this_task_te_filenames, this_task_te_labels, dim_input, half_num_samples,
                                               reshape_with_one=True)
            inputc, labelc = process_un_batch_recur(this_task_un_filenames, this_task_un_labels, dim_input,
                                                  reshape_with_one=True)
            inputd = process_dis_batch_recur(this_task_dis_filenames, dim_input, num=task_dis_num*FLAGS.num_dis, reshape_with_one=True)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, self.model.labela: labela,
                         self.model.labelb: labelb, self.model.inputc: inputc, self.model.labelc: labelc,
                         self.model.inputd: inputd, self.model.meta_lr: 0.0}

            result = self.sess.run([self.model.metaval_total_accuracies,self.model.metaval_total_recur_accuracies,self.model.metaval_total_pl_accuracies], feed_dict)
            metaval_accuracies.append(result[0])
            recur_accuracies.append(result[1])
            pl_accruacies.append(result[2])
            if test_idx % 100 == 0:
                tmp_accies = np.array(metaval_accuracies)
                tmp_means = np.mean(tmp_accies, 0)
                max_acc = np.max(tmp_means)
                print('***** Best Acc: ' + str(max_acc))

        metaval_accuracies = np.array(metaval_accuracies)
        recur_accuracies = np.array(recur_accuracies)
        pl_accruacies = np.array(pl_accruacies)
        means = np.mean(metaval_accuracies, 0)
        recur_means = np.mean(recur_accuracies, 0)
        pl_means = np.mean(pl_accruacies, 0)
        max_idx = np.argmax(means)
        max_acc = np.max(means)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
        max_ci95 = ci95[max_idx]
        print('Mean validation accuracy and confidence intervals')
        print((means, ci95))

        print('***** Best Acc: ' + str(max_acc) + ' CI95: ' + str(max_ci95))

        #####
        logdir_base = FLAGS.test_output_dir + '/' + FLAGS.exp_string
        if not os.path.exists(logdir_base):
            os.mkdir(logdir_base)

        logdir = logdir_base + '/' + 'logdir_LST_dis_exp'
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        out_filename = logdir + '/' + str(FLAGS.num_dis) + '_' + str(FLAGS.pre_train_epoch_num) + '_' + str(
            FLAGS.hard_selection) + '_d_hard.csv'

        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update' + str(i) for i in range(len(means))])
            writer.writerow(recur_means)
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)
