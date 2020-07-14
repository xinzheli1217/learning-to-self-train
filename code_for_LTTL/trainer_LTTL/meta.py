import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import cv2
import pdb

from tqdm import trange
from model_LTTL.meta_model import MetaModel
from tensorflow.python.platform import flags
from misc import process_batch_new, process_un_batch_new, process_dis_batch_2_new, process_batch_test

FLAGS = flags.FLAGS

class MetaTrainer:
    def __init__(self):
        if FLAGS.metatrain:
            print('Building train model')
            self.model = MetaModel()
            self.model.construct_train_model_LTTL()
            self.model.summ_op = tf.summary.merge_all()
            print('Finish building train model')
        else:
            print('Building test model')
            self.model = MetaModel()
            if FLAGS.use_distractors:
                if FLAGS.use_hard:
                    self.model.construct_model_test_LTTL_distractors_hard(prefix='metatest_')
                else:
                    self.model.construct_model_test_LTTL_distractors(prefix='metatest_')
            else:
                if FLAGS.use_hard:
                    self.model.construct_model_test_LTTL_hard(prefix='metatest')
                else:
                    self.model.construct_model_test_LTTL(prefix='metatest')
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

        if FLAGS.metatrain or FLAGS.test_iter == 0:
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
                fc_weights = np.load(os.path.join(weights_path, "fc_weights_"+str(FLAGS.shot_num)+"shot.npy"), allow_pickle=True).tolist()
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

        if FLAGS.metatrain:
            self.train()
        else:
            if FLAGS.use_distractors:
                self.test_with_d()
            else:
                self.test()


    def train(self):
        exp_string = FLAGS.exp_string
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Done initializing, starting training')
        loss_list, acc_list = [], []
        pl_save_list = []
        train_lr = FLAGS.meta_lr
        swn_lr = FLAGS.swn_lr
        kld_lr = FLAGS.kld_lr
        task_un_num = FLAGS.nb_ul_samples * FLAGS.way_num
        num_samples_per_class = FLAGS.shot_num + 15
        task_num = FLAGS.way_num * num_samples_per_class
        num_samples_per_class_test = FLAGS.shot_num * 2
        test_task_num = FLAGS.way_num * num_samples_per_class_test
        task_dis_num = FLAGS.nb_ul_samples

        epitr_sample_num = FLAGS.shot_num
        epite_sample_num = 15
        test_task_sample_num = FLAGS.shot_num
        dim_input = FLAGS.img_size * FLAGS.img_size * 3

        #### Important !!! this_setting_filename_dir
        filename_dir = FLAGS.logdir_base + '/' + 'filenames_and_labels/'
        this_setting_filename_dir = filename_dir + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.way_num) + 'way/'

        all_filenames = np.load(this_setting_filename_dir + 'train_filenames.npy').tolist()
        labels = np.load(this_setting_filename_dir + 'train_labels.npy').tolist()

        un_train_folder = this_setting_filename_dir + 'unlabeled_samples_for_train/'
        all_unlabeled_filenames = np.load(
            un_train_folder + 'train' + '_' + str(FLAGS.nb_ul_samples) + '_un_filenames.npy').tolist()
        all_unlabeled_labels = np.load(
            un_train_folder + 'train' + '_' + str(FLAGS.nb_ul_samples) + '_un_labels.npy').tolist()

        test_idx = 0

        new = False

        for train_idx in trange(FLAGS.metatrain_iterations):

            if new:
                new_train_idx = train_idx + 15000
            else:
                new_train_idx = train_idx

            inputa = []
            labela = []
            inputb = []
            labelb = []
            inputc = []
            labelc = []

            for meta_batch_idx in range(FLAGS.meta_batch_size):

                this_task_filenames = all_filenames[(new_train_idx * FLAGS.meta_batch_size + meta_batch_idx) * task_num:(new_train_idx * FLAGS.meta_batch_size + meta_batch_idx + 1) * task_num]
                this_task_un_filenames = all_unlabeled_filenames[(new_train_idx * FLAGS.meta_batch_size + meta_batch_idx) * task_un_num:(new_train_idx * FLAGS.meta_batch_size + meta_batch_idx + 1) * task_un_num]
                this_task_un_labels = all_unlabeled_labels[(new_train_idx * FLAGS.meta_batch_size + meta_batch_idx) * task_un_num:(new_train_idx * FLAGS.meta_batch_size + meta_batch_idx + 1) * task_un_num]

                this_task_tr_filenames = []
                this_task_tr_labels = []
                this_task_te_filenames = []
                this_task_te_labels = []
                this_task_dis_filenames = []

                for class_k in range(FLAGS.way_num):
                    this_class_filenames = this_task_filenames[
                                           class_k * num_samples_per_class:(class_k + 1) * num_samples_per_class]
                    this_class_label = labels[class_k * num_samples_per_class:(class_k + 1) * num_samples_per_class]
                    this_task_tr_filenames += this_class_filenames[0:epitr_sample_num]
                    this_task_tr_labels += this_class_label[0:epitr_sample_num]
                    this_task_te_filenames += this_class_filenames[epitr_sample_num:]
                    this_task_te_labels += this_class_label[epitr_sample_num:]

                this_inputa, this_labela = process_batch_new(this_task_tr_filenames, this_task_tr_labels, dim_input,
                                                         epitr_sample_num, reshape_with_one=False)
                this_inputb, this_labelb = process_batch_new(this_task_te_filenames, this_task_te_labels, dim_input,
                                                         epite_sample_num, reshape_with_one=False)
                this_inputc, this_labelc = process_un_batch_new(this_task_un_filenames, this_task_un_labels, dim_input)

                inputa.append(this_inputa)
                labela.append(this_labela)
                inputb.append(this_inputb)
                labelb.append(this_labelb)
                inputc.append(this_inputc)
                labelc.append(this_labelc)

            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)
            inputc = np.array(inputc)
            labelc = np.array(labelc)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, self.model.labela: labela,
                         self.model.labelb: labelb, self.model.inputc: inputc, self.model.labelc: labelc,
                         self.model.meta_lr: train_lr, self.model.swn_lr: swn_lr}

            input_tensors = [self.model.metatrain_op]
            input_tensors.extend([self.model.meta_swn_train_op])
            input_tensors.extend([self.model.total_loss])
            input_tensors.extend([self.model.total_accuracy])
            input_tensors.extend([self.model.total_accuracy_pl])

            if (train_idx % FLAGS.meta_sum_step == 0 or train_idx % FLAGS.meta_print_step == 0):
                input_tensors.extend([self.model.summ_op, self.model.total_loss])

            result = self.sess.run(input_tensors, feed_dict)

            loss_list.append(result[2])
            acc_list.append(result[3])

            if train_idx % FLAGS.meta_sum_step == 0:
                train_writer.add_summary(result[5], train_idx)

            if train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Acc:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list, acc_pl_list = [], [], []


            if (train_idx != 0) and train_idx % FLAGS.meta_save_step == 0:
                weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights1 = self.sess.run(self.model.fc_weights1)
                fc_weights2 = self.sess.run(self.model.fc_weights2)
                swn_weights = self.sess.run(self.model.swn_weights)
                np.save(FLAGS.logdir + '/' + exp_string + '/weights_' + str(train_idx) + '.npy', weights)
                np.save(FLAGS.logdir + '/' + exp_string + '/ss_weights_' + str(train_idx) + '.npy', ss_weights)
                np.save(FLAGS.logdir + '/' + exp_string + '/fc_weights1_' + str(train_idx) + '.npy', fc_weights1)
                np.save(FLAGS.logdir + '/' + exp_string + '/fc_weights2_' + str(train_idx) + '.npy', fc_weights2)
                np.save(FLAGS.logdir + '/' + exp_string + '/swn_weights_' + str(train_idx) + '.npy', swn_weights)

                if (train_idx != 0) and train_idx % FLAGS.lr_drop_step == 0:
                    train_lr = train_lr * 0.5
                    swn_lr = swn_lr * 0.5
                    kld_lr = kld_lr * 0.5
                    if train_lr < FLAGS.min_meta_lr:
                        train_lr = FLAGS.min_meta_lr
                    if swn_lr < FLAGS.min_meta_lr:
                        swn_lr = FLAGS.min_meta_lr
                    if kld_lr < FLAGS.min_meta_lr:
                        kld_lr = FLAGS.min_meta_lr
                    print('Train LR: {}'.format(train_lr))

        weights = self.sess.run(self.model.weights)
        ss_weights = self.sess.run(self.model.ss_weights)
        fc_weights1 = self.sess.run(self.model.fc_weights1)
        fc_weights2 = self.sess.run(self.model.fc_weights2)
        swn_weights = self.sess.run(self.model.swn_weights)
        np.save(FLAGS.logdir + '/' + exp_string + '/weights_' + str(train_idx + 1) + '.npy', weights)
        np.save(FLAGS.logdir + '/' + exp_string + '/ss_weights_' + str(train_idx + 1) + '.npy', ss_weights)
        np.save(FLAGS.logdir + '/' + exp_string + '/fc_weights1_' + str(train_idx + 1) + '.npy', fc_weights1)
        np.save(FLAGS.logdir + '/' + exp_string + '/fc_weights2_' + str(train_idx + 1) + '.npy', fc_weights2)
        np.save(FLAGS.logdir + '/' + exp_string + '/swn_weights_' + str(train_idx + 1) + '.npy', swn_weights)


    def test(self):
        NUM_TEST_POINTS = 6000
        np.random.seed(1)
        metaval_accuracies = []
        metaval_accuracies1 = []
        metaval_accuracies2 = []
        ratio1_lists = []
        ratio2_lists = []
        n_eq_list = []
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

            result = self.sess.run([self.model.metaval_total_accuracies, self.model.metaval_total_accuracies1,
                                    self.model.metaval_total_accuracies2, self.model.ratios_1, self.model.n_sum], feed_dict)
            metaval_accuracies.append(result[0])
            metaval_accuracies1.append(result[1])
            metaval_accuracies2.append(result[2])
            ratio1_lists.append(result[3])
            n_eq_list.append(result[4])

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        metaval_accuracies1 = np.array(metaval_accuracies1)
        means1 = np.mean(metaval_accuracies1, 0)
        metaval_accuracies2 = np.array(metaval_accuracies2)
        means2 = np.mean(metaval_accuracies2, 0)
        ratios1 = np.array(ratio1_lists)
        ratios1_means = np.mean(ratios1, 0)
        #np.save('/media/lxz/TOSHIBA EXT1/experiments/ratios1_0'+ str(FLAGS.reject_num) + '_hard.npy', ratios1_means)
        n_eq = np.array(n_eq_list)
        n_eq_means = np.mean(n_eq)

        max_idx = np.argmax(means)
        max_acc = np.max(means)
        max_acc1 = np.max(means1)
        max_acc2 = np.max(means2)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
        max_ci95 = ci95[max_idx]
        print('Split Num: ' + str(FLAGS.split_num))
        print('Mean validation accuracy and confidence intervals')
        print((means, ci95))

        print('***** Best Acc: ' + str(max_acc) + ' CI95: ' + str(max_ci95) + ' Acc1: ' + str(max_acc1) + ' Acc2: ' + str(max_acc2))

        ######

        logdir_base = FLAGS.test_output_dir + '/' + FLAGS.exp_string
        if not os.path.exists(logdir_base):
            os.mkdir(logdir_base)

        logdir = logdir_base + '/' + 'logdir_new'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if FLAGS.use_hard:
            out_filename = logdir + '/' + str(FLAGS.pre_train_epoch_num) + '_' + str(
                    FLAGS.hard_selection) + '_hard_new.csv'
        else:
            out_filename = logdir + '/' + str(FLAGS.pre_train_epoch_num) + '_new.csv'

        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update' + str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(means1)
            writer.writerow(means2)
            writer.writerow(stds)
            writer.writerow(ci95)


    def test_with_d(self):
        NUM_TEST_POINTS = 6000
        np.random.seed(1)
        metaval_accuracies = []
        metaval_accuracies1 = []
        metaval_accuracies2 = []
        ratio1_lists = []
        ratio2_lists = []
        n_sum_list = []
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
            inputc, labelc = process_un_batch_new(this_task_un_filenames, this_task_un_labels, dim_input,
                                                  reshape_with_one=True)
            inputd = process_dis_batch_2_new(this_task_dis_filenames, dim_input, num=task_dis_num*FLAGS.num_dis, reshape_with_one=True)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, self.model.labela: labela,
                         self.model.labelb: labelb, self.model.inputc: inputc, self.model.labelc: labelc,
                         self.model.inputd: inputd, self.model.meta_lr: 0.0}

            result = self.sess.run([self.model.metaval_total_accuracies, self.model.metaval_total_accuracies1,
                                    self.model.metaval_total_accuracies2, self.model.ratios_1, self.model.n_sum],
                                   feed_dict)
            metaval_accuracies.append(result[0])
            metaval_accuracies1.append(result[1])
            metaval_accuracies2.append(result[2])
            ratio1_lists.append(result[3])
            n_sum_list.append(result[4])

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        metaval_accuracies1 = np.array(metaval_accuracies1)
        means1 = np.mean(metaval_accuracies1, 0)
        metaval_accuracies2 = np.array(metaval_accuracies2)
        means2 = np.mean(metaval_accuracies2, 0)
        ratios1 = np.array(ratio1_lists)
        ratios1_means = np.mean(ratios1, 0)
        # np.save('/media/lxz/TOSHIBA EXT1/experiments/ratios1_0'+ str(int(FLAGS.reject_num)) + '_d_hard.npy', ratios1_means)

        n_sum = np.array(n_sum_list)
        n_sum_means = np.mean(n_sum)

        max_idx = np.argmax(means)
        max_acc = np.max(means)
        max_acc1 = np.max(means1)
        max_acc2 = np.max(means2)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)
        max_ci95 = ci95[max_idx]
        print('Split Num: ' + str(FLAGS.split_num))
        print('Mean validation accuracy and confidence intervals')
        print((means, ci95))

        print('***** Best Acc: ' + str(max_acc) + ' CI95: ' + str(max_ci95) + ' Acc1: ' + str(
            max_acc1) + ' Acc2: ' + str(max_acc2))

        #####
        logdir_base = FLAGS.test_output_dir + '/' + FLAGS.exp_string
        if not os.path.exists(logdir_base):
            os.mkdir(logdir_base)

        logdir = logdir_base + '/' + 'logdir_d_new'
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        if FLAGS.use_hard:
            out_filename = logdir + '/' + str(FLAGS.pre_train_epoch_num) + '_' + str(
                    FLAGS.hard_selection) + '_d_hard.csv'
        else:
            out_filename = logdir + '/' + str(FLAGS.pre_train_epoch_num) + '_d.csv'

        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update' + str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(means1)
            writer.writerow(means2)
            writer.writerow(stds)
            writer.writerow(ci95)


