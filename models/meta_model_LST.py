import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
from models import Models

from tensorflow.python.platform import flags
from utils.misc import resnet_conv_block, leaky_relu

FLAGS = flags.FLAGS

class MetaModel(Models):
    def construct_model(self, prefix='metatrain_'):
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.inputc = tf.placeholder(tf.float32)
        self.labelc = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-model', reuse=None) as training_scope:
            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()
            self.swn_weights = swn_weights = self.construct_swn_weights()

            num_updates = FLAGS.train_base_epoch_num

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, _ = inp

                lossb_list = []

                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)
                emb_outputc, conv_featc = self.forward_resnet_2(inputc, weights, ss_weights, reuse=True)
                emb_outputc = tf.stop_gradient(emb_outputc)
                conv_featc = tf.stop_gradient(conv_featc)

                pseudo_labels = self.computing_pl_for_train(emb_outputa, labela, emb_outputc, 10, fc_weights)

                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_featc, swn_weights,
                                                           class_num=FLAGS.way_num,
                                                           samples_num=FLAGS.nb_ul_samples * FLAGS.way_num,
                                                           reuse=reuse)
                soft_weights = tf.reshape(soft_weights, [-1, FLAGS.way_num])

                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                outputa = self.forward_fc(emb_outputa, fc_weights)
                maml_lossa = self.loss_func(outputa, labela)

                outputc = self.forward_fc(emb_outputc, fc_weights)
                new_logits_c = tf.nn.softmax(soft_weights) * outputc
                maml_lossc_p = self.loss_func(new_logits_c, pseudo_labels)

                loss = tf.concat([maml_lossa, maml_lossc_p], axis=0)

                grads = tf.gradients(loss, list(fc_weights.values()))
                gradients = dict(zip(fc_weights.keys(), grads))
                fast_fc_weights = dict(zip(fc_weights.keys(),
                                           [fc_weights[key] - self.update_lr * gradients[key] for key in
                                            fc_weights.keys()]))

                # re-training steps
                for j in range(FLAGS.re_train_epoch_num - 1):
                    maml_lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)

                    logitsc = self.forward_fc(emb_outputc, fast_fc_weights)
                    logits_new_c = tf.nn.softmax(soft_weights) * logitsc
                    loss_c = self.loss_func(logits_new_c, pseudo_labels)

                    loss = tf.concat([maml_lossa, loss_c], axis=0)

                    grads = tf.gradients(loss, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                               [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                                fast_fc_weights.keys()]))

                outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)

                # fine-tuning steps
                for k in range(num_updates - FLAGS.re_train_epoch_num):
                    maml_lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)

                    grads = tf.gradients(maml_lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                               [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                                fast_fc_weights.keys()]))

                outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [lossb_list, accb]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn(
                    (self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0]),
                    False)

            out_dtype = [[tf.float32] * 2, tf.float32]

            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb_list, accsb = result

            self.final_pretrain_loss = final_pretrain_loss = tf.reduce_sum(lossesb_list[-2]) / tf.to_float(
                FLAGS.meta_batch_size)
            self.total_loss = total_loss = tf.reduce_sum(lossesb_list[-1]) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
            meta_optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.metatrain_op = meta_optimizer.minimize(total_loss, var_list=ss_weights.values() + fc_weights.values())
            rn_optimizer = tf.train.AdamOptimizer(self.swn_lr)
            self.meta_swn_train_op = rn_optimizer.minimize(final_pretrain_loss, var_list=swn_weights.values())

            tf.summary.scalar(prefix + 'Final Loss', total_loss)
            tf.summary.scalar(prefix + 'Final Pretrain Loss', final_pretrain_loss)
            tf.summary.scalar(prefix + 'Accuracy', total_accuracy)


    def construct_model_test(self, prefix='metatest_'):
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.inputc = tf.placeholder(tf.float32)
        self.labelc = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-test-model', reuse=None) as training_scope:

            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()
            self.swn_weights = swn_weights = self.construct_swn_weights()

            num_updates = FLAGS.test_base_epoch_num
            nums_for_hard = FLAGS.hard_selection

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, _ = inp

                inputc_lists = []
                featsc_list = []
                conv_featsc_list = []
                recursive_accb_list = []

                samples_in_one_stage = FLAGS.way_num * FLAGS.nums_in_folders
                if FLAGS.shot_num == 1:
                    steps = FLAGS.way_num * 15
                else:
                    steps = FLAGS.way_num * FLAGS.nums_in_folders

                splits_num = (FLAGS.way_num * FLAGS.nb_ul_samples + steps - samples_in_one_stage) // steps

                # Dividing the unlabeled dataset into several splits.
                for i in range(splits_num):
                    inputc_lists.append(inputc[i * steps:i * steps + samples_in_one_stage, :])

                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)

                for i in range(splits_num):
                    emb_outputc, conv_featc = self.forward_resnet_2(inputc_lists[i], weights, ss_weights, reuse=True)
                    featsc_list.append(tf.stop_gradient(emb_outputc))
                    conv_featsc_list.append(tf.stop_gradient(conv_featc))

                # Computing pseudo-labels and applying hard-selection for the unlabeled samples.
                feats_new, conv_feats_new, pseudo_labels = self.computing_pl_for_test(emb_outputa, conv_feata,
                                                                                      labela, featsc_list[0],
                                                                                      conv_featsc_list[0], 10,
                                                                                      nums_for_hard, fc_weights)
                # Computing soft-weights for the unlabled samples.
                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_feats_new, swn_weights,
                                                           class_num=FLAGS.way_num,
                                                           samples_num=nums_for_hard * FLAGS.way_num,
                                                           reuse=tf.AUTO_REUSE)
                soft_weights = tf.reshape(soft_weights, [-1, FLAGS.way_num])

                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                # recurrent updating stages
                for i in range(FLAGS.recurrent_stage_nums):
                    # When the number of recurrent stages is larger than the number of
                    # splits, previous splits will be reused.
                    if i + 1 > splits_num - 1:
                        i_next = (i + 1) % splits_num
                    else:
                        i_next = i + 1

                    fast_fc_weights, feats_output, soft_weights_output, pseudo_labels_output, accb_list, _ = \
                        self.Recurrent_Update_Stage(inputa=emb_outputa,
                                                    conv_feata=conv_feata,
                                                    labela=labela,
                                                    inputc1=feats_new,
                                                    pseudo_labelc1=pseudo_labels,
                                                    inputc2=featsc_list[i_next],
                                                    conv_featc2=conv_featsc_list[i_next],
                                                    inputb=emb_outputb,
                                                    labelb=labelb,
                                                    soft_weights=soft_weights,
                                                    num_updates=FLAGS.local_update_num,
                                                    nums_for_hard=nums_for_hard,
                                                    weights=fc_weights,
                                                    swn_weights=swn_weights)
                    recursive_accb_list.extend(accb_list)
                    pseudo_labels = pseudo_labels_output
                    feats_new = feats_output
                    soft_weights = soft_weights_output

                # Computing remaining finetuning steps for the final recurrent stage
                for j in range(num_updates - FLAGS.local_update_num):
                    maml_lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)

                    grads = tf.gradients(maml_lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                               [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                                fast_fc_weights.keys()]))
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                task_output = [accb_list, recursive_accb_list]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn(
                    (self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0]),
                    False)

            out_dtype = [[tf.float32] * num_updates, [tf.float32] * (FLAGS.local_update_num * FLAGS.recurrent_stage_nums)]

            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            accsb_list, recursive_accsb_list = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_recur_accuracies = [tf.reduce_sum(recursive_accsb_list[j]) for j in
                                                   range(FLAGS.local_update_num * FLAGS.recurrent_stage_nums)]


    def construct_model_test_with_distractors(self, prefix='metatest_'):
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.inputc = tf.placeholder(tf.float32)
        self.labelc = tf.placeholder(tf.float32)
        self.inputd = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-test-model', reuse=None) as training_scope:

            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()
            self.swn_weights = swn_weights = self.construct_swn_weights()

            num_updates = FLAGS.test_base_epoch_num
            nums_for_hard = FLAGS.hard_selection

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, _, inputd = inp

                inputc_lists = []
                inputd_lists = []
                feats_un_list = []
                conv_feats_un_list = []
                recursive_accb_list = []

                samples_in_one_stage = FLAGS.way_num * FLAGS.nums_in_folders
                dis_samples_in_one_stage = FLAGS.num_dis * FLAGS.nums_in_folders
                if FLAGS.shot_num == 1:
                    steps = FLAGS.way_num * 15
                    dis_steps = FLAGS.num_dis * 15
                else:
                    steps = FLAGS.way_num * FLAGS.nums_in_folders
                    dis_steps = FLAGS.num_dis * FLAGS.nums_in_folders

                splits_num = (FLAGS.way_num * FLAGS.nb_ul_samples + steps - samples_in_one_stage) // steps

                # Dividing the unlabeled dataset into several splits.
                for i in range(splits_num):
                    inputc_lists.append(inputc[i * steps:i * steps + samples_in_one_stage, :])
                    inputd_lists.append(inputd[i * dis_steps:i * dis_steps + dis_samples_in_one_stage, :])

                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)

                for i in range(splits_num):
                    emb_outputc, conv_featc = self.forward_resnet_2(inputc_lists[i], weights, ss_weights, reuse=True)
                    emb_outputd, conv_featd = self.forward_resnet_2(inputd_lists[i], weights, ss_weights, reuse=True)
                    feats_un_list.append(tf.stop_gradient(tf.concat([emb_outputc, emb_outputd], axis=0)))
                    conv_feats_un_list.append(tf.stop_gradient(tf.concat([conv_featc, conv_featd], axis=0)))


                # Computing pseudo-labels and applying hard-selection for the unlabeled samples.
                feats_new, conv_feats_new, pseudo_labels = self.computing_pl_for_test(emb_outputa, conv_feata,
                                                                                      labela, feats_un_list[0],
                                                                                      conv_feats_un_list[0], 10,
                                                                                      nums_for_hard, fc_weights)
                # Computing soft-weights for the unlabled samples.
                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_feats_new, swn_weights,
                                                           class_num=FLAGS.way_num,
                                                           samples_num=nums_for_hard * FLAGS.way_num,
                                                           reuse=tf.AUTO_REUSE)
                soft_weights = tf.reshape(soft_weights, [-1, FLAGS.way_num])

                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                # recurrent updating stages
                for i in range(FLAGS.recurrent_stage_nums):
                    # When the number of recurrent stages is larger than the number of
                    # splits, previous splits will be reused.
                    if i + 1 > splits_num - 1:
                        i_next = (i + 1) % splits_num
                    else:
                        i_next = i + 1

                    fast_fc_weights, feats_output, soft_weights_output, pseudo_labels_output, accb_list, _ = \
                        self.Recurrent_Update_Stage(inputa=emb_outputa,
                                                    conv_feata=conv_feata,
                                                    labela=labela,
                                                    inputc1=feats_new,
                                                    pseudo_labelc1=pseudo_labels,
                                                    inputc2=feats_un_list[i_next],
                                                    conv_featc2=conv_feats_un_list[i_next],
                                                    inputb=emb_outputb,
                                                    labelb=labelb,
                                                    soft_weights=soft_weights,
                                                    num_updates=FLAGS.local_update_num,
                                                    nums_for_hard=nums_for_hard,
                                                    weights=fc_weights,
                                                    swn_weights=swn_weights)
                    recursive_accb_list.extend(accb_list)
                    pseudo_labels = pseudo_labels_output
                    feats_new = feats_output
                    soft_weights = soft_weights_output

                # Computing remaining fine-tuning steps for the final recurrent stage
                for j in range(num_updates - FLAGS.local_update_num):
                    maml_lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)

                    grads = tf.gradients(maml_lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                               [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                                fast_fc_weights.keys()]))
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                task_output = [accb_list, recursive_accb_list]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0],
                                         self.labelc[0], self.inputd[0]), False)

            out_dtype = [[tf.float32] * num_updates, [tf.float32] * (FLAGS.local_update_num * FLAGS.recurrent_stage_nums)]

            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc,
                                      self.inputd),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            accsb_list, recursive_accsb_list = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_recur_accuracies = [tf.reduce_sum(recursive_accsb_list[j]) for j in
                                                   range(FLAGS.local_update_num * FLAGS.recurrent_stage_nums)]


    def Recurrent_Update_Stage(self, inputa, conv_feata, labela, inputc1, pseudo_labelc1, inputc2, conv_featc2,
                               inputb, labelb, soft_weights, num_updates, nums_for_hard, weights,
                               swn_weights):

        accb_list = []
        loss_list = []

        outputa = self.forward_fc(inputa, weights)
        maml_lossa = self.loss_func(outputa, labela)

        outputc = self.forward_fc(inputc1, weights)
        outputc_new = tf.nn.softmax(soft_weights) * outputc
        maml_lossc_p = self.loss_func(outputc_new, pseudo_labelc1)

        if FLAGS.re_train_epoch_num == 0:
            maml_lossc_p = maml_lossc_p * tf.zeros_like(maml_lossc_p, dtype=tf.float32)

        loss = tf.concat([maml_lossa, maml_lossc_p], axis=0)

        grads = tf.gradients(loss, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_fc_weights = dict(zip(weights.keys(),
                                   [weights[key] - self.update_lr * gradients[key] for key in
                                    weights.keys()]))

        outputb = self.forward_fc(inputb, fast_fc_weights)
        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
        accb_list.append(accb)

        # re-training steps
        for j in range(FLAGS.re_train_epoch_num - 1):

            maml_lossa = self.loss_func(self.forward_fc(inputa, fast_fc_weights), labela)

            logitsc = self.forward_fc(inputc1, fast_fc_weights)
            logitsc_new = tf.nn.softmax(soft_weights) * logitsc
            loss_c = self.loss_func(logitsc_new, pseudo_labelc1)
            loss = tf.concat([maml_lossa, loss_c], axis=0)

            grads = tf.gradients(loss, list(fast_fc_weights.values()))
            gradients = dict(zip(fast_fc_weights.keys(), grads))
            fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                       [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                        fast_fc_weights.keys()]))
            outputb = self.forward_fc(inputb, fast_fc_weights)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        # fine-tuning steps
        for k in range(num_updates - FLAGS.re_train_epoch_num):

            maml_lossa = self.loss_func(self.forward_fc(inputa, fast_fc_weights), labela)

            grads = tf.gradients(maml_lossa, list(fast_fc_weights.values()))
            gradients = dict(zip(fast_fc_weights.keys(), grads))
            fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                       [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                        fast_fc_weights.keys()]))
            outputb = self.forward_fc(inputb, fast_fc_weights)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)

        maml_lossb = self.loss_func(outputb, labelb)
        loss_list.append(maml_lossb)

        featc2_new, conv_featc2_new, pseudo_labelc2 = self.sample_selection_hard(inputa, conv_feata, labela, inputc2,
                                                                                 conv_featc2, nums_for_hard,
                                                                                 fast_fc_weights)

        relation_scores2 = self.computing_soft_weights(conv_feata, labela, conv_featc2_new, swn_weights, class_num=5,
                                                       samples_num=nums_for_hard * FLAGS.way_num,
                                                       reuse=tf.AUTO_REUSE)

        return fast_fc_weights, featc2_new, relation_scores2, pseudo_labelc2, accb_list, loss_list


    def computing_soft_weights(self, feat_s, label_s, feat_un, weights, class_num, samples_num, reuse):
        '''
        Computing soft-weights for unlabeled samples.
        '''
        num_label_s = tf.argmax(label_s, axis=1)
        w_list = []

        for i in range(class_num):
            index_samples_i = tf.where(condition=tf.equal(num_label_s, i))
            feat_class_i = tf.gather_nd(feat_s, index_samples_i)
            emb_class_i = tf.reduce_mean(feat_class_i, axis=0, keep_dims=True)

            tile_emb_class_i = tf.tile(emb_class_i, [samples_num, 1, 1, 1])
            concat_emb = tf.concat([feat_un, tile_emb_class_i], axis=3)

            w_i = tf.reshape(self.forward_swn(concat_emb, weights, reuse=tf.AUTO_REUSE), [-1, 1])
            w_list.append(w_i)

        soft_weights = tf.concat(w_list, axis=1)

        return soft_weights


    def forward_swn(self, inp, weights, reuse):
        net = resnet_conv_block(inp, weights['RN_conv1'], weights['RN_bias1'], reuse, 'RN_0')
        net = resnet_conv_block(net, weights['RN_conv2'], weights['RN_bias2'], reuse, 'RN_1')

        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])

        net = tf.matmul(net, weights['RN_fc_w1']) + weights['RN_fc_b1']
        net = leaky_relu(net)

        net = tf.matmul(net, weights['RN_fc_w2']) + weights['RN_fc_b2']

        return net


    def construct_swn_weights(self):

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        weights = {}

        weights['RN_conv1'] = tf.get_variable('RN_conv1', [3, 3, 1024, 64], initializer=conv_initializer, dtype=dtype)
        weights['RN_bias1'] = tf.Variable(tf.zeros([64]), name='RN_bias1')
        weights['RN_conv2'] = tf.get_variable('RN_conv2', [3, 3, 64, 1], initializer=conv_initializer, dtype=dtype)
        weights['RN_bias2'] = tf.Variable(tf.zeros([1]), name='RN_bias2')

        weights['RN_fc_w1'] = tf.get_variable('RN_fc_w1', [25, 8], initializer=fc_initializer)
        weights['RN_fc_b1'] = tf.Variable(tf.zeros([8]), name='RN_fc_b1')

        weights['RN_fc_w2'] = tf.get_variable('RN_fc_w2', [8, 1], initializer=fc_initializer)
        weights['RN_fc_b2'] = tf.Variable(tf.zeros([1]), name='RN_fc_b2')

        return weights


    def computing_pl_for_train(self, feat_s, label_s, feat_un, num_steps, weights):
        '''
        Computing pseudo labels for unlabeled samples during meta-training.
        '''
        output_s = self.forward_fc(feat_s, weights)
        loss_s = self.loss_func(output_s, label_s)
        grads = tf.gradients(loss_s, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict(zip(weights.keys(),
                                [weights[key] - self.update_lr * gradients[key] for key in
                                 weights.keys()]))

        for i in range(num_steps - 1):
            loss = self.loss_func(self.forward_fc(feat_s, fast_weights), label_s)
            grads = tf.gradients(loss, list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(),
                                    [fast_weights[key] - self.update_lr * gradients[key] for key in
                                     fast_weights.keys()]))

        output_un = self.forward_fc(feat_un, fast_weights)
        probc = tf.nn.softmax(output_un)
        pseudo_labels = tf.stop_gradient(tf.one_hot(
            tf.argmax(probc, axis=-1),
            tf.shape(probc)[1],
            dtype=probc.dtype,
        ))

        return pseudo_labels


    def computing_pl_for_test(self, feat_s, conv_feat_s, label_s, feat_un, conv_feat_un, num_steps, nums_for_hard,
                                  weights):
        '''
        Computing pseudo labels and applying hard-selection for unlabeled samples during meta-testing.
        '''

        output_s = self.forward_fc(feat_s, weights)
        loss_s = self.loss_func(output_s, label_s)
        grads = tf.gradients(loss_s, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict(zip(weights.keys(),
                                [weights[key] - self.update_lr * gradients[key] for key in
                                 weights.keys()]))

        for i in range(num_steps - 1):
            loss = self.loss_func(self.forward_fc(feat_s, fast_weights), label_s)
            grads = tf.gradients(loss, list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(),
                                    [fast_weights[key] - self.update_lr * gradients[key] for key in
                                     fast_weights.keys()]))

        feats_new, conv_feat_new, pseudo_labels = self.sample_selection_hard(feat_s, conv_feat_s, label_s, feat_un,
                                                                             conv_feat_un, nums_for_hard, fast_weights)

        return feats_new, conv_feat_new, pseudo_labels


    def sample_selection_hard(self, inputs_proto, conv_inputs_proto, labels_proto, inputs, conv_inputs, nums_for_hard,
                              weights):
        '''
        Selecting unlabeled samples with high prediction scores.
        '''

        #pseudo labeling for unlabeled samples
        logits = self.forward_fc(inputs, weights)
        probs = tf.nn.softmax(logits)
        labels = tf.argmax(probs, axis=-1)

        labels_num_proto = tf.argmax(labels_proto, axis=-1)
        new_samples = []
        new_conv_samples = []
        new_labels = []

        for i in range(self.dim_output):
            # Sometimes, none unlabeled samples are predicted for the class
            # and labeled samples from support set are used instead.
            index_sample_i = tf.where(condition=tf.equal(labels, i))
            index_sample_proto_i = tf.where(condition=tf.equal(labels_num_proto, i))
            num_i = tf.reduce_sum(tf.cast(tf.equal(labels, i), dtype=tf.int16))
            num_total = num_i + tf.constant(FLAGS.shot_num, dtype=tf.int16)
            #selecting fc features
            samples_i = tf.gather_nd(inputs, index_sample_i)
            samples_proto_i = tf.gather_nd(inputs_proto, index_sample_proto_i)
            new_samples_i = tf.concat([samples_i, samples_proto_i], axis=0)
            #selecting conv features
            conv_samples_i = tf.gather_nd(conv_inputs, index_sample_i)
            conv_samples_proto_i = tf.gather_nd(conv_inputs_proto, index_sample_proto_i)
            new_conv_samples_i = tf.concat([conv_samples_i, conv_samples_proto_i], axis=0)
            #selecting corresponding labels
            labels_i = tf.one_hot(tf.gather_nd(labels, index_sample_i), tf.shape(probs)[1], dtype=probs.dtype)
            labels_proto_i = tf.gather_nd(labels_proto, index_sample_proto_i)
            new_labels_i = tf.concat([labels_i, labels_proto_i], axis=0)
            #selecting corresponding prediction scores
            probs_i = tf.gather_nd(probs, index_sample_i)[:, i]
            probs_proto_fake = tf.constant(0, dtype=tf.float32, shape=[FLAGS.shot_num, ])
            new_probs_i = tf.concat([probs_i, probs_proto_fake], axis=0)

            def repeat_and_rank(samples, conv_samples, labels, probs, k):
                list_samples = [samples for i in range(k)]
                list_conv_samples = [conv_samples for i in range(k)]
                list_labels = [labels for i in range(k)]
                list_probs = [probs]
                list_probs.extend([tf.zeros_like(probs, dtype=probs.dtype) for i in range(k - 1)])

                samples_pad = tf.concat(list_samples, axis=0)
                conv_samples_pad = tf.concat(list_conv_samples, axis=0)
                labels_pad = tf.concat(list_labels, axis=0)
                probs_pad = tf.concat(list_probs, axis=0)

                index_rank = tf.reshape(tf.nn.top_k(probs_pad, k=k).indices, [k, 1])
                samples_rank = tf.gather_nd(samples_pad, index_rank)
                conv_samples_rank = tf.gather_nd(conv_samples_pad, index_rank)
                labels_rank = tf.gather_nd(labels_pad, index_rank)

                return samples_rank, conv_samples_rank, labels_rank

            def normal_rank(samples, conv_samples, labels, probs, k):
                index_rank = tf.reshape(tf.nn.top_k(probs, k=k).indices, [k, 1])
                samples_rank = tf.gather_nd(samples, index_rank)
                conv_samples_rank = tf.gather_nd(conv_samples, index_rank)
                labels_rank = tf.gather_nd(labels, index_rank)

                return samples_rank, conv_samples_rank, labels_rank

            samples_rank_i, conv_samples_rank_i, labels_rank_i = tf.cond(
                num_total < tf.constant(nums_for_hard, dtype=tf.int16),
                lambda: repeat_and_rank(new_samples_i,
                                        new_conv_samples_i,
                                        new_labels_i, new_probs_i, nums_for_hard),
                lambda: normal_rank(new_samples_i, new_conv_samples_i,
                                    new_labels_i, new_probs_i, nums_for_hard))

            new_samples.append(tf.stop_gradient(samples_rank_i))
            new_conv_samples.append(tf.stop_gradient(conv_samples_rank_i))
            new_labels.append(tf.stop_gradient(labels_rank_i))

        new_samples = tf.concat(new_samples, axis=0)
        new_conv_samples = tf.concat(new_conv_samples, axis=0)
        new_labels = tf.concat(new_labels, axis=0)

        return new_samples, new_conv_samples, new_labels



