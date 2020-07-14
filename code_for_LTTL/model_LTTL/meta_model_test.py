import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
from model_LTTL.model_analysis import Analysis

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class MetaModel(Analysis):

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
            self.fc_weights_pl = fc_weights_pl = self.construct_fc_weights()
            self.swn_weights = swn_weights = self.construct_swn_weights()
            self.fc_weights1 = fc_weights1 = self.construct_fc_weights_coteaching(seed=100, str='1')
            self.fc_weights2 = fc_weights2 = self.construct_fc_weights_coteaching(seed=200, str='2')

            num_updates = FLAGS.test_base_epoch_num
            nums_for_hard = FLAGS.hard_selection

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, labelc = inp

                inputc_lists = []
                labelc_lists = []
                featsc_list = []
                pl_lists = []
                conv_featsc_list = []
                recursive_accb_list = []

                samples_in_one_stage = FLAGS.way_num * FLAGS.nums_in_folders
                if FLAGS.shot_num == 1:
                    #steps = FLAGS.way_num * 15
                    steps = FLAGS.way_num * FLAGS.nums_in_folders
                    #samples_num = 5
                    samples_num = 3
                else:
                    steps = FLAGS.way_num * FLAGS.nums_in_folders
                    samples_num = 2

                # Dividing the unlabeled dataset into several splits.
                for i in range(samples_num):
                    inputc_lists.append(inputc[i * steps:i * steps + samples_in_one_stage, :])
                    labelc_lists.append(labelc[i * steps:i * steps + samples_in_one_stage, :])

                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)

                for i in range(samples_num):
                    emb_outputc, conv_featc = self.forward_resnet_2(inputc_lists[i], weights, ss_weights, reuse=True)
                    featsc_list.append(tf.stop_gradient(emb_outputc))
                    conv_featsc_list.append(tf.stop_gradient(conv_featc))

                # Computing pseudo-labels and applying hard-selection for the unlabeled samples.
                feats_new, conv_feats_new, pseudo_labels, acc_pl = self.computing_pl_for_test(emb_outputa, conv_feata,
                                                                                      labela, featsc_list[0],
                                                                                      conv_featsc_list[0],
                                                                                      labelc_lists[0], 10,
                                                                                      nums_for_hard, fc_weights_pl)
                pl_lists.append(acc_pl)
                # Computing soft-weights for the unlabled samples.
                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_feats_new, swn_weights,
                                                           class_num=5,
                                                           samples_num=nums_for_hard * FLAGS.way_num,
                                                           reuse=tf.AUTO_REUSE)

                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                # recurrent updating stages
                s = FLAGS.pre_train_epoch_num
                steps_list = [s, s, s, s, s,
                              s, s, s, s, s,
                              s, s, s, s, s]
                for i in range(FLAGS.recurrent_stage_nums):
                    # When the number of recurrent stages is larger than the number of
                    # splits, previous splits will be reused.
                    if i + 1 > samples_num - 1:
                        i_next = (i + 1) % samples_num
                    else:
                        i_next = i + 1

                    fast_fc_weights1, fast_fc_weights2, feats_output, soft_weights_output, pseudo_labels_output, accb_list, loss_list, acc_pl = \
                        self.Recurrent_Update_Stage_2(inputa=emb_outputa,
                                                     conv_feata=conv_feata,
                                                     labela=labela,
                                                     inputc1=feats_new,
                                                     pseudo_labelc1=pseudo_labels,
                                                     inputc2=featsc_list[i_next],
                                                     conv_featc2=conv_featsc_list[i_next],
                                                     labelc2=labelc_lists[i_next],
                                                     inputb=emb_outputb,
                                                     labelb=labelb,
                                                     soft_weights=soft_weights,
                                                     num_updates=FLAGS.local_update_num,
                                                     nums_for_hard=nums_for_hard,
                                                     weights1=fc_weights1,
                                                     weights2=fc_weights2,
                                                     swn_weights=swn_weights,
                                                     pretrain_steps=steps_list[i],
                                                     steps=i+1)
                    recursive_accb_list.extend(accb_list)
                    pseudo_labels = pseudo_labels_output
                    feats_new = feats_output
                    soft_weights = soft_weights_output
                    pl_lists.append(acc_pl)

                # Computing remaining finetuning steps for the final recurrent stage
                for j in range(num_updates - FLAGS.local_update_num):
                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=0.0), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=0.0), labela)
                    grads1 = tf.gradients(maml_lossa1, list(fast_fc_weights1.values()))
                    gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
                    fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                               [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                                fast_fc_weights1.keys()]))
                    grads2 = tf.gradients(maml_lossa2, list(fast_fc_weights2.values()))
                    gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
                    fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                               [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                                fast_fc_weights2.keys()]))
                    outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                    outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                    outputb = 0.5 * (outputb1 + outputb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                task_output = [accb_list, recursive_accb_list, pl_lists]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn(
                    (self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0]),
                    False)

            out_dtype = [[tf.float32] * num_updates, [tf.float32] * (FLAGS.local_update_num * FLAGS.recurrent_stage_nums),
                         [tf.float32] * (FLAGS.recurrent_stage_nums + 1)]

            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            accsb_list, recursive_accsb_list, accs_pl = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]

            self.metaval_total_pl_accuracies = total_pl_accuracies = [tf.reduce_sum(accs_pl[j]) for j in
                                                                range(FLAGS.recurrent_stage_nums+1)]
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
            self.fc_weights_pl = fc_weights_pl = self.construct_fc_weights()
            self.swn_weights = swn_weights = self.construct_swn_weights()
            self.fc_weights1 = fc_weights1 = self.construct_fc_weights_coteaching(seed=100, str='1')
            self.fc_weights2 = fc_weights2 = self.construct_fc_weights_coteaching(seed=200, str='2')

            num_updates = FLAGS.test_base_epoch_num
            nums_for_hard = FLAGS.hard_selection

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, _, inputd = inp

                input_un_lists = []
                feats_un_list = []
                conv_feats_un_list = []
                recursive_accb_list = []

                samples_in_one_stage = (FLAGS.way_num + FLAGS.num_dis) * FLAGS.nums_in_folders
                half_num = int(samples_in_one_stage / 2)
                if FLAGS.shot_num == 1:
                    steps = (FLAGS.way_num + FLAGS.num_dis) * FLAGS.nums_in_folders
                else:
                    steps = (FLAGS.way_num + FLAGS.num_dis) * FLAGS.nums_in_folders

                splits_num = ((FLAGS.way_num + FLAGS.num_dis) * FLAGS.nb_ul_samples + steps - samples_in_one_stage) // steps

                input_un = tf.concat([inputc, inputd], axis=0)
                input_un = tf.random_shuffle(input_un, seed=6)

                for i in range(splits_num):
                    input_un_lists.append(input_un[i * steps:i * steps + samples_in_one_stage, :])

                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)

                for i in range(splits_num):
                    # Preventing OOM.
                    emb_output_un_1, conv_feat_un_1 = self.forward_resnet_2(input_un_lists[i][:half_num, :], weights,
                                                                            ss_weights, reuse=True)
                    emb_output_un_2, conv_feat_un_2 = self.forward_resnet_2(input_un_lists[i][half_num:, :], weights,
                                                                            ss_weights, reuse=True)
                    feats_un_list.append(tf.stop_gradient(tf.concat([emb_output_un_1, emb_output_un_2], axis=0)))
                    conv_feats_un_list.append(tf.stop_gradient(tf.concat([conv_feat_un_1, conv_feat_un_2], axis=0)))

                # Computing pseudo-labels and applying hard-selection for the unlabeled samples.
                feats_new, conv_feats_new, pseudo_labels = self.computing_pl_for_test(emb_outputa, conv_feata,
                                                                                      labela, feats_un_list[0],
                                                                                      conv_feats_un_list[0], 10,
                                                                                      nums_for_hard, fc_weights_pl)
                # Computing soft-weights for the unlabled samples.
                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_feats_new, swn_weights,
                                                           class_num=5,
                                                           samples_num=nums_for_hard * FLAGS.way_num,
                                                           reuse=tf.AUTO_REUSE)

                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)
                s = FLAGS.pre_train_epoch_num
                steps_list = [s,s,s,s]
                # recurrent updating stages
                for i in range(FLAGS.recurrent_stage_nums):
                    # When the number of recurrent stages is larger than the number of
                    # splits, previous splits will be reused.
                    if i + 1 > splits_num - 1:
                        i_next = (i + 1) % splits_num
                    else:
                        i_next = i + 1

                    fast_fc_weights1, fast_fc_weights2, feats_output, soft_weights_output, pseudo_labels_output, accb_list, _ = \
                        self.Recurrent_Update_Stage_3(inputa=emb_outputa,
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
                                                    weights1=fc_weights1,
                                                    weights2=fc_weights2,
                                                    swn_weights=swn_weights,
                                                    pretrain_steps=steps_list[i],
                                                    steps=i+1)
                    recursive_accb_list.extend(accb_list)
                    pseudo_labels = pseudo_labels_output
                    feats_new = feats_output
                    soft_weights = soft_weights_output

                # Computing remaining fine-tuning steps for the final recurrent stage
                for j in range(num_updates - FLAGS.local_update_num):
                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=0.0),
                                                 labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=0.0),
                                                 labela)
                    grads1 = tf.gradients(maml_lossa1, list(fast_fc_weights1.values()))
                    gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
                    fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                                [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                                 fast_fc_weights1.keys()]))
                    grads2 = tf.gradients(maml_lossa2, list(fast_fc_weights2.values()))
                    gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
                    fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                                [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                                 fast_fc_weights2.keys()]))
                    outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                    outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                    outputb = 0.5 * (outputb1 + outputb2)
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
                               inputb, labelb, soft_weights, num_updates, nums_for_hard, weights1, weights2,
                               swn_weights, steps):

        rate = FLAGS.reject_num * 0.1
        rate_schedule = np.ones(20) * rate
        rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

        accb_list = []
        loss_list = []
        logits1 = self.forward_dropout_fc(inputa, weights1, dp_rate=0.0)
        maml_lossa1 = self.loss_func(logits1, labela)
        logits2 = self.forward_dropout_fc(inputa, weights2, dp_rate=0.0)
        maml_lossa2 = self.loss_func(logits2, labela)

        logitsc1 = self.forward_dropout_fc(inputc1, weights1, dp_rate=0.0)
        logitsc2 = self.forward_dropout_fc(inputc1, weights2, dp_rate=0.0)
        nums = nums_for_hard * FLAGS.way_num


        maml_loss_p1, maml_loss_p2 = self.my_coteaching_2(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                          feat_s=inputa, label_s=labela, fc_weights1=weights1,
                                                          fc_weights2=weights2, forget_rate=rate_schedule[0],
                                                          num=nums, hard=True)

        loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
        grads1 = tf.gradients(loss1, list(weights1.values()))
        gradients1 = dict(zip(weights1.keys(), grads1))
        fast_fc_weights1 = dict(zip(weights1.keys(),
                                [weights1[key] - self.update_lr * gradients1[key] for key in
                                 weights1.keys()]))

        loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
        grads2 = tf.gradients(loss2, list(weights2.values()))
        gradients2 = dict(zip(weights2.keys(), grads2))
        fast_fc_weights2 = dict(zip(weights2.keys(),
                                    [weights2[key] - self.update_lr * gradients2[key] for key in
                                     weights2.keys()]))

        outputb1 = self.forward_fc(inputb, fast_fc_weights1)
        outputb2 = self.forward_fc(inputb, fast_fc_weights2)
        outputb = 0.5 * (outputb1 + outputb2)
        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
        accb_list.append(accb)

        # pre-training steps
        for j in range(steps - 1):
            maml_lossa1 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights1, dp_rate=0.0),
                                         labela)
            maml_lossa2 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights2, dp_rate=0.0),
                                         labela)
            logitsc1 = self.forward_dropout_fc(inputc1, fast_fc_weights1, dp_rate=0.0)
            logitsc2 = self.forward_dropout_fc(inputc1, fast_fc_weights2, dp_rate=0.0)

            maml_loss_p1, maml_loss_p2 = self.my_coteaching_2(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                              feat_s=inputa, label_s=labela, fc_weights1=fast_fc_weights1,
                                                              fc_weights2=fast_fc_weights2, forget_rate=rate_schedule[j+1],
                                                              num=nums, hard=True)

            loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
            grads1 = tf.gradients(loss1, list(fast_fc_weights1.values()))
            gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
            fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                        [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                         fast_fc_weights1.keys()]))
            loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
            grads2 = tf.gradients(loss2, list(fast_fc_weights2.values()))
            gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
            fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                        [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                         fast_fc_weights2.keys()]))

            outputb1 = self.forward_fc(inputb, fast_fc_weights1)
            outputb2 = self.forward_fc(inputb, fast_fc_weights2)
            outputb = 0.5 * (outputb1 + outputb2)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        # fine-tuning steps
        for k in range(num_updates - steps):
            maml_lossa1 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights1, dp_rate=0.0), labela)
            maml_lossa2 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights2, dp_rate=0.0), labela)
            grads1 = tf.gradients(maml_lossa1, list(fast_fc_weights1.values()))
            gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
            fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                        [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                         fast_fc_weights1.keys()]))
            grads2 = tf.gradients(maml_lossa2, list(fast_fc_weights2.values()))
            gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
            fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                        [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                         fast_fc_weights2.keys()]))

            outputb1 = self.forward_fc(inputb, fast_fc_weights1)
            outputb2 = self.forward_fc(inputb, fast_fc_weights2)
            outputb = 0.5 * (outputb1 + outputb2)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        maml_lossb = self.loss_func(outputb, labelb)
        loss_list.append(maml_lossb)

        featc2_new, conv_featc2_new, pseudo_labelc2 = self.sample_selection_hard_co(inputa, conv_feata, labela, inputc2,
                                                                                    conv_featc2, nums_for_hard,
                                                                                    fast_fc_weights1, fast_fc_weights2)

        relation_scores2 = self.computing_soft_weights(conv_feata, labela, conv_featc2_new, swn_weights, class_num=5,
                                                       samples_num=nums_for_hard * FLAGS.way_num,
                                                       reuse=tf.AUTO_REUSE)

        return fast_fc_weights1, fast_fc_weights2, featc2_new, relation_scores2, pseudo_labelc2, accb_list, loss_list


    def Recurrent_Update_Stage_2(self, inputa, conv_feata, labela, inputc1, pseudo_labelc1, inputc2, conv_featc2,
                                 labelc2, inputb, labelb, soft_weights, num_updates, nums_for_hard, weights1, weights2,
                                 swn_weights, pretrain_steps, steps):
        if steps == 1:
            rate = FLAGS.reject_num * 0.1
        else:
            rate = 0.0
        rate_schedule = np.ones(40) * rate
        rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

        accb_list = []
        loss_list = []
        logits1 = self.forward_dropout_fc(inputa, weights1, dp_rate=0.0)
        maml_lossa1 = self.loss_func(logits1, labela)
        logits2 = self.forward_dropout_fc(inputa, weights2, dp_rate=0.0)
        maml_lossa2 = self.loss_func(logits2, labela)

        logitsc1 = self.forward_dropout_fc(inputc1, weights1, dp_rate=0.0)
        logitsc2 = self.forward_dropout_fc(inputc1, weights2, dp_rate=0.0)
        nums = nums_for_hard * FLAGS.way_num

        if steps == 1:
            maml_loss_p1, maml_loss_p2 = self.my_coteaching_2(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                            feat_s=inputa, label_s=labela, fc_weights1=weights1,
                                                            fc_weights2=weights2, forget_rate=rate_schedule[0],
                                                            num=nums, hard=True)
        else:
            maml_loss_p1, maml_loss_p2 = self.my_coteaching(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                            forget_rate=rate_schedule[0],num=nums)
            #maml_loss_p1, maml_loss_p2 = self.my_coteaching_3(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                            #feat_s=inputa, label_s=labela, fc_weights1=weights1,
                                                            #fc_weights2=weights2, forget_rate=rate_schedule[0],
                                                            #num=nums, hard=True)
        loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
        grads1 = tf.gradients(loss1, list(weights1.values()))
        gradients1 = dict(zip(weights1.keys(), grads1))
        fast_fc_weights1 = dict(zip(weights1.keys(),
                                [weights1[key] - self.update_lr * gradients1[key] for key in
                                 weights1.keys()]))

        loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
        grads2 = tf.gradients(loss2, list(weights2.values()))
        gradients2 = dict(zip(weights2.keys(), grads2))
        fast_fc_weights2 = dict(zip(weights2.keys(),
                                    [weights2[key] - self.update_lr * gradients2[key] for key in
                                     weights2.keys()]))

        outputb1 = self.forward_fc(inputb, fast_fc_weights1)
        outputb2 = self.forward_fc(inputb, fast_fc_weights2)
        outputb = 0.5 * (outputb1 + outputb2)
        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
        accb_list.append(accb)

        # pre-training steps
        for j in range(pretrain_steps - 1):
            maml_lossa1 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights1, dp_rate=0.0),
                                         labela)
            maml_lossa2 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights2, dp_rate=0.0),
                                         labela)
            logitsc1 = self.forward_dropout_fc(inputc1, fast_fc_weights1, dp_rate=0.0)
            logitsc2 = self.forward_dropout_fc(inputc1, fast_fc_weights2, dp_rate=0.0)
            if steps == 1:
                maml_loss_p1, maml_loss_p2 = self.my_coteaching_2(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                                feat_s=inputa, label_s=labela, fc_weights1=fast_fc_weights1,
                                                                fc_weights2=fast_fc_weights2, forget_rate=rate_schedule[j+1],
                                                                num=nums, hard=True)
            else:
                maml_loss_p1, maml_loss_p2 = self.my_coteaching(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                                forget_rate=rate_schedule[j + 1], num=nums)
                #maml_loss_p1, maml_loss_p2 = self.my_coteaching_3(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                #                                                feat_s=inputa, label_s=labela, fc_weights1=fast_fc_weights1,
                #                                                fc_weights2=fast_fc_weights2, forget_rate=rate_schedule[j+1],
                #                                                num=nums, hard=True)
            loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
            grads1 = tf.gradients(loss1, list(fast_fc_weights1.values()))
            gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
            fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                        [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                         fast_fc_weights1.keys()]))
            loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
            grads2 = tf.gradients(loss2, list(fast_fc_weights2.values()))
            gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
            fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                        [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                         fast_fc_weights2.keys()]))

            outputb1 = self.forward_fc(inputb, fast_fc_weights1)
            outputb2 = self.forward_fc(inputb, fast_fc_weights2)
            outputb = 0.5 * (outputb1 + outputb2)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        # fine-tuning steps
        for k in range(num_updates - pretrain_steps):
            maml_lossa1 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights1, dp_rate=0.0), labela)
            maml_lossa2 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights2, dp_rate=0.0), labela)
            grads1 = tf.gradients(maml_lossa1, list(fast_fc_weights1.values()))
            gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
            fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                        [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                         fast_fc_weights1.keys()]))
            grads2 = tf.gradients(maml_lossa2, list(fast_fc_weights2.values()))
            gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
            fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                        [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                         fast_fc_weights2.keys()]))

            outputb1 = self.forward_fc(inputb, fast_fc_weights1)
            outputb2 = self.forward_fc(inputb, fast_fc_weights2)
            outputb = 0.5 * (outputb1 + outputb2)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        maml_lossb = self.loss_func(outputb, labelb)
        loss_list.append(maml_lossb)

        outputc2_a = self.forward_fc(inputc2, fast_fc_weights1)
        outputc2_b = self.forward_fc(inputc2, fast_fc_weights2)
        probc2 = tf.nn.softmax(outputc2_a+outputc2_b)
        pseudo_labels_pl = tf.stop_gradient(tf.one_hot(
            tf.argmax(probc2, axis=-1),
            tf.shape(probc2)[1],
            dtype=probc2.dtype,
        ))
        acc_pl = tf.contrib.metrics.accuracy(tf.argmax(pseudo_labels_pl, 1), tf.argmax(labelc2, 1))

        featc2_new, conv_featc2_new, pseudo_labelc2 = self.sample_selection_hard_co(inputa, conv_feata, labela, inputc2,
                                                                                    conv_featc2, nums_for_hard,
                                                                                    fast_fc_weights1, fast_fc_weights2)

        relation_scores2 = self.computing_soft_weights(conv_feata, labela, conv_featc2_new, swn_weights, class_num=5,
                                                       samples_num=nums_for_hard * FLAGS.way_num,
                                                       reuse=tf.AUTO_REUSE)

        featc_output = featc2_new
        pseudo_label_output = pseudo_labelc2
        soft_weights_output = relation_scores2

        return fast_fc_weights1, fast_fc_weights2, featc_output, soft_weights_output, pseudo_label_output, accb_list, loss_list, acc_pl


    def Recurrent_Update_Stage_3(self, inputa, conv_feata, labela, inputc1, pseudo_labelc1, inputc2, conv_featc2,
                                 inputb, labelb, soft_weights, num_updates, nums_for_hard, weights1, weights2,
                                 swn_weights, pretrain_steps, steps):
        if steps == 1:
            rate = FLAGS.reject_num * 0.1
        else:
            rate = 0
        rate_schedule = np.ones(20) * rate
        rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

        accb_list = []
        loss_list = []
        logits1 = self.forward_dropout_fc(inputa, weights1, dp_rate=0.0)
        maml_lossa1 = self.loss_func(logits1, labela)
        logits2 = self.forward_dropout_fc(inputa, weights2, dp_rate=0.0)
        maml_lossa2 = self.loss_func(logits2, labela)

        logitsc1 = self.forward_dropout_fc(inputc1, weights1, dp_rate=0.0)
        logitsc2 = self.forward_dropout_fc(inputc1, weights2, dp_rate=0.0)
        nums = nums_for_hard * FLAGS.way_num

        if steps == 1:
            maml_loss_p1, maml_loss_p2 = self.my_coteaching_2(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                            feat_s=inputa, label_s=labela, fc_weights1=weights1,
                                                            fc_weights2=weights2, forget_rate=rate_schedule[0],
                                                            num=nums, hard=True)
        else:
            maml_loss_p1, maml_loss_p2 = self.my_coteaching(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                            forget_rate=rate_schedule[0],num=nums)
            #maml_loss_p1, maml_loss_p2 = self.my_coteaching_3(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                            #feat_s=inputa, label_s=labela, fc_weights1=weights1,
                                                            #fc_weights2=weights2, forget_rate=rate_schedule[0],
                                                            #num=nums, hard=True)
        loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
        grads1 = tf.gradients(loss1, list(weights1.values()))
        gradients1 = dict(zip(weights1.keys(), grads1))
        fast_fc_weights1 = dict(zip(weights1.keys(),
                                [weights1[key] - self.update_lr * gradients1[key] for key in
                                 weights1.keys()]))

        loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
        grads2 = tf.gradients(loss2, list(weights2.values()))
        gradients2 = dict(zip(weights2.keys(), grads2))
        fast_fc_weights2 = dict(zip(weights2.keys(),
                                    [weights2[key] - self.update_lr * gradients2[key] for key in
                                     weights2.keys()]))

        outputb1 = self.forward_fc(inputb, fast_fc_weights1)
        outputb2 = self.forward_fc(inputb, fast_fc_weights2)
        outputb = 0.5 * (outputb1 + outputb2)
        accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
        accb_list.append(accb)

        # pre-training steps
        for j in range(pretrain_steps - 1):
            maml_lossa1 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights1, dp_rate=0.0),
                                         labela)
            maml_lossa2 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights2, dp_rate=0.0),
                                         labela)
            logitsc1 = self.forward_dropout_fc(inputc1, fast_fc_weights1, dp_rate=0.0)
            logitsc2 = self.forward_dropout_fc(inputc1, fast_fc_weights2, dp_rate=0.0)
            if steps == 1:
                maml_loss_p1, maml_loss_p2 = self.my_coteaching_2(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                                feat_s=inputa, label_s=labela, fc_weights1=fast_fc_weights1,
                                                                fc_weights2=fast_fc_weights2, forget_rate=rate_schedule[j+1],
                                                                num=nums, hard=True)
            else:
                maml_loss_p1, maml_loss_p2 = self.my_coteaching(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                                                                forget_rate=rate_schedule[j + 1], num=nums)
                #maml_loss_p1, maml_loss_p2 = self.my_coteaching_3(logitsc1, logitsc2, soft_weights, pseudo_labelc1,
                #                                                feat_s=inputa, label_s=labela, fc_weights1=fast_fc_weights1,
                #                                                fc_weights2=fast_fc_weights2, forget_rate=rate_schedule[j+1],
                #                                                num=nums, hard=True)
            loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
            grads1 = tf.gradients(loss1, list(fast_fc_weights1.values()))
            gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
            fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                        [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                         fast_fc_weights1.keys()]))
            loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
            grads2 = tf.gradients(loss2, list(fast_fc_weights2.values()))
            gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
            fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                        [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                         fast_fc_weights2.keys()]))

            outputb1 = self.forward_fc(inputb, fast_fc_weights1)
            outputb2 = self.forward_fc(inputb, fast_fc_weights2)
            outputb = 0.5 * (outputb1 + outputb2)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        # fine-tuning steps
        for k in range(num_updates - pretrain_steps):
            maml_lossa1 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights1, dp_rate=0.0), labela)
            maml_lossa2 = self.loss_func(self.forward_dropout_fc(inputa, fast_fc_weights2, dp_rate=0.0), labela)
            grads1 = tf.gradients(maml_lossa1, list(fast_fc_weights1.values()))
            gradients1 = dict(zip(fast_fc_weights1.keys(), grads1))
            fast_fc_weights1 = dict(zip(fast_fc_weights1.keys(),
                                        [fast_fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                         fast_fc_weights1.keys()]))
            grads2 = tf.gradients(maml_lossa2, list(fast_fc_weights2.values()))
            gradients2 = dict(zip(fast_fc_weights2.keys(), grads2))
            fast_fc_weights2 = dict(zip(fast_fc_weights2.keys(),
                                        [fast_fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                         fast_fc_weights2.keys()]))

            outputb1 = self.forward_fc(inputb, fast_fc_weights1)
            outputb2 = self.forward_fc(inputb, fast_fc_weights2)
            outputb = 0.5 * (outputb1 + outputb2)
            accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
            accb_list.append(accb)
            maml_lossb = self.loss_func(outputb, labelb)
            loss_list.append(maml_lossb)

        maml_lossb = self.loss_func(outputb, labelb)
        loss_list.append(maml_lossb)

        featc2_new, conv_featc2_new, pseudo_labelc2 = self.sample_selection_hard_co(inputa, conv_feata, labela, inputc2,
                                                                                    conv_featc2, nums_for_hard,
                                                                                    fast_fc_weights1, fast_fc_weights2)

        relation_scores2 = self.computing_soft_weights(conv_feata, labela, conv_featc2_new, swn_weights, class_num=5,
                                                       samples_num=nums_for_hard * FLAGS.way_num,
                                                       reuse=tf.AUTO_REUSE)

        featc_output = featc2_new
        pseudo_label_output = pseudo_labelc2
        soft_weights_output = relation_scores2

        return fast_fc_weights1, fast_fc_weights2, featc_output, soft_weights_output, pseudo_label_output, accb_list, loss_list




