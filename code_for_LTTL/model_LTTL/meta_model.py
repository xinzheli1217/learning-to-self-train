import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
from model_analysis import Analysis

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class MetaModel(Analysis):

    def construct_train_model_LTTL(self, prefix='metatrain_'):

        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.inputc = tf.placeholder(tf.float32)
        self.labelc = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-model', reuse=False) as training_scope:
            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights_pl = fc_weights_pl = self.construct_fc_weights()
            self.swn_weights = swn_weights = self.construct_swn_weights()
            self.fc_weights1 = fc_weights1 = self.construct_fc_weights_coteaching(seed=100, str='1')
            self.fc_weights2 = fc_weights2 = self.construct_fc_weights_coteaching(seed=200, str='2')

            num_updates = FLAGS.train_base_epoch_num

            rate = FLAGS.reject_num * 0.1
            rate_schedule = np.ones(20) * rate
            rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, labelc = inp

                lossb_list = []

                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)
                emb_outputc, conv_featc = self.forward_resnet_2(inputc, weights, ss_weights, reuse=True)
                emb_outputc = tf.stop_gradient(emb_outputc)
                conv_featc = tf.stop_gradient(conv_featc)
                emb_outputb, _ = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                # Main_pretrain_parts
                pseudo_labels = self.computing_pseudo_labels(emb_outputa, labela, emb_outputc, 10, fc_weights_pl, one_hot=True)

                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_featc, swn_weights, class_num=5,
                                                           samples_num=FLAGS.nb_ul_samples * FLAGS.way_num,
                                                           reuse=tf.AUTO_REUSE)
                soft_weights = tf.reshape(soft_weights, [-1, 5])
                logits1 = self.forward_dropout_fc(emb_outputa, fc_weights1, dp_rate=0.2)
                maml_lossa1 = self.loss_func(logits1, labela)
                logits2 = self.forward_dropout_fc(emb_outputa, fc_weights2, dp_rate=0.2)
                maml_lossa2 = self.loss_func(logits2, labela)

                logitsc1 = self.forward_dropout_fc(emb_outputc, fc_weights1, dp_rate=0.2)
                logitsc2 = self.forward_dropout_fc(emb_outputc, fc_weights2, dp_rate=0.2)
                nums = FLAGS.nb_ul_samples * FLAGS.way_num
                maml_loss_p1, maml_loss_p2 = self.my_coteaching(logitsc1, logitsc2, soft_weights, pseudo_labels,
                                                                forget_rate=rate_schedule[0], num=nums)

                loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
                grads1 = tf.gradients(loss1, list(fc_weights1.values()))
                gradients1 = dict(zip(fc_weights1.keys(), grads1))
                fast_fc_weights1 = dict(zip(fc_weights1.keys(),
                                            [fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                             fc_weights1.keys()]))

                loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
                grads2 = tf.gradients(loss2, list(fc_weights2.values()))
                gradients2 = dict(zip(fc_weights2.keys(), grads2))
                fast_fc_weights2 = dict(zip(fc_weights2.keys(),
                                            [fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                             fc_weights2.keys()]))

                # pre-training steps
                for j in range(FLAGS.pre_train_epoch_num - 1):
                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=0.2),
                                                 labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=0.2),
                                                 labela)
                    logitsc1 = self.forward_dropout_fc(emb_outputc, fast_fc_weights1, dp_rate=0.2)
                    logitsc2 = self.forward_dropout_fc(emb_outputc, fast_fc_weights2, dp_rate=0.2)
                    maml_loss_p1, maml_loss_p2 = self.my_coteaching(logitsc1, logitsc2, soft_weights, pseudo_labels,
                                                                    forget_rate=rate_schedule[j + 1], num=nums)
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

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)

                # fine-tuning steps
                for k in range(num_updates - FLAGS.pre_train_epoch_num):
                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=0.2), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=0.2), labela)
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
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [lossb_list, accb]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0],
                                         self.labelc[0]), False)

            out_dtype = [[tf.float32] * 2, tf.float32]

            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb_list, accsb = result

            self.final_pretrain_loss = final_pretrain_loss = tf.reduce_sum(lossesb_list[-2]) / tf.to_float(FLAGS.meta_batch_size)
            self.total_loss = total_loss = tf.reduce_sum(lossesb_list[-1]) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
            meta_optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.metatrain_op = meta_optimizer.minimize(total_loss, var_list=ss_weights.values()+fc_weights1.values()+fc_weights2.values())
            swn_optimizer = tf.train.AdamOptimizer(self.swn_lr)
            self.meta_swn_train_op = swn_optimizer.minimize(final_pretrain_loss, var_list=swn_weights.values())
            tf.summary.scalar(prefix + 'Final Loss', total_loss)
            tf.summary.scalar(prefix + 'Final Pretrain Loss', final_pretrain_loss)
            tf.summary.scalar(prefix + 'Accuracy', total_accuracy)


    def construct_model_test_LTTL(self, prefix='metaval_'):
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

            rate = FLAGS.reject_num * 0.1
            rate_schedule = np.ones(20) * rate
            rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, labelc = inp

                lossb_list = []
                accb_list = []
                accb1_list = []
                accb2_list = []
                #For analysis
                r1_list = []
                r2_list = []
                p=0.0
                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)
                emb_outputc, conv_featc = self.forward_resnet_2(inputc, weights, ss_weights, reuse=True)
                emb_outputc = tf.stop_gradient(emb_outputc)
                conv_featc = tf.stop_gradient(conv_featc)

                pseudo_labels = self.computing_pseudo_labels(emb_outputa, labela, emb_outputc, 10, fc_weights_pl, one_hot=True)

                #For analysis
                g_i = tf.argmax(labelc, axis=-1)
                p_i = tf.argmax(pseudo_labels, axis=-1)
                n_eq = tf.cast(tf.not_equal(g_i, p_i),dtype=tf.int64)
                n_eq_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)

                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_featc, swn_weights, class_num=5,
                                                           samples_num= FLAGS.nb_ul_samples * FLAGS.way_num, reuse=reuse)

                soft_weights = tf.reshape(soft_weights, [-1,5])
                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                outputa1 = self.forward_dropout_fc(emb_outputa, fc_weights1, dp_rate=p)
                maml_lossa1 = self.loss_func(outputa1, labela)
                outputa2 = self.forward_dropout_fc(emb_outputa, fc_weights2, dp_rate=p)
                maml_lossa2 = self.loss_func(outputa2, labela)

                logitsc1 = self.forward_dropout_fc(emb_outputc, fc_weights1, dp_rate=p)
                logitsc2 = self.forward_dropout_fc(emb_outputc, fc_weights2, dp_rate=p)
                #For analysis
                nums = FLAGS.nb_ul_samples * FLAGS.way_num
                maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logitsc1, logitsc2, soft_weights, pseudo_labels,
                                                                                  feat_s=emb_outputa, label_s=labela, fc_weights1=fc_weights1,
                                                                                  fc_weights2=fc_weights2, n_eq=n_eq, forget_rate=rate_schedule[0], num=nums)
                r1_list.append(r1)
                r2_list.append(r2)
                loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
                grads1 = tf.gradients(loss1, list(fc_weights1.values()))
                gradients1 = dict(zip(fc_weights1.keys(), grads1))
                fast_fc_weights1 = dict(zip(fc_weights1.keys(),
                                           [fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                            fc_weights1.keys()]))

                loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
                grads2 = tf.gradients(loss2, list(fc_weights2.values()))
                gradients2 = dict(zip(fc_weights2.keys(), grads2))
                fast_fc_weights2 = dict(zip(fc_weights2.keys(),
                                           [fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                            fc_weights2.keys()]))

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)
                accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                accb1_list.append(accb1)
                accb2_list.append(accb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                accb_list.append(accb)

                #pre-training steps
                for j in range(FLAGS.pre_train_epoch_num - 1):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
                    logitsc1 = self.forward_dropout_fc(emb_outputc, fast_fc_weights1, dp_rate=p)
                    logitsc2 = self.forward_dropout_fc(emb_outputc, fast_fc_weights2, dp_rate=p)
                    # For analysis
                    maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logitsc1, logitsc2, soft_weights, pseudo_labels,
                                                                                      feat_s=emb_outputa, label_s=labela, fc_weights1=fast_fc_weights1,
                                                                                      fc_weights2=fast_fc_weights2, n_eq=n_eq, forget_rate=rate_schedule[j+1], num=nums)
                    r1_list.append(r1)
                    r2_list.append(r2)
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

                    outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                    outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                    outputb = 0.5 * (outputb1 + outputb2)
                    maml_lossb = self.loss_func(outputb, labelb)
                    lossb_list.append(maml_lossb)
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)

                #fine-tuning steps
                for k in range(num_updates - FLAGS.pre_train_epoch_num):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
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
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [lossb_list, accb, accb_list, accb1_list, accb2_list, r1_list, r2_list, n_eq_sum]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0]), False)

            out_dtype = [[tf.float32]*FLAGS.pre_train_epoch_num, tf.float32, [tf.float32] * num_updates, [tf.float32] * num_updates, [tf.float32] * num_updates, [tf.float32]*FLAGS.pre_train_epoch_num, [tf.float32]*FLAGS.pre_train_epoch_num, tf.float32]

            result = tf.map_fn(task_metalearn,
                                elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc),
                                dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb_list, accsb, accsb_list, accsb1_list, accsb2_list, ratio1_list, ratio2_list, n_sum = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies1 = total_accuracies1 = [tf.reduce_sum(accsb1_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accsb2_list[j]) for j in
                                                                range(num_updates)]
            self.ratios_1 = [tf.reduce_sum(ratio1_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.ratios_2 = [tf.reduce_sum(ratio2_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.n_sum = tf.reduce_sum(n_sum)


    def construct_model_test_LTTL_hard(self, prefix='metaval_'):
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

            rate = FLAGS.reject_num * 0.1
            rate_schedule = np.ones(20) * rate
            rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, labelc = inp

                lossb_list = []
                accb_list = []
                accb1_list = []
                accb2_list = []
                #For analysis
                r1_list = []
                r2_list = []
                p=0.0
                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)
                emb_outputc, conv_featc = self.forward_resnet_2(inputc, weights, ss_weights, reuse=True)
                emb_outputc = tf.stop_gradient(emb_outputc)
                conv_featc = tf.stop_gradient(conv_featc)

                emb_unlabeled, conv_unlabeled, pseudo_labels, n_eq = self.computing_pl_for_test_analysis(emb_outputa, conv_feata,
                                                                                          labela, emb_outputc,
                                                                                          conv_featc, labelc, 10,
                                                                                          nums_for_hard=FLAGS.hard_selection,
                                                                                          weights=fc_weights_pl, d=False)
                n_eq_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)
                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_unlabeled, swn_weights, class_num=5,
                                                           samples_num= FLAGS.hard_selection * FLAGS.way_num, reuse=reuse)

                soft_weights = tf.reshape(soft_weights, [-1,5])
                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                outputa1 = self.forward_dropout_fc(emb_outputa, fc_weights1, dp_rate=p)
                maml_lossa1 = self.loss_func(outputa1, labela)
                outputa2 = self.forward_dropout_fc(emb_outputa, fc_weights2, dp_rate=p)
                maml_lossa2 = self.loss_func(outputa2, labela)

                logits_un_1 = self.forward_dropout_fc(emb_unlabeled, fc_weights1, dp_rate=p)
                logits_un_2 = self.forward_dropout_fc(emb_unlabeled, fc_weights2, dp_rate=p)
                nums = FLAGS.hard_selection * FLAGS.way_num
                maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logits_un_1, logits_un_2, soft_weights, pseudo_labels,
                                                                                  feat_s=emb_outputa, label_s=labela, fc_weights1=fc_weights1,
                                                                                  fc_weights2=fc_weights2, n_eq=n_eq,
                                                                                  forget_rate=rate_schedule[0], num=nums, hard=True)
                r1_list.append(r1)
                r2_list.append(r2)
                loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
                grads1 = tf.gradients(loss1, list(fc_weights1.values()))
                gradients1 = dict(zip(fc_weights1.keys(), grads1))
                fast_fc_weights1 = dict(zip(fc_weights1.keys(),
                                           [fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                            fc_weights1.keys()]))

                loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
                grads2 = tf.gradients(loss2, list(fc_weights2.values()))
                gradients2 = dict(zip(fc_weights2.keys(), grads2))
                fast_fc_weights2 = dict(zip(fc_weights2.keys(),
                                           [fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                            fc_weights2.keys()]))

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)
                accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                accb1_list.append(accb1)
                accb2_list.append(accb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                accb_list.append(accb)

                #pre-training steps
                for j in range(FLAGS.pre_train_epoch_num - 1):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
                    logits_un_1 = self.forward_dropout_fc(emb_unlabeled, fast_fc_weights1, dp_rate=p)
                    logits_un_2 = self.forward_dropout_fc(emb_unlabeled, fast_fc_weights2, dp_rate=p)
                    maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logits_un_1, logits_un_2, soft_weights, pseudo_labels,
                                                                                      feat_s = emb_outputa, label_s=labela,
                                                                                      fc_weights1=fast_fc_weights1, fc_weights2=fast_fc_weights2,
                                                                                      n_eq=n_eq, forget_rate=rate_schedule[j+1], num=nums, hard=True)
                    r1_list.append(r1)
                    r2_list.append(r2)
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

                    outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                    outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                    outputb = 0.5 * (outputb1 + outputb2)
                    maml_lossb = self.loss_func(outputb, labelb)
                    lossb_list.append(maml_lossb)
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                #fine-tuning steps
                for k in range(num_updates - FLAGS.pre_train_epoch_num):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
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
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [lossb_list, accb, accb_list, accb1_list, accb2_list, r1_list, r2_list, n_eq_sum]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0]), False)

            out_dtype = [[tf.float32] * FLAGS.pre_train_epoch_num, tf.float32, [tf.float32] * num_updates,
                         [tf.float32] * num_updates,
                         [tf.float32] * num_updates, [tf.float32] * FLAGS.pre_train_epoch_num,
                         [tf.float32] * FLAGS.pre_train_epoch_num, tf.float32]

            result = tf.map_fn(task_metalearn,
                                elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc),
                                dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb_list, accsb, accsb_list, accsb1_list, accsb2_list, ratio1_list, ratio2_list, n_sum = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies1 = total_accuracies1 = [tf.reduce_sum(accsb1_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accsb2_list[j]) for j in
                                                                range(num_updates)]
            self.ratios_1 = [tf.reduce_sum(ratio1_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.ratios_2 = [tf.reduce_sum(ratio2_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.n_sum = tf.reduce_sum(n_sum)


    def construct_model_test_LTTL_distractors(self, prefix='metaval_'):
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

            rate = FLAGS.reject_num * 0.1
            rate_schedule = np.ones(20) * rate
            rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, labelc, inputd = inp

                lossb_list = []
                accb_list = []
                accb1_list = []
                accb2_list = []
                #For analysis
                r1_list = []
                r2_list = []
                p=0.0
                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)
                emb_outputc, conv_featc = self.forward_resnet_2(inputc, weights, ss_weights, reuse=True)
                emb_outputc = tf.stop_gradient(emb_outputc)
                conv_featc = tf.stop_gradient(conv_featc)
                emb_outputd, conv_featd = self.forward_resnet_2(inputd, weights, ss_weights, reuse=True)
                emb_outputd = tf.stop_gradient(emb_outputd)
                conv_featd = tf.stop_gradient(conv_featd)

                emb_unlabeled = tf.concat([emb_outputc, emb_outputd], axis=0)
                conv_unlabeled = tf.concat([conv_featc, conv_featd], axis=0)

                pseudo_labels = self.computing_pseudo_labels(emb_outputa, labela, emb_unlabeled, 10, fc_weights_pl, one_hot=True)

                #For analysis
                c_i = tf.argmax(labelc, axis=-1)
                f_i = -1 * tf.ones((FLAGS.nb_ul_samples * FLAGS.num_dis,), dtype=tf.int64)
                g_i = tf.concat([c_i, f_i], axis=0)
                p_i = tf.argmax(pseudo_labels, axis=-1)
                n_eq = tf.cast(tf.not_equal(g_i, p_i),dtype=tf.int64)

                n_eq_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)

                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_unlabeled, swn_weights, class_num=5,
                                                           samples_num= FLAGS.nb_ul_samples * (FLAGS.way_num + FLAGS.num_dis), reuse=reuse)

                soft_weights = tf.reshape(soft_weights, [-1,5])
                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                outputa1 = self.forward_dropout_fc(emb_outputa, fc_weights1, dp_rate=p)
                maml_lossa1 = self.loss_func(outputa1, labela)
                outputa2 = self.forward_dropout_fc(emb_outputa, fc_weights2, dp_rate=p)
                maml_lossa2 = self.loss_func(outputa2, labela)

                logits_un_1 = self.forward_dropout_fc(emb_unlabeled, fc_weights1, dp_rate=p)
                logits_un_2 = self.forward_dropout_fc(emb_unlabeled, fc_weights2, dp_rate=p)
                #For analysis
                nums = FLAGS.nb_ul_samples * (FLAGS.way_num + FLAGS.num_dis)
                maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logits_un_1, logits_un_2, soft_weights, pseudo_labels,
                                                                                  feat_s=emb_outputa, label_s=labela, fc_weights1=fc_weights1,
                                                                                  fc_weights2=fc_weights2, n_eq=n_eq,
                                                                                  forget_rate=rate_schedule[0], num=nums)
                r1_list.append(r1)
                r2_list.append(r2)
                loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
                grads1 = tf.gradients(loss1, list(fc_weights1.values()))
                gradients1 = dict(zip(fc_weights1.keys(), grads1))
                fast_fc_weights1 = dict(zip(fc_weights1.keys(),
                                           [fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                            fc_weights1.keys()]))

                loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
                grads2 = tf.gradients(loss2, list(fc_weights2.values()))
                gradients2 = dict(zip(fc_weights2.keys(), grads2))
                fast_fc_weights2 = dict(zip(fc_weights2.keys(),
                                           [fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                            fc_weights2.keys()]))

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)
                accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                accb1_list.append(accb1)
                accb2_list.append(accb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                accb_list.append(accb)

                #pre-training steps
                for j in range(FLAGS.pre_train_epoch_num - 1):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
                    logits_un_1 = self.forward_dropout_fc(emb_unlabeled, fast_fc_weights1, dp_rate=p)
                    logits_un_2 = self.forward_dropout_fc(emb_unlabeled, fast_fc_weights2, dp_rate=p)
                    # For analysis
                    maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logits_un_1, logits_un_2, soft_weights, pseudo_labels,
                                                                                      feat_s=emb_outputa, label_s=labela, fc_weights1=fast_fc_weights1,
                                                                                      fc_weights2=fast_fc_weights2, n_eq=n_eq, forget_rate=rate_schedule[j+1],num=nums)
                    r1_list.append(r1)
                    r2_list.append(r2)
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

                    outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                    outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                    outputb = 0.5 * (outputb1 + outputb2)
                    maml_lossb = self.loss_func(outputb, labelb)
                    lossb_list.append(maml_lossb)
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                #fine-tuning steps
                for k in range(num_updates - FLAGS.pre_train_epoch_num):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
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
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [lossb_list, accb, accb_list, accb1_list, accb2_list, r1_list, r2_list, n_eq_sum]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0], self.inputd[0]), False)

            out_dtype = [[tf.float32] * FLAGS.pre_train_epoch_num, tf.float32, [tf.float32] * num_updates,
                         [tf.float32] * num_updates,
                         [tf.float32] * num_updates, [tf.float32] * FLAGS.pre_train_epoch_num,
                         [tf.float32] * FLAGS.pre_train_epoch_num, tf.float32]

            result = tf.map_fn(task_metalearn,
                                elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc, self.inputd),
                                dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb_list, accsb, accsb_list, accsb1_list, accsb2_list, ratio1_list, ratio2_list, n_sum = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies1 = total_accuracies1 = [tf.reduce_sum(accsb1_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accsb2_list[j]) for j in
                                                                range(num_updates)]
            self.ratios_1 = [tf.reduce_sum(ratio1_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.ratios_2 = [tf.reduce_sum(ratio2_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.n_sum = tf.reduce_sum(n_sum)


    def construct_model_test_LTTL_distractors_hard(self, prefix='metaval_'):
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

            rate = FLAGS.reject_num * 0.1
            rate_schedule = np.ones(20) * rate
            rate_schedule[:5] = np.linspace(0, rate ** 1, 5)

            def task_metalearn(inp, reuse=True):

                """ Perform gradient descent for one task in the meta-batch. """

                inputa, inputb, labela, labelb, inputc, labelc, inputd = inp

                lossb_list = []
                accb_list = []
                accb1_list = []
                accb2_list = []
                #For analysis
                r1_list = []
                r2_list = []
                p=0.0
                emb_outputa, conv_feata = self.forward_resnet_2(inputa, weights, ss_weights, reuse=reuse)
                emb_outputa = tf.stop_gradient(emb_outputa)
                conv_feata = tf.stop_gradient(conv_feata)
                emb_outputc, conv_featc = self.forward_resnet_2(inputc, weights, ss_weights, reuse=True)
                emb_outputc = tf.stop_gradient(emb_outputc)
                conv_featc = tf.stop_gradient(conv_featc)
                emb_outputd, conv_featd = self.forward_resnet_2(inputd, weights, ss_weights, reuse=True)
                emb_outputd = tf.stop_gradient(emb_outputd)
                conv_featd = tf.stop_gradient(conv_featd)

                emb_unlabeled_pre = tf.concat([emb_outputc, emb_outputd], axis=0)
                conv_unlabeled_pre = tf.concat([conv_featc, conv_featd], axis=0)

                emb_unlabeled, conv_unlabeled, pseudo_labels, n_eq = self.computing_pl_for_test_analysis(emb_outputa, conv_feata,
                                                                                          labela, emb_unlabeled_pre,
                                                                                          conv_unlabeled_pre, labelc, 10,
                                                                                          nums_for_hard=FLAGS.hard_selection,
                                                                                          weights=fc_weights_pl, d=True)
                n_eq_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)
                soft_weights = self.computing_soft_weights(conv_feata, labela, conv_unlabeled, swn_weights, class_num=5,
                                                           samples_num= FLAGS.hard_selection * FLAGS.way_num, reuse=reuse)

                soft_weights = tf.reshape(soft_weights, [-1,5])
                emb_outputb, conv_featb = self.forward_resnet_2(inputb, weights, ss_weights, reuse=True)

                outputa1 = self.forward_dropout_fc(emb_outputa, fc_weights1, dp_rate=p)
                maml_lossa1 = self.loss_func(outputa1, labela)
                outputa2 = self.forward_dropout_fc(emb_outputa, fc_weights2, dp_rate=p)
                maml_lossa2 = self.loss_func(outputa2, labela)

                logits_un_1 = self.forward_dropout_fc(emb_unlabeled, fc_weights1, dp_rate=p)
                logits_un_2 = self.forward_dropout_fc(emb_unlabeled, fc_weights2, dp_rate=p)

                nums = FLAGS.hard_selection * FLAGS.way_num
                maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logits_un_1, logits_un_2, soft_weights, pseudo_labels,
                                                                                  feat_s=emb_outputa, label_s=labela, fc_weights1=fc_weights1,
                                                                                  fc_weights2=fc_weights2, n_eq=n_eq,
                                                                                  forget_rate=rate_schedule[0], num=nums, hard=True)
                r1_list.append(r1)
                r2_list.append(r2)
                loss1 = tf.concat([maml_lossa1, maml_loss_p1], axis=0)
                grads1 = tf.gradients(loss1, list(fc_weights1.values()))
                gradients1 = dict(zip(fc_weights1.keys(), grads1))
                fast_fc_weights1 = dict(zip(fc_weights1.keys(),
                                           [fc_weights1[key] - self.update_lr * gradients1[key] for key in
                                            fc_weights1.keys()]))

                loss2 = tf.concat([maml_lossa2, maml_loss_p2], axis=0)
                grads2 = tf.gradients(loss2, list(fc_weights2.values()))
                gradients2 = dict(zip(fc_weights2.keys(), grads2))
                fast_fc_weights2 = dict(zip(fc_weights2.keys(),
                                           [fc_weights2[key] - self.update_lr * gradients2[key] for key in
                                            fc_weights2.keys()]))

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                maml_lossb = self.loss_func(outputb, labelb)
                lossb_list.append(maml_lossb)
                accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                accb1_list.append(accb1)
                accb2_list.append(accb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                accb_list.append(accb)

                #pre-training steps
                for j in range(FLAGS.pre_train_epoch_num - 1):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
                    logits_un_1 = self.forward_dropout_fc(emb_unlabeled, fast_fc_weights1, dp_rate=p)
                    logits_un_2 = self.forward_dropout_fc(emb_unlabeled, fast_fc_weights2, dp_rate=p)
                    maml_loss_p1, maml_loss_p2, r1, r2 = self.my_coteaching_analysis2(logits_un_1, logits_un_2, soft_weights, pseudo_labels,
                                                                                      feat_s=emb_outputa, label_s=labela, fc_weights1=fast_fc_weights1,
                                                                                      fc_weights2=fast_fc_weights2, n_eq=n_eq,
                                                                                      forget_rate=rate_schedule[j+1], num=nums, hard=True)
                    r1_list.append(r1)
                    r2_list.append(r2)
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

                    outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                    outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                    outputb = 0.5 * (outputb1 + outputb2)
                    maml_lossb = self.loss_func(outputb, labelb)
                    lossb_list.append(maml_lossb)
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                #fine-tuning steps
                for k in range(num_updates - FLAGS.pre_train_epoch_num):

                    maml_lossa1 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights1, dp_rate=p), labela)
                    maml_lossa2 = self.loss_func(self.forward_dropout_fc(emb_outputa, fast_fc_weights2, dp_rate=p), labela)
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
                    accb1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb1), 1), tf.argmax(labelb, 1))
                    accb2 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb2), 1), tf.argmax(labelb, 1))
                    accb1_list.append(accb1)
                    accb2_list.append(accb2)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                outputb1 = self.forward_fc(emb_outputb, fast_fc_weights1)
                outputb2 = self.forward_fc(emb_outputb, fast_fc_weights2)
                outputb = 0.5 * (outputb1 + outputb2)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [lossb_list, accb, accb_list, accb1_list, accb2_list, r1_list, r2_list, n_eq_sum]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0], self.inputc[0], self.labelc[0], self.inputd[0]), False)

            out_dtype = [[tf.float32] * FLAGS.pre_train_epoch_num, tf.float32, [tf.float32] * num_updates,
                         [tf.float32] * num_updates,
                         [tf.float32] * num_updates, [tf.float32] * FLAGS.pre_train_epoch_num,
                         [tf.float32] * FLAGS.pre_train_epoch_num, tf.float32]

            result = tf.map_fn(task_metalearn,
                                elems=(self.inputa, self.inputb, self.labela, self.labelb, self.inputc, self.labelc, self.inputd),
                                dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            lossesb_list, accsb, accsb_list, accsb1_list, accsb2_list, ratio1_list, ratio2_list, n_sum = result

            self.metaval_total_accuracies = total_accuracies = [tf.reduce_sum(accsb_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies1 = total_accuracies1 = [tf.reduce_sum(accsb1_list[j]) for j in
                                                                range(num_updates)]
            self.metaval_total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accsb2_list[j]) for j in
                                                                range(num_updates)]
            self.ratios_1 = [tf.reduce_sum(ratio1_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.ratios_2 = [tf.reduce_sum(ratio2_list[j]) for j in range(FLAGS.pre_train_epoch_num)]
            self.n_sum = tf.reduce_sum(n_sum)


