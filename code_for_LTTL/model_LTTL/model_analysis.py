import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
from model_LTTL.models import Models

from tensorflow.python.platform import flags
from misc import resnet_conv_block, leaky_relu, mse
from model_LTTL.models_LTT import Model_LTT
FLAGS = flags.FLAGS

class Analysis(Model_LTT):
    def __init__(self):
        Models.__init__(self)
        self.loss_func_rec = mse

    def computing_pseudo_labels_co(self, feat_1, label_1, feat_2, num_steps, weights_a, weights_b, one_hot):

        output_1_a = self.forward_fc(feat_1, weights_a)
        loss_1_a = self.loss_func(output_1_a, label_1)
        grads_a = tf.gradients(loss_1_a, list(weights_a.values()))
        gradients_a = dict(zip(weights_a.keys(), grads_a))
        fast_weights_a = dict(zip(weights_a.keys(),
                                [weights_a[key] - self.update_lr * gradients_a[key] for key in
                                 weights_a.keys()]))
        output_1_b = self.forward_fc(feat_1, weights_b)
        loss_1_b = self.loss_func(output_1_b, label_1)
        grads_b = tf.gradients(loss_1_b, list(weights_b.values()))
        gradients_b = dict(zip(weights_b.keys(), grads_b))
        fast_weights_b = dict(zip(weights_b.keys(),
                                  [weights_b[key] - self.update_lr * gradients_b[key] for key in
                                   weights_b.keys()]))

        for i in range(num_steps - 1):
            loss_a = self.loss_func(self.forward_fc(feat_1, fast_weights_a), label_1)
            grads_a = tf.gradients(loss_a, list(fast_weights_a.values()))
            gradients_a = dict(zip(fast_weights_a.keys(), grads_a))
            fast_weights_a = dict(zip(fast_weights_a.keys(),
                                    [fast_weights_a[key] - self.update_lr * gradients_a[key] for key in
                                     fast_weights_a.keys()]))
            loss_b = self.loss_func(self.forward_fc(feat_1, fast_weights_b), label_1)
            grads_b = tf.gradients(loss_b, list(fast_weights_b.values()))
            gradients_b = dict(zip(fast_weights_b.keys(), grads_b))
            fast_weights_b = dict(zip(fast_weights_b.keys(),
                                    [fast_weights_b[key] - self.update_lr * gradients_b[key] for key in
                                     fast_weights_b.keys()]))

        output_2_a = self.forward_fc(feat_2, fast_weights_a)
        output_2_b = self.forward_fc(feat_2, fast_weights_b)
        probc = tf.nn.softmax(output_2_a+output_2_b)
        pseudo_labels = tf.stop_gradient(tf.one_hot(
            tf.argmax(probc, axis=-1),
            tf.shape(probc)[1],
            dtype=probc.dtype,
        ))

        if one_hot:
            y2 = pseudo_labels
        else:
            y2 = tf.stop_gradient(probc)

        return y2


    def computing_pl_for_test_analysis(self, feat_s, conv_feat_s, label_s, feat_un, conv_feat_un, label_un, num_steps, nums_for_hard,
                                       weights, d=False):
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

        feats_new, conv_feat_new, pseudo_labels, n_eq = self.sample_selection_hard_analysis(feat_s, conv_feat_s, label_s, feat_un,
                                                                                            conv_feat_un, label_un, nums_for_hard,
                                                                                            fast_weights, d=d)

        return feats_new, conv_feat_new, pseudo_labels, n_eq


    def computing_pl_for_test__analysis_co(self, feat_s, conv_feat_s, label_s, feat_un, conv_feat_un, label_un, num_steps, nums_for_hard,
                                           weights1, weights2, d=False):
        '''
        Computing pseudo labels and applying hard-selection for unlabeled samples during meta-testing.
        '''

        output_s_1 = self.forward_fc(feat_s, weights1)
        loss_s_1 = self.loss_func(output_s_1, label_s)
        output_s_2 = self.forward_fc(feat_s, weights2)
        loss_s_2 = self.loss_func(output_s_2, label_s)

        grads1 = tf.gradients(loss_s_1, list(weights1.values()))
        gradients1 = dict(zip(weights1.keys(), grads1))
        fast_weights1 = dict(zip(weights1.keys(),
                                 [weights1[key] - self.update_lr * gradients1[key] for key in
                                  weights1.keys()]))
        grads2 = tf.gradients(loss_s_2, list(weights2.values()))
        gradients2 = dict(zip(weights2.keys(), grads2))
        fast_weights2 = dict(zip(weights2.keys(),
                                 [weights2[key] - self.update_lr * gradients2[key] for key in
                                  weights2.keys()]))

        for i in range(num_steps - 1):
            loss1 = self.loss_func(self.forward_fc(feat_s, fast_weights1), label_s)
            loss2 = self.loss_func(self.forward_fc(feat_s, fast_weights2), label_s)

            grads1 = tf.gradients(loss1, list(fast_weights1.values()))
            gradients1 = dict(zip(fast_weights1.keys(), grads1))
            fast_weights1 = dict(zip(fast_weights1.keys(),
                                     [fast_weights1[key] - self.update_lr * gradients1[key] for key in
                                      fast_weights1.keys()]))
            grads2 = tf.gradients(loss2, list(fast_weights2.values()))
            gradients2 = dict(zip(fast_weights2.keys(), grads2))
            fast_weights2 = dict(zip(fast_weights2.keys(),
                                     [fast_weights2[key] - self.update_lr * gradients2[key] for key in
                                      fast_weights2.keys()]))

        feats_new, conv_feat_new, pseudo_labels, n_eq = self.sample_selection_hard_analysis_co(feat_s, conv_feat_s, label_s, feat_un,
                                                                                               conv_feat_un, label_un, nums_for_hard,
                                                                                               fast_weights1, fast_weights2, d=d)


        return feats_new, conv_feat_new, pseudo_labels


    def sample_selection_hard_analysis_co(self, inputs_proto, conv_inputs_proto, labels_proto, inputs, conv_inputs, labels_g, nums_for_hard,
                                          weights1, weights2, d):
        '''
        Selecting unlabeled samples with high prediction scores.
        '''

        #pseudo labeling for unlabeled samples
        logits1 = self.forward_fc(inputs, weights1)
        logits2 = self.forward_fc(inputs, weights2)
        logits = logits1 + logits2
        probs = tf.nn.softmax(logits)
        labels = tf.argmax(probs, axis=-1)

        if d:
            f_i = -1 * tf.ones((FLAGS.nb_ul_samples * FLAGS.num_dis,), dtype=tf.int64)
            g_i = tf.argmax(labels_g, axis=-1)
            l_f = tf.concat([g_i, f_i], axis=0)
            n_eq = tf.cast(tf.not_equal(l_f, labels), dtype=tf.int64)
        else:
            g_i = tf.argmax(labels_g, axis=-1)
            n_eq = tf.cast(tf.not_equal(g_i, labels), dtype=tf.int64)

        labels_num_proto = tf.argmax(labels_proto, axis=-1)
        new_samples = []
        new_conv_samples = []
        new_labels = []
        new_n_eqs = []

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
            #selecting n_eq
            n_eq_i = tf.gather_nd(n_eq, index_sample_i)
            n_eq_proto_i = tf.zeros((FLAGS.shot_num,), dtype=tf.int64)
            new_n_eq_i = tf.concat([n_eq_i, n_eq_proto_i], axis=0)
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

            def repeat_and_rank(samples, conv_samples, labels, n_eq, probs, k):
                list_samples = [samples for i in range(k)]
                list_conv_samples = [conv_samples for i in range(k)]
                list_labels = [labels for i in range(k)]
                list_n_eq = [n_eq for i in range(k)]
                list_probs = [probs]
                list_probs.extend([tf.zeros_like(probs, dtype=probs.dtype) for i in range(k - 1)])

                samples_pad = tf.concat(list_samples, axis=0)
                conv_samples_pad = tf.concat(list_conv_samples, axis=0)
                labels_pad = tf.concat(list_labels, axis=0)
                probs_pad = tf.concat(list_probs, axis=0)
                n_eq_pad = tf.concat(list_n_eq, axis=0)

                index_rank = tf.reshape(tf.nn.top_k(probs_pad, k=k).indices, [k, 1])
                samples_rank = tf.gather_nd(samples_pad, index_rank)
                conv_samples_rank = tf.gather_nd(conv_samples_pad, index_rank)
                labels_rank = tf.gather_nd(labels_pad, index_rank)
                n_eq_rank = tf.gather_nd(n_eq_pad, index_rank)

                return samples_rank, conv_samples_rank, labels_rank, n_eq_rank

            def normal_rank(samples, conv_samples, labels, n_eq, probs, k):
                index_rank = tf.reshape(tf.nn.top_k(probs, k=k).indices, [k, 1])
                samples_rank = tf.gather_nd(samples, index_rank)
                conv_samples_rank = tf.gather_nd(conv_samples, index_rank)
                labels_rank = tf.gather_nd(labels, index_rank)
                n_eq_rank = tf.gather_nd(n_eq, index_rank)

                return samples_rank, conv_samples_rank, labels_rank, n_eq_rank

            samples_rank_i, conv_samples_rank_i, labels_rank_i, n_eq_rank_i = tf.cond(
                num_total < tf.constant(nums_for_hard, dtype=tf.int16),
                lambda: repeat_and_rank(new_samples_i,
                                        new_conv_samples_i,
                                        new_labels_i, new_n_eq_i, new_probs_i, nums_for_hard),
                lambda: normal_rank(new_samples_i, new_conv_samples_i,
                                    new_labels_i, new_n_eq_i, new_probs_i, nums_for_hard))

            new_samples.append(tf.stop_gradient(samples_rank_i))
            new_conv_samples.append(tf.stop_gradient(conv_samples_rank_i))
            new_labels.append(tf.stop_gradient(labels_rank_i))
            new_n_eqs.append(n_eq_rank_i)

        new_samples = tf.concat(new_samples, axis=0)
        new_conv_samples = tf.concat(new_conv_samples, axis=0)
        new_labels = tf.concat(new_labels, axis=0)
        new_n_eqs = tf.concat(new_n_eqs, axis=0)

        return new_samples, new_conv_samples, new_labels, new_n_eqs


    def sample_selection_hard_analysis(self, inputs_proto, conv_inputs_proto, labels_proto, inputs, conv_inputs, labels_g, nums_for_hard,
                                       weights, d):
        '''
        Selecting unlabeled samples with high prediction scores.
        '''

        #pseudo labeling for unlabeled samples
        logits = self.forward_fc(inputs, weights)
        probs = tf.nn.softmax(logits)
        labels = tf.argmax(probs, axis=-1)

        if d:
            f_i = -1 * tf.ones((FLAGS.nb_ul_samples * FLAGS.num_dis,), dtype=tf.int64)
            g_i = tf.argmax(labels_g, axis=-1)
            l_f = tf.concat([g_i, f_i], axis=0)
            n_eq = tf.cast(tf.not_equal(l_f, labels), dtype=tf.int64)
        else:
            g_i = tf.argmax(labels_g, axis=-1)
            n_eq = tf.cast(tf.not_equal(g_i, labels), dtype=tf.int64)


        labels_num_proto = tf.argmax(labels_proto, axis=-1)
        new_samples = []
        new_conv_samples = []
        new_labels = []
        new_n_eqs = []

        for i in range(self.dim_output):
            # Sometimes, none unlabeled samples are predicted for the class
            # and labeled samples from support set are used instead.
            index_sample_i = tf.where(condition=tf.equal(labels, i))
            index_sample_proto_i = tf.where(condition=tf.equal(labels_num_proto, i))
            num_i = tf.reduce_sum(tf.cast(tf.equal(labels, i), dtype=tf.int16))
            num_total = num_i + tf.constant(FLAGS.shot_num, dtype=tf.int16)
            #selecting n_eq
            n_eq_i = tf.gather_nd(n_eq, index_sample_i)
            n_eq_proto_i = tf.zeros((FLAGS.shot_num,), dtype=tf.int64)
            new_n_eq_i = tf.concat([n_eq_i, n_eq_proto_i], axis=0)
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

            def repeat_and_rank(samples, conv_samples, labels, n_eq, probs, k):
                list_samples = [samples for i in range(k)]
                list_conv_samples = [conv_samples for i in range(k)]
                list_labels = [labels for i in range(k)]
                list_n_eq = [n_eq for i in range(k)]
                list_probs = [probs]
                list_probs.extend([tf.zeros_like(probs, dtype=probs.dtype) for i in range(k - 1)])

                samples_pad = tf.concat(list_samples, axis=0)
                conv_samples_pad = tf.concat(list_conv_samples, axis=0)
                labels_pad = tf.concat(list_labels, axis=0)
                probs_pad = tf.concat(list_probs, axis=0)
                n_eq_pad = tf.concat(list_n_eq, axis=0)

                index_rank = tf.reshape(tf.nn.top_k(probs_pad, k=k).indices, [k, 1])
                samples_rank = tf.gather_nd(samples_pad, index_rank)
                conv_samples_rank = tf.gather_nd(conv_samples_pad, index_rank)
                labels_rank = tf.gather_nd(labels_pad, index_rank)
                n_eq_rank = tf.gather_nd(n_eq_pad, index_rank)

                return samples_rank, conv_samples_rank, labels_rank, n_eq_rank

            def normal_rank(samples, conv_samples, labels, n_eq, probs, k):
                index_rank = tf.reshape(tf.nn.top_k(probs, k=k).indices, [k, 1])
                samples_rank = tf.gather_nd(samples, index_rank)
                conv_samples_rank = tf.gather_nd(conv_samples, index_rank)
                labels_rank = tf.gather_nd(labels, index_rank)
                n_eq_rank = tf.gather_nd(n_eq, index_rank)

                return samples_rank, conv_samples_rank, labels_rank, n_eq_rank

            samples_rank_i, conv_samples_rank_i, labels_rank_i, n_eq_rank_i = tf.cond(
                num_total < tf.constant(nums_for_hard, dtype=tf.int16),
                lambda: repeat_and_rank(new_samples_i,
                                        new_conv_samples_i,
                                        new_labels_i, new_n_eq_i, new_probs_i, nums_for_hard),
                lambda: normal_rank(new_samples_i, new_conv_samples_i,
                                    new_labels_i, new_n_eq_i, new_probs_i, nums_for_hard))

            new_samples.append(tf.stop_gradient(samples_rank_i))
            new_conv_samples.append(tf.stop_gradient(conv_samples_rank_i))
            new_labels.append(tf.stop_gradient(labels_rank_i))
            new_n_eqs.append(n_eq_rank_i)

        new_samples = tf.concat(new_samples, axis=0)
        new_conv_samples = tf.concat(new_conv_samples, axis=0)
        new_labels = tf.concat(new_labels, axis=0)
        new_n_eqs = tf.concat(new_n_eqs, axis=0)

        return new_samples, new_conv_samples, new_labels, new_n_eqs


    def my_coteaching_analysis(self, logits1, logits2, soft_weights, labels, n_eq, forget_rate, num):

        raw_loss1 = self.loss_func(logits1, labels)
        raw_loss2 = self.loss_func(logits2, labels)
        # sort and get low loss indices
        ind1_sorted = tf.argsort(raw_loss1, axis=-1, direction="ASCENDING", stable=True)
        ind2_sorted = tf.argsort(raw_loss2, axis=-1, direction="ASCENDING", stable=True)
        num_remember = tf.cast((1.0 - forget_rate) * num, dtype=tf.int32)
        ind1_update = ind1_sorted[:num_remember]
        ind2_update = ind2_sorted[:num_remember]
        # update logits and compute loss again
        logits1_update = tf.gather(logits1, ind2_update, axis=0)
        labels1_update = tf.gather(labels, ind2_update, axis=0)
        sw1_update = tf.gather(soft_weights, ind2_update, axis=0)
        n_eq_1 = tf.gather(n_eq, ind2_update, axis=0)
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1 = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
        n_eq_2 = tf.gather(n_eq, ind1_update, axis=0)
        outputs2_update = tf.nn.softmax(sw2_update) * logits2_update
        loss2 = self.loss_func(outputs2_update, labels2_update)

        n_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)
        n_sum_1 = tf.cast(tf.reduce_sum(n_eq_1), dtype=tf.float32)
        n_sum_2 = tf.cast(tf.reduce_sum(n_eq_2), dtype=tf.float32)
        ratio1 = 1 - n_sum_1/n_sum
        ratio2 = 1 - n_sum_2/n_sum
        return loss1, loss2, ratio1, ratio2


    def my_coteaching_analysis2(self, logits1, logits2, soft_weights, labels, feat_s, label_s,
                                fc_weights1, fc_weights2, n_eq, forget_rate, num, hard=False):

        raw_loss1 = self.loss_func(logits1, labels)
        raw_loss2 = self.loss_func(logits2, labels)
        # sort and get low loss indices
        ind1_sorted = tf.argsort(raw_loss1, axis=-1, direction="ASCENDING", stable=True)
        ind2_sorted = tf.argsort(raw_loss2, axis=-1, direction="ASCENDING", stable=True)
        num_remember = tf.cast((1.0 - forget_rate) * num, dtype=tf.int32)

        ind1_update = ind1_sorted[:num_remember]
        ind2_update = ind2_sorted[:num_remember]
        # update logits and compute loss again
        logits1_update = tf.gather(logits1, ind2_update, axis=0)
        labels1_update = tf.gather(labels, ind2_update, axis=0)
        sw1_update = tf.gather(soft_weights, ind2_update, axis=0)
        n_eq_1 = tf.gather(n_eq, ind2_update, axis=0)
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1_un = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
        n_eq_2 = tf.gather(n_eq, ind1_update, axis=0)
        outputs2_update = tf.nn.softmax(sw2_update) * logits2_update
        loss2_un = self.loss_func(outputs2_update, labels2_update)

        if forget_rate * num != 0:
            if hard:
                num_forget = 2 * int(forget_rate * num)
            else:
                num_forget = 2 * int(forget_rate * num)
            num_s = FLAGS.shot_num * FLAGS.way_num
            feat_s1, label_s1 = self.MixUp_Operation_2(feat_s, label_s, num_s, num_forget, seed=1)
            outputa1 = self.forward_fc(feat_s1, fc_weights1)
            loss_s1 = self.loss_func(outputa1, label_s1)
            loss1 = tf.concat([loss_s1, loss1_un], axis=0)
            feat_s2, label_s2 = self.MixUp_Operation_2(feat_s, label_s, num_s, num_forget, seed=2)
            outputa2 = self.forward_fc(feat_s2, fc_weights2)
            loss_s2 = self.loss_func(outputa2, label_s2)
            loss2 = tf.concat([loss_s2, loss2_un], axis=0)
        else:
            loss1 = loss1_un
            loss2 = loss2_un

        n_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)
        n_sum_1 = tf.cast(tf.reduce_sum(n_eq_1), dtype=tf.float32)
        n_sum_2 = tf.cast(tf.reduce_sum(n_eq_2), dtype=tf.float32)
        ratio1 = 1 - n_sum_1/n_sum
        ratio2 = 1 - n_sum_2/n_sum

        return loss1, loss2, ratio1, ratio2


    def my_coteaching_analysis3(self, logits1, logits2, soft_weights, labels, feat_s, label_s,
                                fc_weights1, fc_weights2, n_eq, forget_rate, num, hard=False):

        raw_loss1 = self.loss_func(logits1, labels)
        raw_loss2 = self.loss_func(logits2, labels)
        # sort and get low loss indices
        ind1_sorted = tf.argsort(raw_loss1, axis=-1, direction="ASCENDING", stable=True)
        ind2_sorted = tf.argsort(raw_loss2, axis=-1, direction="ASCENDING", stable=True)
        num_remember = tf.cast((1.0 - forget_rate) * num, dtype=tf.int32)

        ind1_update = ind1_sorted[:num_remember]
        ind2_update = ind2_sorted[:num_remember]
        # update logits and compute loss again
        logits1_update = tf.gather(logits1, ind2_update, axis=0)
        labels1_update = tf.gather(labels, ind2_update, axis=0)
        sw1_update = tf.gather(soft_weights, ind2_update, axis=0)
        n_eq_1 = tf.gather(n_eq, ind2_update, axis=0)
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1_un = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
        n_eq_2 = tf.gather(n_eq, ind1_update, axis=0)
        outputs2_update = tf.nn.softmax(sw2_update) * logits2_update
        loss2_un = self.loss_func(outputs2_update, labels2_update)

        if forget_rate * num != 0:
            if hard:
                num_forget = 2 * int(forget_rate * num)
            else:
                num_forget = int(forget_rate * num)
            num_s = FLAGS.shot_num * FLAGS.way_num
            feat_s1, label_s1 = self.only_repeat(feat_s, label_s, num_s, num_forget, seed=1)
            outputa1 = self.forward_fc(feat_s1, fc_weights1)
            loss_s1 = self.loss_func(outputa1, label_s1)
            loss1 = tf.concat([loss_s1, loss1_un], axis=0)
            feat_s2, label_s2 = self.only_repeat(feat_s, label_s, num_s, num_forget, seed=2)
            outputa2 = self.forward_fc(feat_s2, fc_weights2)
            loss_s2 = self.loss_func(outputa2, label_s2)
            loss2 = tf.concat([loss_s2, loss2_un], axis=0)
        else:
            loss1 = loss1_un
            loss2 = loss2_un

        n_sum = tf.cast(tf.reduce_sum(n_eq), dtype=tf.float32)
        n_sum_1 = tf.cast(tf.reduce_sum(n_eq_1), dtype=tf.float32)
        n_sum_2 = tf.cast(tf.reduce_sum(n_eq_2), dtype=tf.float32)
        ratio1 = 1 - n_sum_1/n_sum
        ratio2 = 1 - n_sum_2/n_sum

        return loss1, loss2, ratio1, ratio2
