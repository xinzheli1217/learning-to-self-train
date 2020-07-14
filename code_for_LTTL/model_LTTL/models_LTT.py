import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
from models import Models

from tensorflow.python.platform import flags
from misc import resnet_conv_block, leaky_relu, mse

FLAGS = flags.FLAGS

class Model_LTT(Models):
    def __init__(self):
        Models.__init__(self)
        self.loss_func_rec = mse
        self.swn_lr = tf.placeholder_with_default(FLAGS.swn_lr, ())
        self.meta_kld_lr = tf.placeholder_with_default(FLAGS.kld_lr, ())


    def pre_training(self, feat_s, conv_feat_s, label_s, feat_un, conv_feat_un, fc_weights, swn_weights, lr_weights, seed, if_dis=True):

        if if_dis:
            feat_un = tf.random_shuffle(feat_un, seed=seed)[:(FLAGS.nb_ul_samples * FLAGS.way_num),:]
            conv_feat_un = tf.random_shuffle(conv_feat_un, seed=seed)[:(FLAGS.nb_ul_samples * FLAGS.way_num),:]

        pseudo_labels = self.computing_pseudo_labels(feat_s, label_s, feat_un, 10, fc_weights,
                                                     one_hot=True)

        soft_weights = self.computing_soft_weights(conv_feat_s, label_s, conv_feat_un, swn_weights, class_num=5,
                                                   samples_num=FLAGS.nb_ul_samples * FLAGS.way_num,
                                                   reuse=tf.AUTO_REUSE)
        soft_weights = tf.reshape(soft_weights, [-1, 5])

        output_s = self.forward_fc(feat_s, fc_weights)
        maml_loss_s = self.loss_func(output_s, label_s)

        output_un = self.forward_fc(feat_un, fc_weights)
        new_logits_un = tf.nn.softmax(soft_weights) * output_un
        maml_loss_un = self.loss_func(new_logits_un, pseudo_labels)

        loss = tf.concat([maml_loss_s, maml_loss_un], axis=0)

        grads = tf.gradients(loss, list(fc_weights.values()))
        gradients = dict(zip(fc_weights.keys(), grads))

        if if_dis:
            fast_fc_weights = dict(zip(fc_weights.keys(),
                                    [fc_weights[key] - lr_weights[key]* gradients[key] for key in
                                    fc_weights.keys()]))
        else:
            fast_fc_weights = dict(zip(fc_weights.keys(),
                                    [fc_weights[key] - self.update_lr * gradients[key] for key in
                                    fc_weights.keys()]))

        # pretraining steps
        for j in range(FLAGS.pre_train_epoch_num - 1):
            maml_loss_s = self.loss_func(self.forward_fc(feat_s, fast_fc_weights), label_s)
            logits_un = self.forward_fc(feat_un, fast_fc_weights)
            logits_new_un = tf.nn.softmax(soft_weights) * logits_un
            loss_un = self.loss_func(logits_new_un, pseudo_labels)

            loss = tf.concat([maml_loss_s, loss_un], axis=0)

            grads = tf.gradients(loss, list(fast_fc_weights.values()))
            gradients = dict(zip(fast_fc_weights.keys(), grads))
            if if_dis:
                fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                          [fast_fc_weights[key] - lr_weights[key] * gradients[key] for key in
                                           fast_fc_weights.keys()]))
            else:
                fast_fc_weights = dict(zip(fast_fc_weights.keys(),
                                          [fast_fc_weights[key] - self.update_lr * gradients[key] for key in
                                           fast_fc_weights.keys()]))

        return fast_fc_weights


    def construct_fc_weights_coteaching(self, seed, str):
        dtype = tf.float32
        fc_weights = {}
        fc_initializer = tf.contrib.layers.xavier_initializer(seed=seed, dtype=dtype)
        fc_weights['w5'] = tf.get_variable('fc_w5_'+str, [512, self.dim_output], initializer=fc_initializer)
        fc_weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='fc_b5_'+str)
        return fc_weights


    def forward_dropout_fc(self, inp, fc_weights, dp_rate):
        inp = tf.nn.dropout(inp, rate=dp_rate)
        net = tf.matmul(inp, fc_weights['w5']) + fc_weights['b5']
        return net


    def construct_lr_weights(self):
        dtype = tf.float32
        lr_weights = {}
        lr_initializer =  tf.constant_initializer(value=2* FLAGS.base_lr, dtype=dtype)

        lr_weights['w5'] = tf.get_variable('lr_w5', [512, self.dim_output], initializer=lr_initializer)
        lr_weights['b5'] = tf.get_variable('lr_b5', [self.dim_output], initializer=lr_initializer)

        return lr_weights


    def computing_soft_weights(self, feat_1, label_1, feat_2, weights, class_num, samples_num, reuse):
        num_label_1 = tf.argmax(label_1, axis=1)
        scores_list = []

        for i in range(class_num):
            index_samples_i = tf.where(condition=tf.equal(num_label_1, i))
            feat_class_i = tf.gather_nd(feat_1, index_samples_i)
            emb_class_i = tf.reduce_mean(feat_class_i, axis=0, keep_dims=True)

            tile_emb_class_i = tf.tile(emb_class_i, [samples_num, 1,1,1])
            concat_emb = tf.concat([feat_2, tile_emb_class_i], axis=3)

            scores_i = tf.reshape(self.forward_swn(concat_emb, weights, reuse=tf.AUTO_REUSE), [-1,1])
            scores_list.append(scores_i)

        soft_weights = tf.concat(scores_list, axis=1)

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


    def computing_pseudo_labels(self, feat_1, label_1, feat_2, num_steps, weights, one_hot):

        output_1 = self.forward_fc(feat_1, weights)
        loss_1 = self.loss_func(output_1, label_1)
        grads = tf.gradients(loss_1, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict(zip(weights.keys(),
                                [weights[key] - self.update_lr * gradients[key] for key in
                                 weights.keys()]))

        for i in range(num_steps - 1):
            loss = self.loss_func(self.forward_fc(feat_1, fast_weights), label_1)
            grads = tf.gradients(loss, list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(),
                                    [fast_weights[key] - self.update_lr * gradients[key] for key in
                                     fast_weights.keys()]))

        output_2 = self.forward_fc(feat_2, fast_weights)
        probc = tf.nn.softmax(output_2)
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


    def MixUp_Operation(self, f_s, c_f_s, l_s, f_un, c_f_un, l_un):
        total_un_nums = FLAGS.nb_ul_samples * (FLAGS.way_num + FLAGS.num_dis)
        tile_nums = total_un_nums // (FLAGS.shot_num * FLAGS.way_num)

        f_s_new = tf.tile(f_s, [tile_nums, 1])
        c_f_s_new = tf.tile(c_f_s, [tile_nums, 1, 1, 1])
        l_s_new = tf.tile(l_s, [tile_nums, 1])

        dist_beta = tf.distributions.Beta(1.0, 1.0)
        lmb = dist_beta.sample(total_un_nums)
        lmb_f = tf.reshape(lmb, [-1, 1])
        lmb_c_f = tf.reshape(lmb, [-1, 1, 1, 1])
        lmb_l = tf.reshape(lmb, [-1, 1])

        f = f_s_new * lmb_f + f_un * (1. - lmb_f)
        c_f = c_f_s_new * lmb_c_f + c_f_un * (1. - lmb_c_f)
        l = l_s_new * lmb_l + l_un * (1. - lmb_l)

        return f, c_f, l


    def MixUp_Operation_2(self, f, l, num_s, num_un, seed=0):
        if FLAGS.shot_num == 1:
            f_re = tf.tile(f, [30, 1])
            l_re = tf.tile(l, [30, 1])
            sample_num = num_s * 30
        else:
            f_re = tf.tile(f, [10, 1])
            l_re = tf.tile(l, [10, 1])
            sample_num = num_s * 10

        dist_beta = tf.distributions.Beta(1.0, 1.0)
        lmb = dist_beta.sample(sample_num)
        lmb = tf.reshape(lmb, [-1, 1])

        f_ra = tf.random_shuffle(f_re, seed=seed)
        l_ra = tf.random_shuffle(l_re, seed=seed)

        f_mixup = f_ra * lmb + f_re * (1. - lmb)
        l_mixup = l_ra * lmb + l_re * (1. - lmb)

        return f_mixup[:num_un,:], l_mixup[:num_un,:]


    def only_repeat(self, f, l, num_s, num_un, seed=0):
        if FLAGS.shot_num == 1:
            f_re = tf.tile(f, [15, 1])
            l_re = tf.tile(l, [15, 1])
        else:
            f_re = tf.tile(f, [5, 1])
            l_re = tf.tile(l, [5, 1])

        f_ra = tf.random_shuffle(f_re, seed=seed)
        l_ra = tf.random_shuffle(l_re, seed=seed)

        return f_ra[:num_un, :], l_ra[:num_un, :]


    def kl_divergence_from_logits(self, logits_a, logits_b):
        """Gets KL divergence from logits parameterizing categorical distributions.

        Args:
            logits_a: A tensor of logits parameterizing the first distribution.
            logits_b: A tensor of logits parameterizing the second distribution.

        Returns:
            The (batch_size,) shaped tensor of KL divergences.
        """
        distribution1 = tf.contrib.distributions.Categorical(logits=logits_a)
        distribution2 = tf.contrib.distributions.Categorical(logits=logits_b)
        return tf.contrib.distributions.kl_divergence(distribution1, distribution2)


    def computing_pl_for_test(self, feat_s, conv_feat_s, label_s, feat_un, conv_feat_un, label_un, num_steps, nums_for_hard,
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

        output_un = self.forward_fc(feat_un, fast_weights)
        prob_un = tf.nn.softmax(output_un)
        pseudo_labels_un = tf.stop_gradient(tf.one_hot(
            tf.argmax(prob_un, axis=-1),
            tf.shape(prob_un)[1],
            dtype=prob_un.dtype,
        ))

        acc_pl = tf.contrib.metrics.accuracy(tf.argmax(pseudo_labels_un, 1), tf.argmax(label_un, 1))

        feats_new, conv_feat_new, pseudo_labels = self.sample_selection_hard(feat_s, conv_feat_s, label_s, feat_un,
                                                                             conv_feat_un, nums_for_hard, fast_weights)

        return feats_new, conv_feat_new, pseudo_labels, acc_pl


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


    def sample_selection_hard_co(self, inputs_proto, conv_inputs_proto, labels_proto, inputs, conv_inputs, nums_for_hard,
                                 weights1, weights2):
        '''
        Selecting unlabeled samples with high prediction scores.
        '''

        #pseudo labeling for unlabeled samples
        logits1 = self.forward_fc(inputs, weights1)
        logits2 = self.forward_fc(inputs, weights2)
        logits = logits1 + logits2
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


    def my_coteaching(self, logits1, logits2, soft_weights, labels, forget_rate, num):

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
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1 = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
        outputs2_update = tf.nn.softmax(sw2_update) * logits2_update
        loss2 = self.loss_func(outputs2_update, labels2_update)
        return loss1, loss2


    def my_coteaching_s(self, logits1, logits2, soft_weights, labels, forget_rate):

        s1 = tf.reduce_max(tf.nn.softmax(logits1), axis=1, keepdims=True)
        s2 = tf.reduce_max(tf.nn.softmax(logits2), axis=1, keepdims=True)
        s1 = tf.reshape(s1, (-1,))
        s2 = tf.reshape(s2, (-1,))
        # sort and get high prediction indices
        ind1_sorted = tf.argsort(s1, axis=-1, direction="DESCENDING", stable=True)
        ind2_sorted = tf.argsort(s2, axis=-1, direction="DESCENDING", stable=True)
        num_remember = tf.cast((1.0 - forget_rate) * (FLAGS.way_num*FLAGS.nb_ul_samples), dtype=tf.int32)
        ind1_update = ind1_sorted[:num_remember]
        ind2_update = ind2_sorted[:num_remember]
        # update logits and compute loss again
        logits1_update = tf.gather(logits1, ind2_update, axis=0)
        labels1_update = tf.gather(labels, ind2_update, axis=0)
        sw1_update = tf.gather(soft_weights, ind2_update, axis=0)
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1 = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
        outputs2_update = tf.nn.softmax(sw2_update) * logits2_update
        loss2 = self.loss_func(outputs2_update, labels2_update)
        return loss1, loss2


    def my_coteaching_2(self, logits1, logits2, soft_weights, labels, feat_s, label_s,
                        fc_weights1, fc_weights2, forget_rate, num, hard=False):

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
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1_un = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
        outputs2_update = tf.nn.softmax(sw2_update) * logits2_update
        loss2_un = self.loss_func(outputs2_update, labels2_update)

        if forget_rate * num != 0:
            if hard:
                num_forget = 3 * int(forget_rate * num)
            else:
                num_forget = int(forget_rate * num)
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

        return loss1, loss2


    def my_coteaching_3(self, logits1, logits2, soft_weights, labels, feat_s, label_s,
                        fc_weights1, fc_weights2, forget_rate, num, hard=False):

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
        outputs1_update = tf.nn.softmax(sw1_update) * logits1_update
        loss1_un = self.loss_func(outputs1_update, labels1_update)

        logits2_update = tf.gather(logits2, ind1_update, axis=0)
        labels2_update = tf.gather(labels, ind1_update, axis=0)
        sw2_update = tf.gather(soft_weights, ind1_update, axis=0)
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

        return loss1, loss2








