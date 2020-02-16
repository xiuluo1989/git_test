# -- encoding:utf-8 --
"""
Create on 19/8/9 20:25
"""

import tensorflow as tf
from tensorflow.contrib import slim


class TextCNN(object):
    """
    A CNN for text classification
    """

    def __init__(self, network_name, initializer, regularizer, vocab_size, embedding_size,
                 n_class, batch_size, filter_heights, num_filters):
        self.network_name = network_name
        self.initializer = initializer
        self.regularizer = regularizer
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.batch_size = batch_size
        self.filter_heights = filter_heights
        if isinstance(num_filters, list):
            # isinstance: 判断num_filters对象是不是list，是返回True，否则返回False
            if len(self.filter_heights) != len(num_filters):
                raise Exception("filter_heights和num_filters必须长度一致")
            else:
                self.num_filters = num_filters
        elif isinstance(num_filters, int):
            self.num_filters = [num_filters for _ in self.filter_heights]
        else:
            raise Exception("参数num_filters只能是list列表或者int类型的数字！！！")
        self.embedding_size = embedding_size

        with tf.variable_scope(self.network_name,
                               initializer=self.initializer,
                               regularizer=self.regularizer):
            # 1. Placeholders for input, output, dropout, batch_size
            with tf.variable_scope("placeholders"):
                self.input = tf.placeholder(tf.int32, [None, None], name='input_x')
                self.output = tf.placeholder(tf.float32, [None, self.n_class], name='input_y')
                self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')
                self.batch_size = tf.placeholder_with_default(self.batch_size, shape=[], name='batch_size')

            # 1.5 Embedding Layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.embedding = tf.Variable(
                    # 指定初始化的范围
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
                # embedded_chars结构为[batch_size, sequence_length, embedding_size]
                self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input)
                # 转化为4维的，原本是三维的，tf处理的是4维的，新维度是-1；
                # [batch_size, sequence_length, embedding_size, channel]
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # 2. Build CNN output
            # Create a convolution + maxpool layer for each filter size
            # 创建每个过滤器的大小卷积+ maxpool层
            pooled_outputs = []
            num_filters_total = 0
            with tf.variable_scope("cnn"):
                for i, filter_height in enumerate(self.filter_heights):
                    with tf.variable_scope("conv-maxpool-%s" % filter_height):
                        # Convolution Layer
                        num_filters_total += self.num_filters[i]
                        # filter_size选几个单词h，embedding_size每个占了多长w   7*5*1  输入1维，输出128维 128个特征图
                        filter_shape = [filter_height, self.embedding_size, 1, self.num_filters[i]]
                        # 高斯初始化
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                        # 初始化为常量0.1
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                        conv = tf.nn.conv2d(
                            self.embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",  # 不做padding
                            name="conv")
                        # Apply nonlinearity: [N, H, W, C]
                        # N: 样本数目(批次大小)
                        # H: 卷积之后的高度
                        # W: 1
                        # C: self.num_filters[i]
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs; [N, C]（两种计算方式）
                        pooled = tf.reduce_max(input_tensor=h, axis=[1, 2], keep_dims=True)
                        # pooled = tf.nn.max_pool(
                        #     h,
                        #     # (len-fiter+padding)/strides =len-filter
                        #     ksize=[1, sequence_length - filter_height + 1, 1, 1],
                        #     strides=[1, 1, 1, 1],
                        #     padding='VALID',
                        #     name="pool")
                        pooled_outputs.append(pooled)

                # 做一个合并
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                # d. 做一个drop out操作
                h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)

            # 3. Build FC output
            with tf.variable_scope("fc"):
                in_units = h_drop.get_shape()[-1]
                w = tf.get_variable(name='w', shape=[in_units, self.n_class])
                b = tf.get_variable(name='b', shape=[self.n_class])
                self.scores = tf.nn.xw_plus_b(h_drop, weights=w, biases=b, name='scores')
                self.predictions = tf.argmax(self.scores, axis=1, name='predictions')

            # 4. Build Loss
            with tf.variable_scope("loss"):
                self.losses = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.scores))
                tf.losses.add_loss(self.losses)
                self.total_loss = tf.losses.get_total_loss(name='total_loss')
                tf.summary.scalar('total_loss', self.total_loss)
                tf.summary.scalar('loss', self.losses)

            # 5. Build Estimate eval
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.output, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
                tf.summary.scalar('accuracy', self.accuracy)
