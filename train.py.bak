# -- encoding:utf-8 --

import os
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib import learn

from nets.text_rnn import TextRNN
from nets.text_cnn import TextCNN
from utils import data_helpers

# Parameters
# ===================================================================
# 给定多少数据作为验证集，默认10%的训练数据
tf.flags.DEFINE_float("dev_sample_percentage", .1,
                      "Percentage of the training data to use for validation")
# 数据的路径
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polarity.neg",
                       "Data source for the negative data.")

# Model Hyperparameters
# 每个单词转化为向量的维度是128  word2vec
tf.flags.DEFINE_integer("embedding_dim", 128,
                        "Dimensionality of character embedding (default: 128)")
# RNN中神经元数目(num_units)
tf.flags.DEFINE_integer("num_units", 128,
                        "Number of units per rnn cell(default: 128)")
# CNN中卷积核的高度(filter_heights)
tf.flags.DEFINE_string("filter_heights", "2,3,4",
                       "Comma-separated filter heights (default: '2,3,4')")
tf.flags.DEFINE_integer("num_filter", 128,
                        "Number of filter per filter heights (default: 128)")
tf.flags.DEFINE_integer("num_filters", None,
                        "Comma-separated filter number per filter heights (default: None)")

# dropout参数
tf.flags.DEFINE_float("dropout_keep_prob", 0.9,
                      "Dropout keep probability (default: 0.5)")
# L2的惩罚项
tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularization lambda (default: 0.0)")

# Training parameters  训练参数
tf.flags.DEFINE_bool('text_cnn', True,
                     'Model base on TextCNN.')
tf.flags.DEFINE_integer("batch_size", 8,
                        "Batch Size (default: 8)")
tf.flags.DEFINE_integer("num_epochs", 200,
                        "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_classes", 2,
                        "Number of sample classes(default: 2)")
tf.flags.DEFINE_float('learning_rate', 1e-5,
                      "Train learning rate(default: 1e-3)")
# 评估
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_string("summary_dir", './graph',
                       "Summary output dir path(default: ./graph)")
# 保存
tf.flags.DEFINE_string("checkpoint_dir", './model',
                       "Checkpoint output dir path(default: ./model)")
tf.flags.DEFINE_integer("checkpoint_every", 10,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5,
                        "Number of checkpoints to store (default: 5)")
# 自动分配设备  cpufpu类的分配
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
# 打印分配日志
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# 解析一下
FLAGS._parse_flags()
print("Parameters:")
# 打印一下刚刚设置的参数
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def fetch_variable_initializer():
    """
    获取变量默认的初始化器
    :return:
    """
    return tf.random_normal_initializer(0.0, 0.1)


def fetch_variable_regularizer():
    return None


def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def train():
    """
    模型训练
    :return:
    """
    with tf.Graph().as_default():
        # 0 输入数据构建
        # a. 获取数据（加载所有数据）
        X, Y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        total_samples = len(X)
        # a2. 对数据做处理(将数据转换为相同长度格式的数据)
        # 保证数据矩阵的大小是一样的，但是邮件有大有小，要经过处理
        # 找出邮件长度最大的那个
        max_document_length = max([len(x.split(" ")) for x in X])
        # 传入最大的长度，默认为我们填充0
        # VocabularyProcessor：做一个词表转换(将文字转换为id，并且让文本长度一致，如果输入文本过短，直接填充0)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        # 训练(也就是构建词汇表映射关系)
        vocab_processor.fit(X)
        # 数据转换
        X = np.array(list(vocab_processor.transform(X)))
        # 记录词汇
        vocab_processor.save(os.path.join(FLAGS.checkpoint_dir, "vocab"))
        # a3. 数据打乱顺序
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(total_samples))
        x_shuffled = X[shuffle_indices]
        y_shuffled = Y[shuffle_indices]
        # b. 划分训练集和验证集
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * total_samples)
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        # c. 迭代器生成
        batches = data_helpers.batch_iter(
            data=list(zip(x_train, y_train)),
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs
        )

        # 1. 构建Session的初始化参数
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )

        # 2. 回话构建，并运行
        with tf.Session(config=session_conf) as sess:
            # 一、构建网络结构
            if FLAGS.text_cnn:
                prefix = 'text_cnn'
                net = TextCNN(
                    network_name="TextCNN",
                    initializer=fetch_variable_initializer(),
                    regularizer=fetch_variable_regularizer(),
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    n_class=FLAGS.num_classes,
                    batch_size=FLAGS.batch_size,
                    filter_heights=list(map(int, FLAGS.filter_heights.split(","))),
                    num_filters=list(map(int, FLAGS.num_filters.split(","))) if FLAGS.num_filters else FLAGS.num_filter
                )
            else:
                prefix = 'text_rnn'
                net = TextRNN(
                    network_name="TextRNN",
                    initializer=fetch_variable_initializer(),
                    regularizer=fetch_variable_regularizer(),
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    n_class=FLAGS.num_classes,
                    batch_size=FLAGS.batch_size,
                    num_units=FLAGS.num_units
                )

            # 二、构建优化器和训练对象
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = optimizer.minimize(loss=net.total_loss, global_step=global_step)

            # 三、模型持久化相关内容构建
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            # 检查目录。tensorflow假设这个目录已经存在，我们需要去创造它
            checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, prefix)
            checkpoint_prefix = os.path.join(checkpoint_dir, "email.model.ckpt")
            check_directory(checkpoint_dir)
            # 记录全局参数
            saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

            # 四、可视化对象构建
            summary_dir = os.path.join(FLAGS.summary_dir, prefix)
            train_summary_dir = os.path.join(summary_dir, "train")
            dev_summary_dir = os.path.join(summary_dir, "dev")
            check_directory(train_summary_dir)
            check_directory(dev_summary_dir)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            summary_op = tf.summary.merge_all()

            # 五、模型的训练运行
            # 模型初始化恢复
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore model weight from '{}'".format(FLAGS.checkpoint_dir))
                # restore：进行模型恢复操作
                saver.restore(sess, ckpt.model_checkpoint_path)
                # recover_last_checkpoints：模型保存的时候，我们会保存多个模型文件，默认情况下，模型恢复的时候，磁盘文件不会进行任何操作，为了保证磁盘中最多只有max_to_keep个模型文件，故需要使用下列API
                saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # 处理一下当前批次的数据，不要让当前批次的数据长度是最长序列长度(全局)，使用当前批次中的最长长度
                # 1. 将当前批次中的所有值按对应位置累加， 结果是有值的地方非0，这样只要找到第一个0的位置，就可以做截断操作
                tmp_sum = np.sum(x_batch, 0)
                tmp_index = int(max(10, np.argmin(tmp_sum)))
                x_batch = np.asarray(x_batch)[:, :tmp_index]

                feed_dict = {
                    net.input: x_batch,
                    net.output: y_batch,
                    net.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, summary_op, net.total_loss, net.accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    net.input: x_batch,
                    net.output: y_batch
                }

                step, summaries, loss, accuracy = sess.run(
                    [global_step, summary_op, net.total_loss, net.accuracy],
                    feed_dict)
                time_str = datetime.now().isoformat()
                # step  步进  loss
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # d. 迭代所有批次
            for batch in batches:
                # 1. 将x和y分割开
                x_batch, y_batch = zip(*batch)
                # 2. 训练操作
                train_step(x_batch, y_batch)
                # 3. 获取当前的更新的次数
                current_step = tf.train.global_step(sess, global_step)
                # 4. 进行验证数据可视化输出
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                # 5. 进行模型持久化输出
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def check_parameters():
    msg = ''
    if (not FLAGS.positive_data_file) or (not check_directory(FLAGS.positive_data_file, False)):
        msg += '参数positive_data_file必须给定!!!\n'
    if (not FLAGS.negative_data_file) or (not check_directory(FLAGS.negative_data_file, False)):
        msg += '参数negative_data_file必须给定!!!\n'
    if msg == '':
        return True, None
    else:
        return False, msg


if __name__ == '__main__':
    # 0. 设置一下日志
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # 1. 参数验证
    flag, msg = check_parameters()
    print(msg)
    print(flag)
    if not flag:
        raise Exception(msg)

    # 2. 开始训练
    train()
