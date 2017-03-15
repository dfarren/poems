from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cPickle
import time
import os
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import pdb

# To run tensorboard:
# tensorboard --logdir=. --port 5000
# Go to Chrome and open: localhost:[port]
# (For example at Amazon: http://u9c5c8e4f9752570c1885.ant.amazon.com:[port])

# Explicitly set GPUs for use
# def set_gpus(gpus):
#   arg = ",".join(map(str, gpus))
#   os.environ["CUDA_VISIBLE_DEVICES"] = arg

# set_gpus([0])

# General parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate')
flags.DEFINE_integer('num_epochs', 1, 'number of epochs to run trainer')
flags.DEFINE_integer('batch_size', 500, 'size of the minibatch')
flags.DEFINE_integer('display_step', 10, 'number of batch-steps after which to display results')
flags.DEFINE_float('keep_prob', 0.85, 'probability of keeping data after dropout')

flags.DEFINE_string('data_dir', '../', 'directory with all the data')
flags.DEFINE_string('summaries_dir', 'summary/', 'directory with summary')
flags.DEFINE_string('save_dir', 'savedmodel/', 'directory with saved model')

# Neural network parameters
flags.DEFINE_integer('n_input', 50, 'dimension of the input')
flags.DEFINE_integer('n_hidden_1', 64, 'number of neurons in the first hidden layer')
flags.DEFINE_integer('n_hidden_2', 32, 'number of neurons in the first hidden layer')
flags.DEFINE_integer('n_classes', 2, 'number of categories')

# Load data
def load_data():
    with open(FLAGS.data_dir + 'poem_vectors.pkl', 'rb') as f:
        poem_vectors = cPickle.load(f)

    with open(FLAGS.data_dir + 'labels.pkl', 'rb') as f:
        labels = cPickle.load(f)

    X = []
    y = []

    for title, vector in poem_vectors.iteritems():
        X.append(vector)
        if title in labels:
            y.append(1)
        else:
            y.append(0)

    # Data is heavily skewed, so we balance it. 
    sme = SMOTEENN(random_state=42)
    X, y = sme.fit_sample(X, y)

    y_neg = []
    for e in y:
        if e == 1:
            y_neg.append(0)
        else:
            y_neg.append(1)
    
    y = np.hstack((np.reshape(y_neg, (len(y_neg), 1)), np.reshape(y, (len(y), 1))))

    permutation = np.random.permutation(y.shape[0])
    X = np.array(X)[permutation]
    y = np.array(y)[permutation]

    trainVal_X, test_X, trainVal_y, test_y = train_test_split(X, y, test_size=0.2, random_state=36)
    train_X, val_X, train_y, val_y = train_test_split(trainVal_X, trainVal_y, test_size=0.25, random_state=96)

    return train_X, val_X, test_X, train_y, val_y, test_y

# loading ..
train_X, val_X, test_X, train_y, val_y, test_y = load_data()

print("Load in data successfully !!")

# tf graph input
X = tf.placeholder(tf.float32, shape=[None, FLAGS.n_input], name = "X")
y = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes], name = "y")
keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

# Weights and biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Layer
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(input_tensor, weights), biases)

        print(preactivate)
        if act is not None:
            activations = act(preactivate, name='activation')
        else:
            activations = preactivate

    return activations

# Cross-entropy loss
def cal_cross_entropy(pred, labels, name):
    with tf.name_scope(name):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))

        _ = tf.summary.scalar(name + ' ' + 'cross entropy', loss)

    return loss


# Model
def multilayer_perceptron(X, y, keep_prob, name):
    with tf.variable_scope('train'):
        layer_1 = nn_layer(X, FLAGS.n_input, FLAGS.n_hidden_1, 'hidden_1', act=tf.nn.relu)
  
        with tf.name_scope('dropout_layer_1'):
            layer_1 = tf.nn.dropout(layer_1, keep_prob)

        layer_2 = nn_layer(layer_1, FLAGS.n_hidden_1, FLAGS.n_hidden_2, 'hidden_2', act=tf.nn.relu)

        with tf.name_scope('dropout_layer_2'):
            layer_2 = tf.nn.dropout(layer_2, keep_prob)

        pred = nn_layer(layer_2, FLAGS.n_hidden_2, FLAGS.n_classes, 'output_loss', act=tf.nn.relu)
        loss = cal_cross_entropy(pred, y, name)

        labels_pred = nn_layer(layer_2, FLAGS.n_hidden_2, FLAGS.n_classes, 'output_pred', act=tf.nn.softmax)
        
    return loss, labels_pred 
    
def evaluate (preds, labels, name):
    with tf.name_scope(name):
        pos_predictions = tf.argmax(preds, 1) > 0
        neg_predictions = tf.argmax(preds, 1) < 1
        pos_labels = tf.argmax(labels, 1) > 0
        neg_labels = tf.argmax(labels, 1) < 1

        TP = tf.reduce_sum(tf.to_float(tf.logical_and(pos_predictions, pos_labels)))
        FP = tf.reduce_sum(tf.to_float(tf.logical_and(pos_predictions, neg_labels)))
        FN = tf.reduce_sum(tf.to_float(tf.logical_and(neg_predictions, pos_labels)))
        TN = tf.reduce_sum(tf.to_float(tf.logical_and(neg_predictions, neg_labels)))

        precision = tf.cond(TP + FP > 0, 
                        lambda: TP / (TP + FP), 
                        lambda: tf.constant(0.0))

        recall = tf.cond(TP + FN > 0, 
                        lambda: TP / (TP + FN), 
                        lambda: tf.constant(0.0))

        f1 = tf.cond(precision + recall > 0,
                        lambda: 2 * precision * recall / (precision + recall),
                        lambda: tf.constant(0.0))

        acc = tf.cond(TP + TN + FP + FN > 0,
                        lambda: (TP + TN) / (TP + TN + FP + FN),
                        lambda: tf.constant(0.0))

        _ = tf.summary.scalar(name + ' ' + 'precision', precision)
        _ = tf.summary.scalar(name + ' ' + 'recall', recall)
        _ = tf.summary.scalar(name + ' ' + 'F1', f1)
        _ = tf.summary.scalar(name + ' ' + 'accuracy', acc)

    return precision, recall, f1, acc 

# Loss & optimizer
xentropy_loss, labels_pred = multilayer_perceptron(X, y, keep_prob, "train")
opt_operation = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(xentropy_loss)
precision, recall, f1, acc = evaluate(labels_pred, y, "train")

xentropy_loss_val, labels_pred_val = multilayer_perceptron(X, y, keep_prob, "val")
precision_val, recall_val, f1_val, acc_val = evaluate(labels_pred_val, y, "val")

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Save best model so far
saver = tf.train.Saver()
best_val_xentropy = float('inf')

# Launch the graph & training
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Merge all summaries together and write them to summaries_dir
    merged = tf.summary.merge()
    writer = tf.summary.FileWriter('%s' % (FLAGS.summaries_dir), sess.graph)
                
    # Initialize the variables (the trained variables and the epoch counter).
    sess.run(init_op)    

    # Training
    for epoch in xrange(FLAGS.num_epochs):
        number_batch = int((train_y.shape[0] - 1) / FLAGS.batch_size)
    
        for step in xrange((number_batch + 1) * epoch, (number_batch + 1)*(epoch + 1)):
            new_step = step % (number_batch + 1)
            start = FLAGS.display * new_step
            end = min(FLAGS.batch_size * (new_step + 1), train_y.shape[0]) - 1

            #batch_x = train_X[(FLAGS.batch_size * step):min(FLAGS.batch_size * (step + 1), train_y.shape[0]) - 1,]
            #batch_y = train_y[(FLAGS.batch_size * step):min(FLAGS.batch_size * (step + 1), train_y.shape[0]) - 1,]

            batch_x = train_X[start:end, ]
            batch_y = train_y[start:end, ]

            _ = sess.run(opt_operation, feed_dict = {X: batch_x,
                                                     y: batch_y,
                                                     keep_prob: FLAGS.keep_prob})
            if step % FLAGS.display_step == 0:
                start_time = time.time()

                ops_train = [merged, xentropy_loss, precision, recall, f1, acc] 
                train_summary, x_loss, pre, rec, f1_score, accuracy = \
                                  sess.run(ops_train, feed_dict = {X: batch_x, 
                                                                   y: batch_y,
                                                                   keep_prob: FLAGS.keep_prob})
                duration = time.time() - start_time
                
                train_format_string = 'Training: Step = %d, XEntropy = %.6f, Pre = %.6f, ' \
                                                        'Rec = %.6f, F1 = %.6f, Acc = %.6f (%.3f sec)'
                print(train_format_string % (step, x_loss, pre, rec, f1_score, accuracy, duration))

                writer.add_summary(train_summary, step)   
            
                # Metrics on validation set
                ops_val = [merged, xentropy_loss_val, f1_val, acc_val]
                val_summary, val_xentropy, val_f1, val_acc = sess.run(ops_val, feed_dict = {X: val_X,
                                                                           y: val_y,
                                                                           keep_prob: 1})

                val_format_string = 'Validation: Step = %d, XEntropy = %.6f, F1 = %.6f, Acc = %.6f'

                print(val_format_string % (step, val_xentropy, val_f1, val_acc))
                
                writer.add_summary(val_summary, step)
               
                # Save the best model
                if val_xentropy < best_val_xentropy:
                    best_val_xentropy = val_xentropy
                    saver.save(sess, FLAGS.save_dir + 'model.ckpt', global_step=step)
        

    # Testing
    ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restored!')
    else:
        print("No Checkpoint Found")
    ops_test = [merged, xentropy_loss, precision, recall, f1, acc]
    # test_summary, test_loss, test_pre, test_rec, test_f1, test_acc = sess.run(ops_test, feed_dict = {X: test_X,
    #                                                                                                 y: test_y,
    #                                                                                                 keep_prob: 1})
    # print('Test: F1 score = %.6f, Accuracy = %.6f' % (test_f1, test_acc))
    
