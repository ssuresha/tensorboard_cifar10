import numpy as np 
import os
import tensorflow as tf 
from cifar_model import cifar_model
from keras.utils import np_utils
from keras.datasets import cifar10
from tensorflow.contrib.tensorboard.plugins import projector

# Batch Generator for a dataset
# ================================================================
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size 
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index], epoch, batch_num 


# Data Preparation
# =================================================================
print ("Loading data...")
(train_features, train_labels), (dev_features, dev_labels) = cifar10.load_data()

num_classes = len(np.unique(train_labels))
x_train = train_features.astype('float32')/255
x_dev = dev_features.astype('float32')/255
# convert class labels to binary class labels
y_train = np_utils.to_categorical(train_labels, num_classes)
y_dev = np_utils.to_categorical(dev_labels, num_classes)

print ("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training 
# =================================================================
learn_rts = [1e-2, 1e-3, 1e-4] 

for learn_rt in learn_rts:

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement= True,
                                      log_device_placement= True)

        sess = tf.Session()
        #sess = tf.Session(config = session_conf) 

        with sess.as_default():
            # Build Graph
            cnn = []
            cnn = cifar_model(num_classes)
            cnn.build_graph()

            # Define Optimization procedure
            global_step = tf.Variable(0, name = "global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(learn_rt)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            step_update = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

            # Keep track of summaries (for Tensorboard)
            loss_summary = tf.summary.scalar("loss",cnn.loss)
            accuracy_summary = tf.summary.scalar("accuracy",cnn.accuracy)
            # image_summary = tf.summary.image("input_images",cnn.input_x)

            # Keep track of weights and their gradient values  
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    var_summary = tf.summary.histogram("{}/hist".format(v.name),v)
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)

            # Summarize 
            summary_op = tf.summary.merge_all()

            # Checkpoint Directory 
            out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs"))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model_lr_{}".format(learn_rt))

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(),max_to_keep = 5)

            # Train Summary
            train_summary_dir = os.path.join(out_dir,"summaries","lr_{}".format(learn_rt))
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Initialize all variables 
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, batch_idx, epoch_idx):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 0.75
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [step_update, global_step, summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                print("Epoch {}, Batch_no {} : loss {:g}, acc {:g}".format(epoch_idx, batch_idx, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)

                print("Dev loss {:g}, Dev acc {:g}".format(loss, accuracy))
                
            # Generate Batches
            num_epochs = 5
            batch_size = 128
            batches = batch_iter(list(zip(x_train,y_train)),batch_size,num_epochs)

            # Training loop
            print ("Training Started with learning rate : {}".format(learn_rt))
            for batch_tup in batches:

                batch, epoch_idx, batch_idx = batch_tup
                x_batch, y_batch = zip(*batch)
                train_step(x_batch,y_batch, batch_idx+1, epoch_idx+1)
                current_step = tf.train.global_step(sess,global_step)

                if current_step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            print ("\n Model Evaluation (with learning rate : {}) on Validation Set:".format(learn_rt))
            dev_step(x_dev,y_dev)
            print ("")












