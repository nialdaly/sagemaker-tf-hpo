from __future__ import print_function

import argparse
import logging
import os
import json
import gzip
import numpy as np
import traceback
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def conv_model():
    model = models.Sequential()
    # 2D convolution layer
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    # Densely-connected NN layer
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=10))
    return model

# Decodes and preprocesses data
def convert_to_numpy(data_dir, images_file, labels_file):
    """ Converts the byte string to numpy arrays """
    with gzip.open(os.path.join(data_dir, images_file), 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    
    with gzip.open(os.path.join(data_dir, labels_file), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (images, labels)

def mnist_to_numpy(data_dir, train):
    """Loads the raw MNIST data into Numpy array
    
    Args:
        data_dir (str): directory of MNIST raw data. 
            This argument can be accessed via SM_CHANNEL_TRAINING
        
        train (bool): use training data
    Returns:
        tuple of images and labels as numpy array
    """
    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    return convert_to_numpy(data_dir, images_file, labels_file)

def normalize(x, axis):
    eps = np.finfo(float).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std

def train(args):
    # Creates the data loader from the train / test channels
    x_train, y_train = mnist_to_numpy(data_dir=args.train, train=True)
    x_test, y_test = mnist_to_numpy(data_dir=args.test, train=False)
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    # Normalizes the inputs to mean of 0 and std of 1
    x_train, x_test = normalize(x_train, (1, 2)), normalize(x_test, (1, 2))

    # Expand channel axis (TensorFlow uses depth minor convention)
    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
    
    # Normalizea the data to a mean of 0 and std of 1
    train_loader = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(len(x_train)).batch(args.batch_size)

    test_loader = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(args.batch_size)

    model = conv_model()
    model.compile()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, 
        beta_1=args.beta_1,
        beta_2=args.beta_2
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        
        train_loss(loss)
        train_accuracy(labels, predictions)
        return 
        
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return
    
    logger.info("Training starts ...")
    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        for batch, (images, labels) in enumerate(train_loader):
            train_step(images, labels)

        logger.info(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result()}, '
        )
        
        for images, labels in test_loader:
            test_step(images, labels)

        # Hyperparameter tuning metric
        logger.info(f'Test Loss: {test_loss.result()}')
        logger.info(f'Test Accuracy: {test_accuracy.result()}')
        
    # Saves the model (version number required for the serving container to load model)
    version_num = '00000000'
    ckpt_dir = os.path.join(args.model_dir, version_num)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save(ckpt_dir)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    
    # Environment variables provided by the training image
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)