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
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Embedding
from tensorflow.keras.models import Model

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class ConvModel(Model):
    def __init__(self):
        self.embe = Embedding(8185, output_dim=16)
        self.conv = Conv1D(32, kernel_size=6, activation='elu')
        self.glob = GlobalAveragePooling1D()
        self.dens = Dense(2)
        
    def call(self, x, training=None, mask=None):
        x = self.embe(x)
        x = self.conv(x)
        x = self.glob(x)
        x = self.dens(x)
        return x
                      
                    
def download_imdb_data(local_data_dir):
    """ Downloads the IMDB data from TensorFlow Datsets """
    if not os.path.exists(local_data_dir):
        os.makedirs(local_data_dir)
        
    (train_data, test_data), info = tfds.load(
        name='imdb_reviews/subwords8k',
        data_dir=local_data_dir,
        split=[tfds.Split.TRAIN, tfds,Split.TEST],
        with_info=True,
        as_supervised=True
    )
    return (train_data, test_data), info


def padded_batch_dataset(args, train_data, test_data):
    padded_shapes = ([9000], ())
    
    train_dataset = train_data.shuffle(25000).padded_batch(
        padded_shapes=padded_shapes,
        batch_size=args.batch_size
    )
    
    test_dataset = test_data.shuffle(25000).padded_batch(
        padded_shapes=padded_shapes,
        batch_size=args.batch_size
    )
    
    return train_dataset, test_dataset
    

def train(args):
    # Local data directory path
    local_data_dir = './imdb_data_8k'
    (train_data, test_data), info = download_imdb_data(local_data_dir)
    
    train_dataset, test_dataset = padded_batch_dataset(args, train_data, test_data)

    model = ConvModel()
    model(next(iter(train_dataset))[0])
       
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, 
        beta_1=args.beta_1,
        beta_2=args.beta_2
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = loss_fn(labels, logits)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_accuracy(logits, labels)
        return 
        
        
    @tf.function
    def test_step(inputs, labels):
        logits = model(images, training=False)
        loss = loss_fn(labels, logits)
        test_loss(loss)
        test_accuracy(logits, labels)
        return
    
    
    logger.info("Training starts ...")
    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        for text, target in train_dataset:
            train_step(inputs=text, labels=target)

        logger.info(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result()}, '
        )
        
        for text, target in test_dataset:
            test_step(inputs=text, labels=target)

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
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)