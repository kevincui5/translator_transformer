# -*- coding: utf-8 -*-
import tensorflow as tf
from model import Encoder, Decoder
from util import CustomSchedule

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, data, training=True):
    inp, tar = data
    enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(inp,
        tar, enc_output, training)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

  def train_step(self, data):
    print(data)
    inp, tar = data
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    with tf.GradientTape() as tape:
      predictions, _ = self((inp, tar_inp), True)
      loss = self.loss_function(tar_real, predictions)
  
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return {m.name: m.result() for m in self.metrics}
  
  def loss_function(self, real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')
      loss_ = loss_object(real, pred)
    
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
    
      return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
  def accuracy_function(self, real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
  
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
  
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    