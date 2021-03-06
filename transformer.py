# -*- coding: utf-8 -*-
import tensorflow as tf
from model import Encoder, Decoder

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, data, training):
    inp, tar = data
    enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(inp,
        tar, enc_output, training)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

  def train_step(self, data):
    inp, tar = data
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    with tf.GradientTape() as tape:
      predictions, _ = self((inp, tar_inp), True)
      #loss = self.loss_function(tar_real, predictions)
      loss = self.compiled_loss(tar_real, predictions)
  
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(tar_real, predictions)
    return {m.name: m.result() for m in self.metrics}
