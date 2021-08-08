# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# for Positional encoding
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

#   # add extra dimensions to add the padding
#   # to the attention logits.
#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# def loss_function(real, pred):
#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   loss_ = loss_object(real, pred)

#   mask = tf.cast(mask, dtype=loss_.dtype)
#   loss_ *= mask

#   return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


# def accuracy_function(real, pred):
#   accuracies = tf.equal(real, tf.argmax(pred, axis=2))

#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   accuracies = tf.math.logical_and(mask, accuracies)

#   accuracies = tf.cast(accuracies, dtype=tf.float32)
#   mask = tf.cast(mask, dtype=tf.float32)
#   return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# def create_masks(inp, tar):
#   # Encoder padding mask
#   enc_padding_mask = create_padding_mask(inp)

#   # Used in the 2nd attention block in the decoder.
#   # This padding mask is used to mask the encoder outputs.
#   dec_padding_mask = create_padding_mask(inp)

#   # Used in the 1st attention block in the decoder.
#   # It is used to pad and mask future tokens in the input received by
#   # the decoder.
#   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#   dec_target_padding_mask = create_padding_mask(tar)
#   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#   return enc_padding_mask, combined_mask, dec_padding_mask

def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')