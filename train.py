# -*- coding: utf-8 -*-
import logging
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text
from util import positional_encoding, CustomSchedule
from transformer import Transformer
from tensorflow.keras.callbacks import EarlyStopping

n, d = 2048, 512
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
max_input_len = 1000

model_name = 'ted_hrlr_translate_pt_en_converter'
# tf.keras.utils.get_file(
# f"{model_name}.zip",
# f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
# cache_dir='.', cache_subdir='', extract=True)    
tokenizers = tf.saved_model.load(model_name)

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size(),
    target_vocab_size=tokenizers.en.get_vocab_size(),
    pe_input=max_input_len,
    pe_target=max_input_len,
    rate=dropout_rate)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                           data_dir=".", download=False, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en

EPOCHS = 10
BUFFER_SIZE =  len(train_examples)
BATCH_SIZE = 64
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))
#encoded = tokenizers.en.tokenize(en_examples)
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

pos_encoding = positional_encoding(n, d)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

checkpoint_path = 'trained_model'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')
  
train_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=['SparseCategoricalCrossentropy', 'SparseCategoricalAccuracy']

transformer.compile(optimizer, train_loss, metrics)  
transformer.fit(train_batches, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)

