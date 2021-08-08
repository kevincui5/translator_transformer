# -*- coding: utf-8 -*-
from util import print_translation, CustomSchedule
from transformer import Transformer
import tensorflow as tf
import tensorflow_text as text
import logging

checkpoint_path = 'trained_model'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def predict(sentence, max_length=40):
  tokenizers = tf.saved_model.load('ted_hrlr_translate_pt_en_converter')
  # inp sentence is portuguese, hence adding the start and end token
  sentence = tf.convert_to_tensor([sentence])
  sentence = tokenizers.pt.tokenize(sentence).to_tensor()

  encoder_input = sentence

  # as the target is english, the first word to the transformer should be the
  # english start token.
  start, end = tokenizers.en.tokenize([''])[0]
  output = tf.convert_to_tensor([start])
  output = tf.expand_dims(output, 0)

  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1
  
  transformer = Transformer(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      input_vocab_size=tokenizers.pt.get_vocab_size(),
      target_vocab_size=tokenizers.en.get_vocab_size(),
      pe_input=1000,
      pe_target=1000,
      rate=dropout_rate)
  
  learning_rate = CustomSchedule(d_model)
  checkpoint_path = "./trained_model"
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)  
  ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
  ckpt.restore(ckpt_manager.latest_checkpoint)
  for i in range(max_length):
    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
    #     encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer((encoder_input, output))

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.argmax(predictions, axis=-1)

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == end:
      break

  # output.shape (1, tokens)
  text = tokenizers.en.detokenize(output)[0]  # shape: ()

  tokens = tokenizers.en.lookup(output)[0]

  return text, tokens, attention_weights

sentence = "este é um problema que temos que resolver."
ground_truth = "this is a problem we have to solve ."

translated_text, translated_tokens, attention_weights = predict(sentence)
print_translation(sentence, translated_text, ground_truth)

sentence = "os meus vizinhos ouviram sobre esta ideia."
ground_truth = "and my neighboring homes heard about this idea ."

translated_text, translated_tokens, attention_weights = predict(sentence)
print_translation(sentence, translated_text, ground_truth)

sentence = "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram."
ground_truth = "so i \'ll just share with you some stories very quickly of some magical things that have happened ."

translated_text, translated_tokens, attention_weights = predict(sentence)
print_translation(sentence, translated_text, ground_truth)