import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

def build_classifier_model(tf_preprocess, tf_encoder):
   text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
   preprocessing_layer = hub.KerasLayer(tf_preprocess, name='preprocessing')
   encoder_inputs = preprocessing_layer(text_input)
   encoder = hub.KerasLayer(tf_encoder, trainable=True, name='BERT_encoder')
   outputs = encoder(encoder_inputs)
   net = outputs['pooled_output']
   net = tf.keras.layers.Dropout(0.1)(net)
   net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)

   return tf.keras.Model(text_input, net)

def main():
  tf.get_logger().setLevel('ERROR')
  
  AUTOTUNE = tf.data.AUTOTUNE
  batch_size = 32
  seed = 42

  raw_train_ds = tf.keras.utils.text_dataset_from_directory(
     'aclImdb/train',
     batch_size=batch_size,
     validation_split=0.2,
     subset='training',
     seed=seed
  )

  class_names = raw_train_ds.class_names
  train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

  val_ds = tf.keras.utils.text_dataset_from_directory(
     'aclImdb/train',
     batch_size=batch_size,
     validation_split=0.2,
     subset='validation',
     seed=seed
  )

  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
     batch_size=batch_size
  )

  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

  for text_batch, label_batch in train_ds.take(1):
    for i in range(0, 3):
      print(f'Review: {text_batch.numpy()[i]}')
      label = label_batch.numpy()[i]
      print(f'Label : {label} ({class_names[label]})')

  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  preprocessor = hub.KerasLayer("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
  encoder_inputs = preprocessor(text_input)
  encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/albert_en_base/3",
    trainable=True
  )
  outputs = encoder(encoder_inputs)
  pooled_output = outputs["pooled_output"]      # [batch_size, 768].
  sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
  embedding_model = tf.keras.Model(text_input, pooled_output)

  text_test = ['this is such an amazing movie!']
  text_preprocessed = preprocessor(text_test)
  print(f'Keys       : {list(text_preprocessed.keys())}')
  print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
  print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
  print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
  print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

  bert_results = encoder(text_preprocessed)

  print(f'Loaded BERT: albert')
  print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
  print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
  print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
  print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


if __name__ == '__main__':
    main()