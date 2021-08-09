
# Neural Machine Translation with Transformer Model

In this tutorial I would like to improve the [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer)  tutorial from tensorflow website by using some of the tensorflow 2 features such as subclassing Keras layers and models classes and use Keras model's build-in compile and fit function for training and evaluation. The coding in original tutorial is maily for demonstration purpose.  By using the above mentioned features in tensorflow 2 it will make the program more "object oriented" and the code easier to understand, make change, and maintain.

I have done similar thing in [my other tutorial](https://github.com/kevincui5/translator_tf) on NMT with attention machanism.

## Difference between transformer model and encoder-decoder attention mechanism
Transformer is no longer a RNN, and  that is why you don't see LTSM or GRU any where in the Transformer encoder and decoder.  Instead, there are causal attention layers which no longer do sequantial computation per layer, but in a single step.  For details please see the original tutorial.


In order to use Keras model.fit() function for training, our model's call function need to have signature of self, data, and training:

```
class Transformer(tf.keras.Model):
    ...
    def call(self, data, training):
    ...
```      
All the masks parameters in original tutorial can be moved to inner layers' implementations.  For example in Encoder class:
```
class Encoder(tf.keras.layers.Layer):
    ...
    def call(self, x, training):
        mask = self.create_padding_mask(x)
        ...
```
Our transformer, a custom keras model, inherits the training argument from Layer, pass True or False to training argument implicitly.  For example, when you call model.fit, it will pass True implicitly to Transformer's call function and trickle down to encoder and decoder layer and tell for example dropout layer to behave in training or inference mode accordingly.
```
transformer.fit(train_batches, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
```
This is run in training mode
```
predictions, attention_weights = transformer((encoder_input, output))

```
This is run in inference mode, no need to explicitly specify train mode

Another neat thing is that by assigning Keras Layers' instance as attributes of other Layers, all the weights, including that of inner layers become trackable:
```
gradients = tape.gradient(loss, self.trainable_variables)
```      
This makes the code look so much cleaner.

As always we don't specify input shape during layers' instantiation as best practice.  Notice there is no input_shape specified in embedding init().  The d_model is embedding dimension and input_vocab_size is the total vocabulary count:

```
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    ...
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
```


## Loss Function
For the loss function, you can either implement one like that in the original tutorial, which a mask is used to exclude the padding in the input data during training, a small optimization, or you can create a custom loss and metrics classes and pass them in model.compile() function.  In this tutorial, I just use the Keras' build-in loss functions and pass it in model.compile() in train.py:

`
train_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=['SparseCategoricalCrossentropy', 'SparseCategoricalAccuracy']
`

`
transformer.compile(optimizer, train_loss, metrics)
`

Also in the training loop, we can use the specified loss object and metrics objects from model.compile function inside train_step function because we are overriding train_step:
```
class Transformer(tf.keras.Model):
    def train_step(self, data):
        ...
        loss += self.compiled_loss(targ[:, t], predictions)
        self.compiled_metrics.update_state(targ[:, t], predictions)
        ...
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
...
```
Note: there is no need to decorate train_step with tf.function because train_step() will be executed in graph mode by default.

## Override train_step()
By subclassing Keras Model class and overriding train_step(), we can use model.fit() to train our model.  We can now take advantage of all the functionality that come with fit(), such as callbacks, batch and epoch handling, validation set metrics monitoring, and custom training loop logic etc.

`model.fit(dataset_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=2, callbacks=[early_stopping], validation_data=dataset_valid)`

Also, there is no need to reset loss and metrics state manually at each start of epoch, it's all been taken care of.
```
class Transformer(tf.keras.Model):
    def train_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
          predictions, _ = self((inp, tar_inp), True)
          loss = self.compiled_loss(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(tar_real, predictions)
        return {m.name: m.result() for m in self.metrics
```
Teacher's forcing is used here.  The coding is straightforward, unlike that in encoder-decoder model with attention, where attention is calculated at each decoder time step.

You can also overwrite test_step to provide custom evaluation logic.
```
class Transformer(tf.keras.Model):
    def test_step(self, data):
        ...
```

## Requirements:
 * Python 3, Tensorflow 2.4
 * You need tensorflow-text for tokenizing your input text
 ```
 pip install -q -U tensorflow-text
 ```
 * You need tensorflow datasets package for easy dataset import.
 ```
 pip install -q tensorflow_datasets
 ```
 
## What Each File Does: 
 * ted_hrlr_translate/:  Portuguese-English translation dataset from the TED Talks Open Translation Project. It is already downloaded for you.  You can download the dataset by yourself by setting download to True:
 ```
 examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                           data_dir=".", download=True, as_supervised=True)
 ```
 * ted_hrlr_translate_pt_en_converter/: The keras model of tokenizers of pt and en.  It is already downloaded for you.  You can download it by yourself by uncomment the following lines:
 ```
 # tf.keras.utils.get_file(
 # f"{model_name}.zip",
 # f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
 # cache_dir='.', cache_subdir='', extract=True)
 ```

 * eval.py: Load the keras model saved in training and translate some sample pt sentences into en.
 * model.py: All the layers classes used in transformer model.
 * train.py: train the transformer model.
 * transformer.py: Define the transformer model, which contains reference to encoder and decoder layers.  Also contains the overriden train_step logic.
 * util.py: All the utility functions and classes used by model.py, transformer.py and eval.py.  Most from the original tutorial.
 
## Usage
### Training
 1) Training.  Change epoch or batch size to fit your machine capability: 
 
 ``
 python train.py
 ``
    
 2) The model is saved in trained_model directory.
  
  
### Translating
 ``
python eval.py
  ``

<!--<div align="left">-->
  <!--<br><br><img  width="100%" "height:100%" "object-fit: cover" "overflow: hidden" src=""><br><br>-->
<!--</div>-->




## References

 * [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer) 

 

