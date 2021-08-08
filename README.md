
# Neural Machine Translation with Transformer Model

![Compatibility](img/Python-3.7-blue.svg)![Compatibility](img/Tensorflow-2.4-blue.svg)

In this tutorial I would like to improve the [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer)  tutorial from tensorflow website since the code is for demonstration purpose.  I'd like to make it more "object oriented" by using tensorflow 2 features such as subclass Keras layers and models classes and use Keras model's build-in compile and fit function.  Doing so will make the code easier to understand, make change, and maintain.

I have done similar thing in [my other tutorial](https://github.com/kevincui5/translator_tf) on NMT with attention machanism.

## Difference between transformer model and encoder-decoder attention mechanism
Transformer is no longer a RNN, and  that is why you don't see LTSM or GRU any where in the Transformer encoder and decoder.  Instead, there are causal attention layers which no longer do sequantial computation per layer, but in a single step.  For details please see the original tutorial.


In order to use Keras model.fit() function for training, our model's call function need to have signature of self, data, and training:

```
class Transformer(tf.keras.Model):
    ...
    def call(self, data, training=True):
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
Our transformer, a custom keras model, inherits the training argument from Layer, knows by itself when to pass True or False to training argument.  For example, when you call model.fit, it will pass True to Transformer's call function and trickle down to encoder and decoder layer and tell for example dropout layer to behave accordingly.
```
transformer.fit(train_batches, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
```
This is run in training mode
```
predictions, attention_weights = transformer((encoder_input, output))

```
This is run in inference mode, no need to explicitly specify train mode

By assigning Keras Layers' instance as attributes of other Layers, all the weights, including that of inner layers become trackable:
```
gradients = tape.gradient(loss, self.trainable_variables)
```      

Note: we lazily create layers' weights during layers' instantiation like Kera's best practice guide suggests.  Notice there is no input_shape specified in lstm __init__() or input_length in embedding init():

```
class Encoder(Layer):
  def __init__(self, vocab_size, embedding_dim, units, 
               name="Encoder", **kwargs):
    super(Encoder, self).__init__(**kwargs)
    self.units = units
    self.embedding = Embedding(vocab_size, embedding_dim)
    self.bi_LSTM_last = Bidirectional(LSTM(units, return_sequences=True, 
                                      return_state = True))
    self.bi_LSTM = Bidirectional(LSTM(units, return_sequences=True))
```


## Loss Function
For the loss function, you can either implement one like that in the original tutorial, or use the Keras' build-in loss functions passed in through model.compile():

`
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
`

`
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
`

Also in the training loop, we can use the specified loss object and metrics objects from model.compile function inside train_step function because we are overriding train_step:
```
...
loss += self.compiled_loss(targ[:, t], predictions)
self.compiled_metrics.update_state(targ[:, t], predictions)
...
self.optimizer.apply_gradients(zip(gradients, trainable_vars))
...
```
Notice we pass from_logits=True to the loss function object for numerical stability.  We have to use linear activation function in the Dense layer in Decoder then.  The loss function object will take care of softmax function part.  See [this](https://stackoverflow.com/questions/52125924/why-does-sigmoid-crossentropy-of-keras-tensorflow-have-low-precision/52126567#52126567) for detail.

## Override train_step()
By subclassing Keras Model class and overriding train_step(), we can use model.fit() to train our model.  We can now take advantage of all the functionality that come with fit(), such as callbacks, batch and epoch handling, validation set metrics monitoring, and custom training loop logic etc.

`model.fit(dataset_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=2, callbacks=[early_stopping], validation_data=dataset_valid)`

We are also able to provide different logic in back prop, which is in train_step(), from that in forward pass, which is inside model's call() function.
```
def train_step(self, data):
      inp, targ = data
      loss = 0
      with tf.GradientTape() as tape:
        enc_output, enc_hidden, enc_cell = self.encoder(inp)    
        dec_hidden = enc_hidden
        dec_cell = enc_cell
        dec_input = tf.expand_dims(targ[:,0], 1) 
        # Teacher forcing - feeding the target (ground truth) as the next decoder input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder. predictions shape == (batch_size, vocab_tar_size)
          predictions, dec_hidden, dec_cell, _ = self.decoder(dec_input, dec_hidden,
                                                         dec_cell, enc_output) #throws away attension weights
          #targ shape == (batch_size, ty)
          loss += self.compiled_loss(targ[:, t], predictions) #take parameters of ground truth and prediction
          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)  # targ[:, t] is y
          self.compiled_metrics.update_state(targ[:, t], predictions)

      trainable_vars = self.trainable_variables    
      gradients = tape.gradient(loss, trainable_vars)    
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      # Return a dict mapping metric names to current value
      return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}
```
Here, teacher's forcing is used, just like in the original Keras tutorial, where ground truth at time step t is extracted and feed in as decoder's input.


Here in the forward pass, at each time step, y_pred is appended to TensorArray.  Then at the end the prediction tensor is transposed so that first dimension is batch size, second is timestep.  greedy sampling method is used here, where max is applied on the probability distribution and converted to word index.  Beam search can be implemented here.
Again TensorArray is used here instead of python list to avoid potential python side effects as suggested by tensorflow guide as best practice.

We can also override test_step() to provide custom evaluation logic, so we can use model.evaluate() with all the functionality it brings to monitor the loss and metrics on validation set. 
```
def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(data, training=False)

        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
```
Inside the function it simply calls model's call() function which just invokes forward pass logic that records loss and metrics.


## Requirements:
 * Python 3, Tensorflow 2.4
 * pip install -q -U tensorflow-text
 pip install -q tensorflow_datasets
## What Each File Does: 
 * trainer6/BahdanauAttention.py: Define attention layer.  Uses "add" attention mechanism. 
 * trainer6/Decoder.py: Define decoder layer, which contains a single LSTM layer.
 * trainer6/Encoder.py: Define encoder layer, which contains a single Bi-LSTM layer.
 * trainer6/model.py: Get input and target tokenizers, max length of input and target language sentences from the "full" dataset file, and read in training data from training file and validation data from validation file and does the training.
 * trainer6/task.py: Parse the commad arguments.
 * trainer6/Translator.py: Define the Translator model, which contains reference to encoder and decoder layers.  Also contains the overriden train_step and test_step functions and translate function, which is just prediction on a single example.
 * trainer6/util.py: All the utility functions and classes used by model.py and eval6.py.  Many from the Keras NMT tutorial.
 * config.yaml: For Google cloud use. See train-gcp.sh.
 * deu.txt: Raw translation input file.  See prepare_input_files.py.
 * english-german-x.csv: x represent the number of examples in "full" dataset file. Contains tab seperated language pairs examples.  Created by prepare_input_files.py

 * english-german-test-x.csv: Language pairs examples for testing.
 * english-german-train-x.csv: Language pairs examples for training.
 * english-german-valid-x.csv: Language pairs examples for validation.
 * eval6.py: Get input and target tokenizers, max length of input and target language sentences from the "full" dataset file, and read in training data from training file and validation data from validation file and does the evaluation in metrics from model.compile and in BLEU scores.
 * prepare_input_files.py: process sentences pair in deu.txt and converts it to english-german-x.csv.  Also split the "full" dataset file into training, test, and validation dataset files.  The ratio can be changed.
 * train-gcp.sh: A shell script submitting training job to AI platform service from Google Cloud.  Modify the config.yaml to configure the cloud server instance. 
 * train-local6.sh: Run the training job locally through gcloud.  Google cloud sdk required.

## Usage

### Preparing Raw Input Files
 1) Download the input file here at [Language dataset source: www.ManyThings.org](http://www.manythings.org/anki/) and unzip to deu.txt. It looks like this:
 ```
 ...
She is kind.	Sie ist liebenswürdig.
She woke up.	Sie wachte auf.
She's a dog.	Sie hat einen Hund.
She's happy.	Sie ist glücklich.
Show him in.	Bring ihn herein.
Show him in.	Bringen Sie ihn herein.
Sit with me.	Setz dich zu mir!
Stand aside.	Geh zur Seite!
Stand aside.	Gehen Sie zur Seite!
...
```

 2) Put deu.txt in the same directory as  prepare_input_files.py, configure and execute it.  It produces 4 files, a full, a training , a test and a validation dataset file:
 
   ``
 python prepare_input_files.py
 ``
```
Saved: english-german-180000.csv
[Hi.] => [Hallo!]
[Hi.] => [Grüß Gott!]
[Run!] => [Lauf!]
[Wow!] => [Potzdonner!]
[Wow!] => [Donnerwetter!]
[Fire!] => [Feuer!]
[Help!] => [Hilfe!]
[Help!] => [Zu Hülf!]
[Stop!] => [Stopp!]
[Wait!] => [Warte!]
Saved: english-german-train-180000.csv
Saved: english-german-test-180000.csv
Saved: english-german-valid-180000.csv
```
Note: you can change how many examples you'd like to have and the ratio between training and testing/validation in prepare_input_files.py.



### Training
 1) Train locally: 
 
 ``
 ./train-local6.sh
 ``
    Note: change parameters in train-local6.sh such as example_limit, batch size, epoch, etc.  example_limit is the total example size in the full dataset file, which is the sum of size of training, testing and validation set.
    
  3) The model is saved in trained_model6_x directory where x is the size of full dataset.  If the training is very long and you want the training to be continued after an interruption, you can define an callback that save the model and optimizer at some interval of epochs.
  
  
### Translating
Make sure eval6.py is in same directory as trained_model6_x.  Evaluation is first done by calculating bleu scores on a single batch of training set.  It then calls model.translate() function on each example to get the translated sentence, and with the original sentence and target sentence the bleu scores on the whole set are calculated.

example_limit needs to match the size of full dataset file.  Set the value in eval6.py.  Let's use a smaller example set first, say 40,000.  So configure prepare_input_files.py and execute it to generate the input files.  Then execute eval6.py:

```
Score on training set:
...
BLEU-1: 0.973352
BLEU-2: 0.958513
BLEU-3: 0.952494
BLEU-4: 0.935998
```

```
Score on validation set:
...
BLEU-1: 0.791289
BLEU-2: 0.688844
BLEU-3: 0.651557
BLEU-4: 0.572273
```
see the difference between the bleu scores on validation set are quite big because of overfitting on small dataset.
Now let's prepare 180000 input files by configure and executing prepare_input_files.py, then execute eval6.py:

 ``
python3 eval6.py
  ``
 
```
Score on training set:
...
src=[<start> ich spiele gern poker . <end> ], target=[<start> i like to play poker . <end> ], predicted=[<start> i like playing golf . <end> ]
...
BLEU-1: 0.961240
BLEU-2: 0.942952
BLEU-3: 0.936637
BLEU-4: 0.917990
...
```


```
Score on validation set:
...
BLEU-1: 0.950670
BLEU-2: 0.928722
BLEU-3: 0.921977
BLEU-4: 0.900512
```
Now there is little difference between scores on training set and validation set, we can see larger training set overcomes overfitting issue.

Finally, we can call evaluation to check on this model's loss and accuracy on test set:

`model.evaluate(dataset_test)`
 
```
1/1 [==============================] - ETA: 0s - loss: 10.8105 - sparse_categori1/1 [==============================] - 18s 18s/step - loss: 10.8105 - sparse_categorical_crossentropy: 6.4926 - sparse_categorical_accuracy: 0.4898
```
To see bleu score in the metrics we would have to implement a user-defined metrics class and pass it in model.compile() function, but I am not sure how to do that in graph mode since that seems to require processing each example in a batch in the test_step().

<!--<div align="left">-->
  <!--<br><br><img  width="100%" "height:100%" "object-fit: cover" "overflow: hidden" src=""><br><br>-->
<!--</div>-->

## Model Diagram
![alt text](img/training_model.png)
![alt text](img/inference_model.png)



## References

 * [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer) 

 

