Contained in here will be models that are decent or just collected over time. 


# 61.4-acc-loss-2.39-top.model:
Overall issues here:
val acc/loss is far worse than in-sample. We've got some serious overfitment happening for the in-sample data. 

I have various ideas to help fix this. In-sample data is highly similar to other samples. The FFT from frame-to-frame is going to be very similar. Might be wise to make training data every `n` samples. Preliminary tests doing this have not yielded any more reliable information, but if you do `n=5` this means you have 1/5th the training data...which obviously is pretty brutal. Will have to...make more data!

```py
model = Sequential()

model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv1D(128, (2)))
model.add(Activation('relu'))

model.add(Conv1D(128, (2)))
model.add(Activation('relu'))

model.add(Conv1D(64, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(64, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```


# all-cnn model (63.23-acc-loss-2.52.model)

Overall issues stay the same as for `61.4-acc-loss-2.39-top.model` model.

This model achives a bit higher validation accuracy and accuracy is more stable across epochs during training.

```py
model = Sequential()

model.add(Conv1D(64, (5), padding='same', input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv1D(128, (5), padding='same'))
model.add(Activation('relu'))

model.add(Conv1D(256, (5), padding='same'))
model.add(Activation('relu'))

model.add(Conv1D(512, (5), padding='same'))
model.add(Activation('relu'))

model.add(Conv1D(3, (16)))
model.add(Reshape((3,)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

This model does not make a use of Densely connected layers, output layer is shaped to have 3 outputs.

Accuracy of 63.26% was achieved by additional clipping and scaling of data of samples:

```py
train_X = np.clip(np.array(train_X).reshape(reshape), -10, 10) / 10
test_X = np.clip(np.array(test_X).reshape(reshape), -10, 10) / 10
```

Another working approach:

```py
train_X = np.clip(np.array(train_X).reshape(reshape), -3, 3)
test_X = np.clip(np.array(test_X).reshape(reshape), -3, 3)
```

Code for this model: https://gist.github.com/daniel-kukiela/8282612a23c9646cc8314bf3b3905d85

Remember to update `analysis.py` with same type of scaling on `X` before predicting.

Confusion matrix for this trained model:

![confusion matrix](https://pythonprogramming.net/static/images/bci/currentbest.png)
