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
