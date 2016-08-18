This post basically takes the tutorial on [Classifying MNIST digits using Logistic Regression](http://deeplearning.net/tutorial/logreg.html#logreg) which is primarily written for Theano and attempts to port it to Keras.

## What is Keras?

This is what the official [Keras](https://keras.io) site says.

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano.

So, what better way to put that claim to the test than to write some code!

Keras comes with great documentation. One can _really_ get up and running in a matter of minutes. Everything needed to accomplish the goal can be found on the [Guide to Sequential Model](https://keras.io/getting-started/sequential-model-guide/) page (assuming of course the initial setup and configuration is all taken care of).

## Getting the data

Keras also offers a collection of [datasets](https://keras.io/datasets/) that can be used to train and test the model. The [MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits) set is a part of the available datasets and can be loaded as shown below.

    from keras.datasets import mnist
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

### Reshaping and normalizing the inputs

    input_dim = 784  #28*28
    X_train = X_train.reshape(60000, input_dim)
    X_test = X_test.reshape(10000, input_dim)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

### Convert class vectors to binary class matrices

    from keras.utils import np_utils
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

## Build the model

    from keras.models import Sequential
    from keras.layers import Dense, Activation
    output_dim = nb_classes = 10
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    batch_size = 128
    nb_epoch = 20

## Compile the model

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    ## Save model and weights
    json_string = model.to_json() # as json
    open('mnist_Logistic_model.json', 'w').write(json_string)
    yaml_string = model.to_yaml() #as yaml
    open('mnist_Logistic_model.yaml', 'w').write(yaml_string)
    # save the weights in h5 format
    model.save_weights('mnist_Logistic_wts.h5')
    # uncomment the code below (and modify accordingly) to read a saved model and weights
    # model = model_from_json(open('my_model_architecture.json').read()) # if json
    # model = model_from_yaml(open('my_model_architecture.yaml').read()) # if yaml
    # model.load_weights('my_model_weights.h5')

And that’s it! Full source code available [here](https://github.com/the1ju/Keras4DLTutorials) 

