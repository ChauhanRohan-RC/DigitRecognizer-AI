from tensorflow import keras
import R


def create_and_test_neural_network():
    dataset = R.DigitDataset.get_singleton()

    model = keras.Sequential([
        keras.layers.Dense(128, activation=keras.activations.relu, input_shape=(dataset.img_pixels,)),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])  # directly using Accuracy() instance causes errors
    model.fit(dataset.x_train, dataset.y_train, batch_size=50, epochs=10)
    R.save_keras_model(model, R.FILE_NAME_MODEL_KERAS_DNN)

    loss, acc = model.evaluate(dataset.x_test, dataset.y_test)
    print(f"\n (DNN) Loss: {loss}, Accuracy: {acc}")
    return model, acc


def test_saved_model():
    # # Loading saved model
    model = R.load_keras_model(R.FILE_NAME_MODEL_KERAS_DNN)
    if not model:
        return

    dataset = R.DigitDataset.get_singleton()
    loss, acc = model.evaluate(dataset.x_train, dataset.y_train)
    print(f"\n (DNN) Loss: {loss}, Accuracy: {acc}")
