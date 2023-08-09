# import numpy as np
# from tensorflow import keras
# import R
# from models import ModelInfo
#
#
# # ................  Convolutional Neural Network (CNN)  .................
#
# def train_cnn():
#     dataset = R.DigitDataset.get_singleton()
#
#     model = keras.Sequential([
#         keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu",
#                             input_shape=(*dataset.img_shape, 1)),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dense(64, activation=keras.activations.relu),
#         keras.layers.Dense(10, activation=keras.activations.softmax)
#     ])
#
#     model.summary()
#     model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
#                   metrics=['accuracy'])  # directly using Accuracy() instance causes errors
#
#     x_train = dataset.x_train(shape=(*dataset.org_x_train_shape, 1))
#     model.fit(x_train, dataset.y_train, batch_size=50, epochs=10)
#
#     # Saving
#     MODEL_INFO_CNN.save_model(model)
#
#     x_test = dataset.x_test(shape=(*dataset.org_x_train_shape, 1))
#     loss, acc = model.evaluate(x_test, dataset.y_test)
#
#     print(f"\n (CNN) Loss: {loss}, Accuracy: {acc}")
#     return model, acc
#
#
# # def predict_single_cnn(model, img: np.ndarray):
# #     img = img.reshape((1, *img.shape, 1))
# #     pred = model.predict(img)
# #     return np.argmax(pred[0])
#
#
# def predict_cnn(model, images_array: np.ndarray):
#     images_array = images_array.reshape((*images_array.shape, 1))
#     pred = model.predict(images_array)
#     samples = pred.shape[0]
#
#     out = np.zeros(samples, dtype=np.int32)
#     for r in range(samples):
#         out[r] = np.argmax(pred[r])
#     return out
#
#
# def test_saved_cnn():
#     # # Loading saved model
#     model = MODEL_INFO_CNN.load_model()
#     if not model:
#         return
#
#     dataset = R.DigitDataset.get_singleton()
#     x_test = dataset.x_test(shape=(*dataset.org_x_train_shape, 1))
#
#     loss, acc = model.evaluate(x_test, dataset.y_test)
#     print(f"\n (CNN) Loss: {loss}, Accuracy: {acc}")
#
#
# MODEL_INFO_CNN = ModelInfo("CNN",
#                            "Convolutional Neural Net",
#                            "cnn_digits.keras",
#                            model_loader=R.load_keras_model,
#                            model_saver=R.save_keras_model,
#                            model_predictor=predict_cnn)
