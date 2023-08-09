# import numpy as np
# from tensorflow import keras
# import R
# from models import ModelInfo
#
#
# # .....................  Artificial Neural Network (ANN)  .........................
#
# def train_ann():
#     dataset = R.DigitDataset.get_singleton()
#
#     model = keras.Sequential([
#         keras.layers.Dense(128, activation=keras.activations.relu, input_shape=(dataset.img_pixels,)),
#         keras.layers.Dense(32, activation=keras.activations.relu),
#         keras.layers.Dense(10, activation=keras.activations.softmax)
#     ])
#
#     model.summary()
#     model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
#                   metrics=['accuracy'])  # directly using Accuracy() instance causes errors
#
#     model.fit(dataset.x_train_flattened(), dataset.y_train, batch_size=50, epochs=10)
#     MODEL_INFO_ANN.save_model(model)
#
#     loss, acc = model.evaluate(dataset.x_test_flattened(), dataset.y_test)
#     print(f"\n (DNN) Loss: {loss}, Accuracy: {acc}")
#     return model, acc
#
#
# def predict_ann(model, images_array: np.ndarray):
#     samples = images_array.shape[0]
#
#     images_array = images_array.reshape((samples, -1))      # Flatten samples
#     pred = model.predict(images_array)
#
#     out = np.zeros(samples, dtype=np.int32)
#     for r in range(samples):
#         out[r] = np.argmax(pred[r])
#     return out
#
#
# MODEL_INFO_ANN = ModelInfo("ANN",
#                            "Artificial Neural Net",
#                            "ann_digits.keras",
#                            model_loader=R.load_keras_model,
#                            model_saver=R.save_keras_model,
#                            model_predictor=predict_ann)
#
#
# def test_saved_ann():
#     # # Loading saved model
#     model = MODEL_INFO_ANN.load_model()
#     if not model:
#         return
#
#     dataset = R.DigitDataset.get_singleton()
#     loss, acc = model.evaluate(dataset.x_test_flattened(), dataset.y_test)
#     print(f"\n (DNN) Loss: {loss}, Accuracy: {acc}")
#
#
