import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import neighbors, metrics, svm
from tensorflow import keras
import R


class ModelInfo:
    _sId = 100
    _reg_arr = []
    _reg_dict = {}

    @classmethod
    def _next_id(cls) -> int:
        _id = cls._sId
        cls._sId += 1
        return _id

    @classmethod
    def _register(cls, model_info):
        cls._reg_dict[model_info.id] = model_info
        cls._reg_arr.append(model_info)

    @classmethod
    def from_id(cls, _id):
        return cls._reg_dict[_id]

    @classmethod
    def get_all(cls):
        return cls._reg_arr.copy()

    def __init__(self, short_label: str, long_label: str,
                 file_name: str,
                 model_loader: callable,  # will receive file_name as argument
                 model_saver: callable,  # will receive model and file_name as argument
                 model_predictor: callable):  # arguments : (model, images_array), Output : array of predicted labels

        self.id = self.__class__._next_id()
        self.short_label = short_label
        self.long_label = long_label
        self.display_name = short_label
        self.file_name = file_name
        self.model_loader = model_loader
        self.model_saver = model_saver
        self.model_predictor = model_predictor

        self.__class__._register(self)

    def __eq__(self, other):
        return (type(other) == type(self)
                and other.id == self.id
                and other.file_name == self.file_name
                and other.short_label == self.short_label
                and other.long_label == self.long_label)

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"Model(id={self.id}, short_label={self.short_label}, long_label={self.long_label})"

    def __str__(self):
        return self.__repr__()

    def load_model(self):
        return self.model_loader(self.file_name)

    def save_model(self, model):
        self.model_saver(model, self.file_name)

    def predict(self, model, images_array):
        return self.model_predictor(model, images_array)


# ...................  K-Nearest Neighbours (KNN)  ...........................
# experimentally, best k=3

def train_knn(k):
    dataset = R.DigitDataset.get_singleton()

    model = neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(dataset.x_train_flattened(), dataset.y_train)

    y_pred = model.predict(dataset.x_test_flattened())
    acc = metrics.accuracy_score(dataset.y_test, y_pred)

    print(f"(KNN) k: {k}, Accuracy: {acc * 100: .2f}")
    return model, acc


def predict_knn(model, images_array: np.ndarray):
    images_array = images_array.reshape((images_array.shape[0], -1))  # flatten images
    pred = model.predict(images_array)
    return pred


MODEL_INFO_KNN = ModelInfo("KNN",
                           "K-Nearest Neighbours",
                           file_name="knn_digits.gzip",
                           model_loader=R.load_sklearn_model,
                           model_saver=R.save_sklearn_model,
                           model_predictor=predict_knn)


def find_best_knn(k_range=None, plot=True, live_save=True):
    # Finding optimal value of hyperparameter k
    if not k_range:
        k_range = list(range(1, 7))

    highest_acc = 0
    best_model = None
    accuracies = []
    for k in k_range:
        model, acc = train_knn(k)
        accuracies.append(acc * 100)
        if acc > highest_acc:
            highest_acc = acc
            best_model = model
            if live_save:
                MODEL_INFO_KNN.save_model(best_model)

    if not live_save and best_model:
        MODEL_INFO_KNN.save_model(best_model)

    if plot:
        sb.set_style('darkgrid')
        plt.title("KNN Classifier")
        plt.xlabel("No of Nearest Neighbours (k)")
        plt.ylabel("Accuracy (%)")
        sb.lineplot(x=k_range, y=accuracies)
        plt.show()

    return best_model, highest_acc


# ....................  Support Vector Machine (SVM) ...............................
# experimentally, best kernel = rbf

def train_svm(C=1.0, kernel='rbf'):
    model = svm.SVC(C=C, kernel=kernel)

    dataset = R.DigitDataset.get_singleton()
    model.fit(dataset.x_train_flattened(), dataset.y_train)

    y_pred = model.predict(dataset.x_test_flattened())
    acc = metrics.accuracy_score(dataset.y_test, y_pred)
    print(f"(SVM) kernel: {kernel}, Accuracy: {acc * 100: .2f}")
    return model, acc


def predict_svm(model, images_array: np.ndarray):
    images_array = images_array.reshape((images_array.shape[0], -1))  # flatten images
    pred = model.predict(images_array)
    return pred


MODEL_INFO_SVM = ModelInfo("SVM",
                           "Support Vector Machine",
                           "svm_digits.gzip",
                           model_loader=R.load_sklearn_model,
                           model_saver=R.save_sklearn_model,
                           model_predictor=predict_svm)


def find_best_svm(kernels=None, plot=True, live_save=True):
    # Finding optimal kernel
    if not kernels:
        kernels = ['linear', 'poly', 'rbf']

    highest_acc = 0
    best_model = None
    accuracies = []
    for kernel in kernels:
        model, acc = train_svm(kernel=kernel)
        accuracies.append(acc * 100)
        if acc > highest_acc:
            highest_acc = acc
            best_model = model
            if live_save:
                MODEL_INFO_SVM.save_model(best_model)

    if not live_save and best_model:
        MODEL_INFO_SVM.save_model(best_model)

    if plot:
        sb.set_style('darkgrid')
        plt.title("SVM Classifier")
        plt.xlabel("Kernel")
        plt.ylabel("Accuracy (%)")
        sb.lineplot(x=kernels, y=accuracies)
        plt.show()

    return best_model, highest_acc


# .....................  Artificial Neural Network (ANN)  .........................

def train_ann():
    dataset = R.DigitDataset.get_singleton()

    model = keras.Sequential([
        keras.layers.Dense(128, activation=keras.activations.relu, input_shape=(dataset.img_pixels,)),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])  # directly using Accuracy() instance causes errors

    model.fit(dataset.x_train_flattened(), dataset.y_train, batch_size=50, epochs=10)
    MODEL_INFO_ANN.save_model(model)

    loss, acc = model.evaluate(dataset.x_test_flattened(), dataset.y_test)
    print(f"\n (DNN) Loss: {loss}, Accuracy: {acc}")
    return model, acc


def predict_ann(model, images_array: np.ndarray):
    samples = images_array.shape[0]

    images_array = images_array.reshape((samples, -1))      # Flatten samples
    pred = model.predict(images_array, verbose=0)

    out = np.zeros(samples, dtype=np.int32)
    for r in range(samples):
        out[r] = np.argmax(pred[r])
    return out


def test_saved_ann():
    # # Loading saved model
    model = MODEL_INFO_ANN.load_model()
    if not model:
        return

    dataset = R.DigitDataset.get_singleton()
    loss, acc = model.evaluate(dataset.x_test_flattened(), dataset.y_test)
    print(f"\n (DNN) Loss: {loss}, Accuracy: {acc}")


MODEL_INFO_ANN = ModelInfo("ANN",
                           "Artificial Neural Network",
                           "ann_digits.keras",
                           model_loader=R.load_keras_model,
                           model_saver=R.save_keras_model,
                           model_predictor=predict_ann)


# ................  Convolutional Neural Network (CNN)  .................

def train_cnn():
    dataset = R.DigitDataset.get_singleton()

    model = keras.Sequential([
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu",
                            input_shape=(*dataset.img_shape, 1)),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])  # directly using Accuracy() instance causes errors

    x_train = dataset.x_train(shape=(*dataset.org_x_train_shape, 1))
    model.fit(x_train, dataset.y_train, batch_size=50, epochs=10)

    # Saving
    MODEL_INFO_CNN.save_model(model)

    x_test = dataset.x_test(shape=(*dataset.org_x_test_shape, 1))
    loss, acc = model.evaluate(x_test, dataset.y_test)

    print(f"\n (CNN) Loss: {loss}, Accuracy: {acc}")
    return model, acc


# def predict_single_cnn(model, img: np.ndarray):
#     img = img.reshape((1, *img.shape, 1))
#     pred = model.predict(img)
#     return np.argmax(pred[0])


def predict_cnn(model, images_array: np.ndarray):
    images_array = images_array.reshape((*images_array.shape, 1))
    pred = model.predict(images_array, verbose=0)
    samples = pred.shape[0]

    out = np.zeros(samples, dtype=np.int32)
    for r in range(samples):
        out[r] = np.argmax(pred[r])
    return out


def test_saved_cnn():
    # # Loading saved model
    model = MODEL_INFO_CNN.load_model()
    if not model:
        return

    dataset = R.DigitDataset.get_singleton()
    x_test = dataset.x_test(shape=(*dataset.org_x_test_shape, 1))

    loss, acc = model.evaluate(x_test, dataset.y_test)
    print(f"\n (CNN) Loss: {loss}, Accuracy: {acc}")


MODEL_INFO_CNN = ModelInfo("CNN",
                           "Convolutional Neural Network",
                           "cnn_digits.keras",
                           model_loader=R.load_keras_model,
                           model_saver=R.save_keras_model,
                           model_predictor=predict_cnn)

# ................................................................................................

DEFAULT_MODEL_INFO = MODEL_INFO_CNN