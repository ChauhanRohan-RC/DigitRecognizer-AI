import os
import sys
import time

import numpy as np
import pygame
import sklearn
from pygame import Color
from tensorflow import keras

import C

# ...........................  File Paths  ......................................
FROZEN = getattr(sys, 'frozen', False)
DIR_MAIN = os.path.dirname(sys.executable) if FROZEN else (
    os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

DIR_MODELS = os.path.join(DIR_MAIN, "models")
DIR_RES = os.path.join(DIR_MAIN, "res")
DIR_RES_IMAGES = os.path.join(DIR_RES, "images")
DIR_RES_SOUND = os.path.join(DIR_RES, "sound")
DIR_RES_FONT = os.path.join(DIR_RES, "font")

FILE_NAME_MODEL_SKLEARN_KNN = "knn_digits.gzip"
FILE_NAME_MODEL_SKLEARN_SVM = "svm_digits.gzip"
FILE_NAME_MODEL_KERAS_DNN = "dnn_digits.keras"


def get_model_file_path(file_name: str):
    return os.path.join(DIR_MODELS, file_name)


# ..........................  Models  ..............................

def save_sklearn_model(model, file_name: str, compression_level: int = 2):
    """
    :param model: model to save
    :param file_name: name of the model
    :param compression_level: level of compression, in range of 0-9
    """
    print(f"Saving sklearn model {file_name} ...")
    sklearn.utils._joblib.dump(model, get_model_file_path(file_name), compress=compression_level)


def load_sklearn_model(file_name: str):
    """
    :return loaded model
    """
    return sklearn.utils._joblib.load(get_model_file_path(file_name))


def save_keras_model(model, file_name):
    print(f"Saving keras model {file_name} ...")
    model.save(get_model_file_path(file_name))


def load_keras_model(file_name):
    return keras.models.load_model(get_model_file_path(file_name))


class ModelInfo:

    def __init__(self, id: int, short_label: str, long_label: str):
        self.id = id
        self.short_label = short_label
        self.long_label = long_label
        self.display_name = short_label

    def __eq__(self, other):
        return type(other) == type(self) and other.id == self.id

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"Model(id={self.id}, short_label={self.short_label}, long_label={self.long_label})"

    def __str__(self):
        return self.__repr__()


MODEL_DNN = ModelInfo(0, "DNN", "Deep Neural Network")
MODEL_KNN = ModelInfo(1, "KNN", "K-Nearest Neighbours")
MODEL_SVM = ModelInfo(2, "SVM", "Support Vector Machine")

MODELS = [
    MODEL_DNN,
    MODEL_KNN,
    MODEL_SVM
]

MODEL_DEFAULT = MODEL_DNN


class Models:
    DEFAULT_PRELOAD = True
    DEFAULT_PRELOAD_IN_BG = True

    _sInstance = None

    @classmethod
    def get_singleton(cls):
        if not cls._sInstance:
            cls._sInstance = cls()

        return cls._sInstance

    def __init__(self, preload: bool = DEFAULT_PRELOAD):
        self._knn = None
        self._svm = None
        self._dnn = None

        if preload:
            self.preload()

    def _get_knn(self):
        if not self._knn:
            self._knn = load_sklearn_model(FILE_NAME_MODEL_SKLEARN_KNN)
        return self._knn

    @property
    def knn(self):
        return self._get_knn()

    def _get_svm(self):
        if not self._svm:
            self._svm = load_sklearn_model(FILE_NAME_MODEL_SKLEARN_SVM)
        return self._svm

    @property
    def svm(self):
        return self._get_svm()

    def _get_dnn(self):
        if not self._dnn:
            self._dnn = load_keras_model(FILE_NAME_MODEL_KERAS_DNN)
        return self._dnn

    @property
    def dnn(self):
        return self._get_dnn()

    def preload(self, in_bg: bool = DEFAULT_PRELOAD_IN_BG):
        if not self._knn:
            knn_ = C.execute_on_thread(self._get_knn) if in_bg else self._get_knn()

        if not self._svm:
            svm_ = C.execute_on_thread(self._get_svm) if in_bg else self._get_svm()

        if not self._dnn:
            dnn_ = C.execute_on_thread(self._get_dnn) if in_bg else self._get_dnn()

    def predict(self, input_img: np.ndarray, model: ModelInfo = MODEL_DEFAULT) -> int:
        input = np.array([input_img])

        if model == MODEL_KNN:
            _pred = self.knn.predict(input)
            return _pred[0]

        if model == MODEL_SVM:
            _pred = self.svm.predict(input)
            return _pred[0]

        if model == MODEL_DNN:
            _pred = self.dnn.predict(input, verbose=0)
            return np.argmax(_pred[0])

        raise ValueError("Unknown model type: " + repr(model))


# .................................  Dataset  ............................

class DigitDataset:
    IMG_SIZE = (28, 28)
    IMG_PIXELS = np.product(IMG_SIZE)

    _sInstance = None

    @classmethod
    def get_singleton(cls):
        if not cls._sInstance:
            cls._sInstance = cls()

        return cls._sInstance

    def __init__(self):
        start = time.time_ns()
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.img_pixels = np.product(x_train.shape[1:])

        # Parsing (Flattening and normalizing each image)
        self.__x_train = x_train.reshape(-1, self.img_pixels) / 255.0
        self.__x_test = x_test.reshape(-1, self.img_pixels) / 255.0
        self.__y_train = y_train
        self.__y_test = y_test

        end = time.time_ns()
        print(f"MNIST Digits dataset loaded in {(end - start) / 1.0E6} ms")

    @property
    def x_train(self):
        """"
        :return: Training input data. 2D numpy array, shape = (n_train_samples, img_pixels)
        """
        return self.__x_train

    @property
    def y_train(self):
        """"
        :return: Training output data. 1D numpy array, shape = (n_train_samples, )
        """
        return self.__y_train

    @property
    def x_test(self):
        """"
        :return: Testing input data. 2D numpy array, shape = (n_test_samples, img_pixels)
        """
        return self.__x_test

    @property
    def y_test(self):
        """"
        :return: Testing output data. 1D numpy array, shape = (n_test_samples, )
        """
        return self.__y_test


# .............................  UI ...........................

def gray(i) -> Color:
    return Color(i, i, i)


WIN_WIDTH, WIN_HEIGHT = 560, 620
GRID_PAD = 0
GRID_WIDTH = min(WIN_WIDTH, WIN_HEIGHT) - (GRID_PAD * 2)
GRID_HEIGHT = GRID_WIDTH
GRID_OUTLINE_WIDTH = 2
GRID_OUTLINE_RADIUS = 2

WIN_PADX = 10
WIN_PADY = 10

GRID_MARK_NEIGHBOUR_SPAN = 1
GRID_UNMARK_NEIGHBOUR_SPAN = 2

FPS = 120

GRID_ROWS = 56
GRID_COLS = 56

# Colors
WHITE = gray(255)
BLACK = gray(0)

BG_DARK = BLACK
BG_MEDIUM = gray(20)
BG_LIGHT = gray(40)

FG_DARK = WHITE
FG_MEDIUM = gray(225)
FG_LIGHT = gray(190)

# COLOR_ACCENT_DARK = Color(57, 206, 255)
# COLOR_ACCENT_MEDIUM = Color(33, 187, 255)
# COLOR_ACCENT_LIGHT = Color(2, 169, 255)

COLOR_ACCENT_DARK = Color(29, 255, 0)
COLOR_ACCENT_MEDIUM = Color(29, 226, 0)
COLOR_ACCENT_LIGHT = Color(29, 190, 0)

COLOR_HIGHLIGHT = Color(253, 255, 52)

COLOR_TRANSPARENT = Color(0, 0, 0, 0)
COLOR_TRANSLUCENT = Color(0, 0, 0, 125)

TINT_SELF_DARK = Color(120, 255, 120)
TINT_SELF_MEDIUM = Color(55, 255, 55)
TINT_SELF_LIGHT = Color(0, 255, 0)

TINT_ENEMY_DARK = Color(255, 120, 120)
TINT_ENEMY_MEDIUM = Color(255, 55, 55)
TINT_ENEMY_LIGHT = Color(255, 0, 0)


ID_CLEAR_BUTTON = 0xFF0AC
CLEAR_BUTTON_TEXT = "Clear"
EXIT_BUTTON_TEXT = "Quit"

ID_PREDICT_BUTTON = 0xFAFB
PREDICT_BUTTON_TEXT = "Predict"


# Fonts
FILE_PATH_FONT_PD_SANS = os.path.join(DIR_RES_FONT, 'product_sans_regular.ttf')
FILE_PATH_FONT_PD_SANS_MEDIUM = os.path.join(DIR_RES_FONT, 'product_sans_medium.ttf')
FILE_PATH_FONT_PD_SANS_LIGHT = os.path.join(DIR_RES_FONT, 'product_sans_light.ttf')
FILE_PATH_FONT_AQUIRE = os.path.join(DIR_RES_FONT, 'aquire.otf')
FILE_PATH_FONT_AQUIRE_LIGHT = os.path.join(DIR_RES_FONT, 'aquire_light.otf')
FILE_PATH_FONT_AQUIRE_BOLD = os.path.join(DIR_RES_FONT, 'aquire_bold.otf')

pygame.font.init()
# FONT_TITLE = pygame.font.Font(FILE_PATH_FONT_AQUIRE, 30)
# FONT_SUMMARY = pygame.font.Font(FILE_PATH_FONT_PD_SANS_LIGHT, 19)
FONT_STATUS = pygame.font.Font(FILE_PATH_FONT_PD_SANS_LIGHT, 22)
FONT_BUTTONS = pygame.font.Font(FILE_PATH_FONT_AQUIRE, 18)
FONT_BUTTONS_MEDIUM = pygame.font.Font(FILE_PATH_FONT_AQUIRE, 14)
FONT_BUTTONS_SMALL = pygame.font.Font(FILE_PATH_FONT_PD_SANS, 13)


# Sound

DEFAULT_SOUND_ENABLED = True

FILE_PATH_SOUND_BUTTON_HOVER = os.path.join(DIR_RES_SOUND, "button_hover.wav")
FILE_PATH_SOUND_BUTTON_CLICK = os.path.join(DIR_RES_SOUND, "button_click.wav")
FILE_PATH_SOUND_PREDICT = os.path.join(DIR_RES_SOUND, "predict.wav")

pygame.mixer.init()
SOUND_BUTTON_HOVER = pygame.mixer.Sound(FILE_PATH_SOUND_BUTTON_HOVER)
SOUND_BUTTON_CLICK = pygame.mixer.Sound(FILE_PATH_SOUND_BUTTON_CLICK)
SOUND_PREDICT = pygame.mixer.Sound(FILE_PATH_SOUND_PREDICT)


def play_button_sound(hover: bool):
    (SOUND_BUTTON_HOVER if hover else SOUND_BUTTON_CLICK).play()


def play_predict_sound():
    SOUND_PREDICT.play()
