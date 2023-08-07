# DIgit Recognizer AI
(RC @July 23, 2022)

####  A hand written digit recognizer using Deep Neural Network (DNN), K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms

![DNN](graphics/dnn_recognize2.png?raw=true)

### Overview

Contains three AI models listed below
* _**Deep Neural Network (DNN)**_: 
  * Input Layer (784 neurons)
  * Dense 1 (128 neurons)
  * Dense 2 (32 neurons)
  * Output Layer (10 neurons)
  * Tested Accuracy: 98.4 %
* **_K-Nearest Neighbour (KNN) Classifier_** 
  * k: 3-7
  * Tested Accuracy: 96.7 %
* **_Support vector Machine (SVM) Classifier_**
  * kernel: rbf, poly
  * C = 2.0
  * Tested Accuracy: 98.1 %

All the models are trained on [MNIST digits dataset](https://www.tensorflow.org/datasets/catalog/mnist) containing 60,000 samples of 28x28 images.

### Usage
* Clone repository `git clone https://github.com/ChauhanRohan-RC/DigitRecognizer-AI.git`
* `python main.py`

![KNN](graphics/knn_recognize7.png?raw=true)

### Controls
* Mouse L-Click Drag:  Draw
* Mouse R-CLick Drag :  Erase
* Enter/Space :  Recognize drawn digit
* Escape :  Clear canvas / Quit
* S : Toggle Sound

![SVM](graphics/svm_recognize9.png?raw=true)
