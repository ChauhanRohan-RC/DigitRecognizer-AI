import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import svm, metrics
import R


def create_and_train_SVM(C=1.0, kernel='rbf'):
    model = svm.SVC(C=C, kernel=kernel)

    dataset = R.DigitDataset.get_singleton()
    model.fit(dataset.x_train, dataset.y_train)

    y_pred = model.predict(dataset.x_test)
    acc = metrics.accuracy_score(dataset.y_test, y_pred)
    print(f"(SVM) kernel: {kernel}, Accuracy: {acc * 100: .2f}")
    return model, acc


def find_best_SVM(kernels=None, plot=True, live_save=True):
    # Finding optimal kernel
    if not kernels:
        kernels = ['linear', 'poly', 'rbf']

    highest_acc = 0
    best_model = None
    accuracies = []
    for kernel in kernels:
        model, acc = create_and_train_SVM(kernel=kernel)
        accuracies.append(acc * 100)
        if acc > highest_acc:
            highest_acc = acc
            best_model = model
            if live_save:
                R.save_sklearn_model(model, R.FILE_NAME_MODEL_SKLEARN_SVM)

    if not live_save and best_model:
        R.save_sklearn_model(best_model, R.FILE_NAME_MODEL_SKLEARN_SVM)

    if plot:
        sb.set_style('darkgrid')
        plt.title("SVM Classifier")
        plt.xlabel("Kernel")
        plt.ylabel("Accuracy (%)")
        sb.lineplot(x=kernels, y=accuracies)
        plt.show()

    return best_model, highest_acc

# best kernel is rbf (experimentally)
