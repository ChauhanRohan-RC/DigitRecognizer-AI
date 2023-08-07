import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import neighbors, metrics
import R


def create_and_train_kNN(k):
    model = neighbors.KNeighborsClassifier(n_neighbors=k)
    dataset = R.DigitDataset.get_singleton()
    model.fit(dataset.x_train, dataset.y_train)

    y_pred = model.predict(dataset.x_test)
    acc = metrics.accuracy_score(dataset.y_test, y_pred)
    print(f"(KNN) k: {k}, Accuracy: {acc * 100: .2f}")
    return model, acc


def find_best_KNN(k_range=None, plot=True, live_save=True):
    # Finding optimal value of hyperparameter k
    if not k_range:
        k_range = list(range(1, 7))

    highest_acc = 0
    best_model = None
    accuracies = []
    for k in k_range:
        model, acc = create_and_train_kNN(k)
        accuracies.append(acc * 100)
        if acc > highest_acc:
            highest_acc = acc
            best_model = model
            if live_save:
                R.save_sklearn_model(model, R.FILE_NAME_MODEL_SKLEARN_KNN)

    if not live_save and best_model:
        R.save_sklearn_model(best_model, R.FILE_NAME_MODEL_SKLEARN_KNN)

    if plot:
        sb.set_style('darkgrid')
        plt.title("KNN Classifier")
        plt.xlabel("No of Nearest Neighbours (k)")
        plt.ylabel("Accuracy (%)")
        sb.lineplot(x=k_range, y=accuracies)
        plt.show()

    return best_model, highest_acc

# BEST K = 3 (experimentally determined)
# model, acc = create_and_train_kNN(3)
# sklearn.utils._joblib.dump(model, 'knn_digits3.gzip', compress=2)
