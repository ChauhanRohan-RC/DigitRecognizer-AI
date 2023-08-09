# import matplotlib.pyplot as plt
# import seaborn as sb
# from sklearn import neighbors, metrics
# import R
# import numpy as np
#
# from models import MODEL_INFO_KNN
#
#
# # ...................  K-Nearest Neighbours (KNN)  ...........................
# # experimentally, best k=3
#
# def train_knn(k):
#     dataset = R.DigitDataset.get_singleton()
#
#     model = neighbors.KNeighborsClassifier(n_neighbors=k)
#     model.fit(dataset.x_train_flattened(), dataset.y_train)
#
#     y_pred = model.predict(dataset.x_test_flattened())
#     acc = metrics.accuracy_score(dataset.y_test, y_pred)
#
#     print(f"(KNN) k: {k}, Accuracy: {acc * 100: .2f}")
#     return model, acc
#
#
# def predict_knn(model, images_array: np.ndarray):
#     images_array = images_array.reshape((images_array.shape[0], -1))       # flatten images
#     pred = model.predict(images_array)
#     return pred
#
#
# def find_best_knn(k_range=None, plot=True, live_save=True):
#     # Finding optimal value of hyperparameter k
#     if not k_range:
#         k_range = list(range(1, 7))
#
#     highest_acc = 0
#     best_model = None
#     accuracies = []
#     for k in k_range:
#         model, acc = train_knn(k)
#         accuracies.append(acc * 100)
#         if acc > highest_acc:
#             highest_acc = acc
#             best_model = model
#             if live_save:
#                 MODEL_INFO_KNN.save_model(best_model)
#
#     if not live_save and best_model:
#         MODEL_INFO_KNN.save_model(best_model)
#
#     if plot:
#         sb.set_style('darkgrid')
#         plt.title("KNN Classifier")
#         plt.xlabel("No of Nearest Neighbours (k)")
#         plt.ylabel("Accuracy (%)")
#         sb.lineplot(x=k_range, y=accuracies)
#         plt.show()
#
#     return best_model, highest_acc
#
