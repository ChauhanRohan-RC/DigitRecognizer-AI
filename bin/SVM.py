# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sb
# from sklearn import svm, metrics
# import R
# from models import ModelInfo
#
#
# # ....................  Support Vector Machine (SVM) ...............................
# # experimentally, best kernel = rbf
#
# def train_svm(C=1.0, kernel='rbf'):
#     model = svm.SVC(C=C, kernel=kernel)
#
#     dataset = R.DigitDataset.get_singleton()
#     model.fit(dataset.x_train_flattened(), dataset.y_train)
#
#     y_pred = model.predict(dataset.x_test_flattened())
#     acc = metrics.accuracy_score(dataset.y_test, y_pred)
#     print(f"(SVM) kernel: {kernel}, Accuracy: {acc * 100: .2f}")
#     return model, acc
#
#
# def predict_svm(model, images_array: np.ndarray):
#     images_array = images_array.reshape((images_array.shape[0], -1))       # flatten images
#     pred = model.predict(images_array)
#     return pred
#
#
# MODEL_INFO_SVM = ModelInfo("SVM",
#                            "Support Vector Machine",
#                            "svm_digits.gzip",
#                            model_loader=R.load_sklearn_model,
#                            model_saver=R.save_sklearn_model,
#                            model_predictor=predict_svm)
#
#
# def find_best_svm(kernels=None, plot=True, live_save=True):
#     # Finding optimal kernel
#     if not kernels:
#         kernels = ['linear', 'poly', 'rbf']
#
#     highest_acc = 0
#     best_model = None
#     accuracies = []
#     for kernel in kernels:
#         model, acc = train_svm(kernel=kernel)
#         accuracies.append(acc * 100)
#         if acc > highest_acc:
#             highest_acc = acc
#             best_model = model
#             if live_save:
#                 R.save_sklearn_model(model, MODEL_INFO_SVM.file_name)
#
#     if not live_save and best_model:
#         R.save_sklearn_model(best_model, MODEL_INFO_SVM.file_name)
#
#     if plot:
#         sb.set_style('darkgrid')
#         plt.title("SVM Classifier")
#         plt.xlabel("Kernel")
#         plt.ylabel("Accuracy (%)")
#         sb.lineplot(x=kernels, y=accuracies)
#         plt.show()
#
#     return best_model, highest_acc
