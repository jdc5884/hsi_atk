# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

from sklearn.metrics import accuracy_score
from HyperspecClassifier import HyperImageData

hyper_data = HyperImageData('../Data/headers3mgperml.csv', 15, 3)

hyper_data.train_test()

hyper_data.SVM_make_train()

print(hyper_data.SVM_pred())
print(accuracy_score((hyper_data.y_test), (hyper_data.SVM_pred())))