import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path = "dataset"

X = []
y = []

labels = {"normal":0,"risk":1}

for folder in os.listdir(dataset_path):

    for img in os.listdir(dataset_path+"/"+folder):

        path = dataset_path+"/"+folder+"/"+img
        image = cv2.imread(path)
        image = cv2.resize(image,(64,64))
        X.append(image)
        y.append(labels[folder])

X = np.array(X)/255.0
y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# -------------------------
# CNN MODEL
# -------------------------

cnn = Sequential()

cnn.add(Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)))
cnn.add(MaxPooling2D())

cnn.add(Conv2D(64,(3,3),activation="relu"))
cnn.add(MaxPooling2D())

cnn.add(Flatten())
cnn.add(Dense(128,activation="relu"))
cnn.add(Dense(1,activation="sigmoid"))

cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

cnn.fit(X_train,y_train,epochs=5)

cnn.save("models/cnn_model.h5")

# -------------------------
# ML MODELS
# -------------------------

X_train_flat = X_train.reshape(len(X_train),-1)
X_test_flat = X_test.reshape(len(X_test),-1)

svm = SVC()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()

svm.fit(X_train_flat,y_train)
rf.fit(X_train_flat,y_train)
knn.fit(X_train_flat,y_train)

joblib.dump(svm,"models/svm_model.pkl")
joblib.dump(rf,"models/rf_model.pkl")
joblib.dump(knn,"models/knn_model.pkl")

# -------------------------
# Accuracy Comparison
# -------------------------

acc = {}

acc["SVM"] = accuracy_score(y_test,svm.predict(X_test_flat))
acc["Random Forest"] = accuracy_score(y_test,rf.predict(X_test_flat))
acc["KNN"] = accuracy_score(y_test,knn.predict(X_test_flat))

plt.bar(acc.keys(),acc.values())
plt.title("Model Accuracy Comparison")
plt.savefig("static/results/accuracy_graph.png")

# -------------------------
# Confusion Matrix
# -------------------------

pred = rf.predict(X_test_flat)

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,fmt="d")

plt.savefig("static/results/confusion_matrix.png")

print("Training Completed")