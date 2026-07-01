import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# Load models
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/rf_model.pkl")
knn = joblib.load("models/knn_model.pkl")
cnn = load_model("models/cnn_model.h5")


def predict_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img,(64,64))

    img = img/255.0


    # For CNN
    img_cnn = np.reshape(img,(1,64,64,3))


    # For ML models
    img_flat = img.reshape(1,-1)


    # Use Random Forest as main predictor
    pred = rf.predict(img_flat)[0]


    if pred == 1:
        return "Heart Disease Risk Detected"
    else:
        return "Normal Retina"