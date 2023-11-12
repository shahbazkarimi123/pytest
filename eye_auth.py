import cv2
import numpy as np
import os

# Load the iris database
#database = np.loadtxt("iris_database.csv", delimiter=",")

# Capture the iris image
eye_image = cv2.imread("eye_image.jpg")

# Localize the iris region
iris_region = cv2.GaussianBlur(eye_image, (5, 5), 0)
iris_region = cv2.Canny(iris_region, 25, 125)

# Extract the iris features
iris_features = cv2.HOGDescriptor(iris_region, (9, 9), (8, 8), (1, 1))
iris_features = iris_features.compute()

# Classify the iris
iris_class = cv2.ml.SVM.load("iris_classifier.xml")
iris_class.predict(iris_features)

# Print the iris authentication result
if iris_class.predict(iris_features) == 1:
    print("Iris authentication successful")
else:
    print("Iris authentication failed")