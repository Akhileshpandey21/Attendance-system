import cv2
import numpy as np
import os

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define the path to the images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
idList = []

# Loop through the images and assign IDs
for path in pathList:
    img = cv2.imread(os.path.join(folderPath, path))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgList.append(gray_img)
    idList.append(int(os.path.splitext(path)[0]))  # Convert filename to ID (assumes filename is numeric)

# Train the recognizer
recognizer.train(imgList, np.array(idList))

# Save the trained model
recognizer.write('trained_model.yml')

print("Model trained and saved as trained_model.yml")
