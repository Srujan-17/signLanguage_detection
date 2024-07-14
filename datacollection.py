import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\srujan.000\Sign-Language-detection-main\data\2"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        if imgCrop is None or imgCrop.size == 0:
            print("Image cropping failed or returned an empty result.")
            time.sleep(1)  # add delay if you want
            continue

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord("q"):  # you can exit anytime by pressing q
        break
cap.release()
cv2.destroyAllWindows()

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import pyautogui


# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# offset = 20
# imgSize = 300
# counter = 0

# labels = ["Hello", "1", "2", "I love you"]

# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

#         if imgCrop is None or imgCrop.size == 0:
#             print("Image cropping failed or returned an empty result.")
#             continue

#         imgCropShape = imgCrop.shape

#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap: wCal + wGap] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             gesture_label = labels[index]

#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap: hCal + hGap, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             gesture_label = labels[index]

#         # Volume control based on gestures
#         if gesture_label == "1":
#             pyautogui.press("volumeup")
#         elif gesture_label == "I love you":
#             pyautogui.press("volumedown")

#         cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
#         cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

#         cv2.imshow('ImageCrop', imgCrop)
#         cv2.imshow('ImageWhite', imgWhite)

#         cv2.imshow('Hand Gesture', imgOutput)

#     key = cv2.waitKey(1) & 0xFF  # Use bitwise AND for compatibility
#     if key == ord('q'):  # Press 'q' to exit
#         print("Exiting...")
#         break

# cap.release()
# cv2.destroyAllWindows()
