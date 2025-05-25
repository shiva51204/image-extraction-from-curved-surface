from PIL import Image
import psutil
import cv2
import easyocr
from ultralytics import YOLO
import torch
import numpy as np


def detect(dl, image):
    model = YOLO(dl)

    input = image

    orig_img = cv2.imread(input)

    results = model.predict(input)

    result = results[0]

    if len(result.boxes) != 0:

        box = result.boxes[0]
        cord = []
        a = []
        cord = torch.tensor_split(box.xyxy, 4, 1)

        for j in cord:
            a.append(j.item())

        p = int(a[0])
        q = int(a[1])
        r = int(a[2])
        s = int(a[3])

        arr = result.orig_img
        cropped = arr[q:s, p:r]

        #img = Image.fromarray(cropped.astype('uint8'))
        #img.save("crpd_img.jpg")

        #cv2.imshow("image", orig_img)
        #cv2.imshow("label", cropped)
        #cv2.waitKey(5000)

        return cropped
    else:
        return np.array([0])

mdl="full_run_best.pt"
img1="dettol.jpg"
img2="perfume1.jpg"

crpd = detect(mdl,img1)
n_crpd = detect(mdl,img2)

print(crpd)
print("-------------------------------------------------------------------")
print(n_crpd)