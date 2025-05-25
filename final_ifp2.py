
from PIL import Image
import psutil
import cv2
import easyocr
from ultralytics import YOLO
import torch
import numpy as np
import re, datetime
import matplotlib.pyplot as plt


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

        img = Image.fromarray(cropped)
        img.save("detection1.jpg")

        #cv2.imshow("image", orig_img)
        #cv2.imshow("label", cropped)
        #cv2.waitKey(5000)

        return cropped
    else:
        return np.array([0])

def show_dates(list):
    import re
    manufacturing_dates = []
    expiry_dates = []
    day = []
    month = []
    year = []
    delimiters = r"[./-]"

    for date in list:
        num = re.split(delimiters, date)
        day.append(num[0])
        month.append(num[1])
        dt=2
        if len(date)==3:
            year.append(num[2])
            dt=3

    if len(list) >= 2:
        if dt==3:
            date1 = (int(year[0]), int(month[0]), int(day[0]))
            date2 = (int(year[1]), int(month[1]), int(day[1]))
        else:
            date1 = (int(month[0]), int(day[0]))
            date2 = (int(month[1]), int(day[1]))


        if date1 > date2:
            manufacturing_dates.append(list[1])
            expiry_dates.append(list[0])
        else:
            manufacturing_dates.append(list[0])
            expiry_dates.append(list[1])

        # Output the results for verification
        manufacturing_dates.append(expiry_dates)
        return manufacturing_dates
        #print("Manufacturing Date:", manufacturing_dates)
        #print("Expiry Date:", expiry_dates)


    else:
        return  list
        #print("manufacturing date:", list)



# -------------------------------------------------------------------------------------------------------------------
                                           #video capture
"""
cap = cv2.VideoCapture(0)
n=1
while n<=1:

    name = str(n) + ".jpg"
    ret,frame = cap.read()
    cv2.imshow("frame",frame)
    cv2.imwrite(name,frame)
    key = cv2.waitKey(2000)
    n=n+1
    if key == ord("p"):
        break
cap.release()
cv2.destroyAllWindows() """

# -------------------------------------------------------------------------------------------------------------------
#mdl = "full_run_best.pt"
#final = ["collin.jpg","perfume2.jpg"]

def detect_multiple(final,mdl):
    c_img = []
    for i in final:

        temp = detect(mdl,i)
        if temp.any() != np.array([0]):
            c_img.append(temp)
    return c_img

def date_finder(c_img):
    reader = easyocr.Reader(['en'], detector='dbnet18',quantize=True,gpu=False)

    print("\n")

    if c_img != []:

        for j in c_img:
            result = reader.readtext(j)
            days = []
            for detection in result:

                print(detection[1])
                print("---------------------------------------------------------------------------------")
                detected = detection[1].split(",")                                                                #change1
                date_pattern = r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{2,4})\b'


                for dtd in detected:
                    day = re.search(date_pattern, dtd)
                    if day != None:
                       days.append(day)
                                                 #change2

            all_dates=[]
            for i in days:
                if i != None:
                    all_dates.append(i.group())
            #return show_dates(all_dates)

            print("---------------------------------------------------------------")
            print(all_dates)

            return all_dates

    else:
        print("no predictions")
        print("-----------------------------------------------------------")

'''def display_bbox(dates,result,image):
    img = cv2.imread(image)
    spacer = 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    for detection in result:
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
        if text in dates:
            img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
            img = cv2.putText(img, text, (20, spacer), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        spacer += 15

    plt.imshow(img)
    plt.show()'''