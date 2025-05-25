
from PIL import Image
import psutil
import cv2
import easyocr
from ultralytics import YOLO
import torch
import numpy as np
import re, datetime



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
        if len(date)==3:
            year.append(num[2])
            dt=3
        else:
            year.append("")
            dt=2




    if len(list) == 2:
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
        print("Manufacturing Dates:", manufacturing_dates)
        print("Expiry Dates:", expiry_dates)

    else:
        if(len(list)==1):
            print("manufacturing date:", list)
        else:
            while len(list)>=2:
                list.pop()
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

            print("Manufacturing Dates:", manufacturing_dates)
            print("Expiry Dates:", expiry_dates)



#---------------------------------------------------------------------------------------------------------------------
mdl = "new_8n_best.pt"

final = ['scent_bottle.jpg']

c_img = []
for i in final:

    temp = detect(mdl,i)
    if temp.any() != np.array([0]):
        c_img.append(temp)



reader = easyocr.Reader(['en'], detector='dbnet18',gpu=False,quantize=True)

print("\n")
all_dates=[]
if c_img != []:
    for j in c_img:
        result = reader.readtext(j)
        days = []
        for detection in result:

            #print(detection[1])
            #print("---------------------------------------------------------------------------------")
            detected = detection[1].split(",")
            date_pattern = r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{2,4})\b'


            for dtd in detected:
                day = re.search(date_pattern, dtd)
                if day != None:
                   days.append(day)

        for i in days:
            if i != None:
                all_dates.append(i.group())
        #print(detection[1])
        #print(all_dates)
        show_dates(all_dates)

        print("---------------------------------------------------------------")
        #print(all_dates)
else:
    print("no predictions")
    print("-----------------------------------------------------------")