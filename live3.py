

#THIS IS THE FULL PROGRAM WITH LIVE DETECTON



import cv2
from ultralytics import YOLO
from final_ifp2 import date_finder

# Load the YOLOv8 model
model = YOLO('models/best_yolov8n_new.pt')

# Open the video stream
cap = cv2.VideoCapture(0)
frames=15 #no of detected images
frame_number = 0
t=0
c_img_list=[]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Check if any objects are detected
    if len(results[0].boxes.xyxy) > 0 :
        # Extract and save/display cropped images
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cropped_image = frame[y1:y2, x1:x2]
            #cv2.imshow(f'Cropped Image {i}', cropped_image)
            #cv2.imwrite(f'cropped_frame_{frame_number}_object_{i}.jpg', cropped_image)
            c_img_list.append(cropped_image)
        t=t+1
        frame_number += 1

        # Render results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('YOLOv8 Live', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or t==frames:
            break

        # Close cropped images windows
        #for i in range(len(results[0].boxes.xyxy)):
            #cv2.destroyWindow(f'Cropped Image {i}')
    else:
        # Display the frame without annotations if no objects are detected
        cv2.imshow('YOLOv8 Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(len(c_img_list))
final=date_finder(c_img_list)

print(final)

