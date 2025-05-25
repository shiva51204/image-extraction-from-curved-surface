import cv2
from ultralytics import YOLO


#THIS IS A PROGRAM TO SHOW THE DETECTED DATE LABEL IN VIDEO


# Load the YOLOv8 model
model = YOLO('new_8n_best.pt')  # replace with your model path if different

# Open the video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Live', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
