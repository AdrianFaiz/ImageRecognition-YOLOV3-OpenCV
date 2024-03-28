import cv2
import numpy as np

# Load YOLO
net_yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes_yolo = []
with open("coco.names", "r") as f:
    classes_yolo = [line.strip() for line in f.readlines()]
layer_names_yolo = net_yolo.getLayerNames()
output_layers_yolo = [layer_names_yolo[i[0] - 1] for i in net_yolo.getUnconnectedOutLayers()]

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read and display each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_yolo.setInput(blob)
    outs = net_yolo.forward(output_layers_yolo)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                label = classes_yolo[class_id]
                if label == 'cell phone' or label == 'watch':
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Draw bounding boxes and labels
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'GWAPO', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes_yolo[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
