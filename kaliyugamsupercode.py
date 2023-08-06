import cv2
import numpy as np
import csv
import os

# Load YOLOv3-tiny configuration and weights
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Read presence_absence.csv file
presence_absence_file = "presence_absence.csv"
class_presence = {}
with open(presence_absence_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        for class_name, presence in zip(header, row):
            class_presence[class_name] = bool(int(presence))

# Create a folder to store cropped images
output_folder = "cropped_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to perform object detection
def detect_objects(frame):
    height, width, _ = frame.shape

    # Prepare the frame as a blob to feed into the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pass the blob through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process the network output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to get rid of overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels, and crop images
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check if the class is in presence_absence.csv and it is absent
            if label in class_presence and not class_presence[label]:
                # Crop the region of interest and save it as an image
                roi = frame[y:y + h, x:x + w]
                image_path = os.path.join(output_folder, f"{label}_crop_{i}.jpg")
                cv2.imwrite(image_path, roi)

    return frame

# Main function to process live video stream
def main():
    # Replace '0' with the path to your video file if you want to process a video file instead
    cap = cv2.VideoCapture(0)

    absent_counts = {class_name: 0 for class_name in class_presence.keys()}
    max_absent_frames = 20  # Define the threshold for the number of frames with a class absent

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)

        # Check if any class is absent for more than the threshold number of frames
        for class_name in class_presence.keys():
            if class_name not in absent_counts:
                continue
            if class_presence[class_name]:
                absent_counts[class_name] = 0
            else:
                if absent_counts[class_name] >= max_absent_frames:
                    print(f"Class {class_name} is absent for more than {max_absent_frames} frames.")
                else:
                    absent_counts[class_name] += 1

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

