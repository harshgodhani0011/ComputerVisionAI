import cv2
from ultralytics import YOLO

# Load custom trained model
model = YOLO("../models/best.pt")

# Open video
video = cv2.VideoCapture("videos/helmet_test.mp4")

while True:
    success, frame = video.read()

    if not success:
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Helmet Detection AI", annotated_frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()