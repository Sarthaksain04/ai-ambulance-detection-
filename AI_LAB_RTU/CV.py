import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
        print("Error: Unable to access the camera feed.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from the camera feed.")
        break

    cv2.imshow("Smart CCTV Analysis", frame)

        # Press 'q' to exit the feed
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()