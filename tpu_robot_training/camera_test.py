import cv2

CAMERA = "/dev/video0"

cap = cv2.VideoCapture(CAMERA)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Camera opened successfully.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    cv2.imshow("Camera", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")