import cv2
import numpy as np

def is_fingerprint(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Calculate edge density
    edge_density = np.sum(edges > 0) / edges.size

    # Texture variation (standard deviation)
    texture_var = np.std(gray)

    # Simple threshold logic (tuned empirically)
    if 0.02 < edge_density < 0.25 and 30 < texture_var < 80:
        return True
    return False


cap = cv2.VideoCapture(0)
print("Press 'c' to capture image and detect fingerprint.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam - Press 'c' to capture", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        if is_fingerprint(frame):
            print(" Fingerprint Detected!")
            cv2.putText(frame, "Fingerprint Detected !! ", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            print(" Not a Fingerprint")
            cv2.putText(frame, "Not a Fingerprint !!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(1500)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
