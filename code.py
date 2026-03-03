import cv2
import mediapipe as mp
#this is for test git commit second
print(cv2.__version__)
print("hello from code.py....")

# Use tasks API instead of mp.solutions
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download this model file once and keep in same folder:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task
model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    finger_count = 0

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Thumb
        if landmarks[4].x < landmarks[3].x:
            finger_count += 1

        # Fingers
        tips = [8, 12, 16, 20]
        for tip in tips:
            if landmarks[tip].y < landmarks[tip - 2].y:
                finger_count += 1

        # Draw simple circles
        for lm in landmarks:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.putText(frame, f'Fingers: {finger_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()