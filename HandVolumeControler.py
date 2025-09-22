import cv2
import mediapipe
import numpy as np
import time
import math
import osascript

cap = cv2.VideoCapture(0)
pTime = 0  # For Print the fps
precentage = 0

mpHands = mediapipe.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mediapipe.solutions.drawing_utils

while True:
    _, img = cap.read()

    # Detect the hands
    imgRGB = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dtype=np.uint8)
    results = hands.process(imgRGB)
    numHands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if results.multi_hand_landmarks:  # If a hand is detected:
        # Drawy the hands
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        # Mark points of thumb and pointer
        h, w, c = imgRGB.shape
        landMark = results.multi_hand_landmarks[0].landmark
        thumbX, thumbY = int(landMark[4].x * w), int(landMark[4].y * h)
        pointX, pointY = int(landMark[8].x * w), int(landMark[8].y * h)
        centerX, centerY = (thumbX + pointX) // 2, (thumbY + pointY) // 2

        # Draw the finger points and the line
        cv2.circle(img, (thumbX, thumbY), 8, (0, 0, 255), cv2.FILLED)  # thumb
        cv2.circle(img, (pointX, pointY), 8, (0, 0, 255), cv2.FILLED)  # pointer
        cv2.line(img, (thumbX, thumbY), (pointX, pointY), (255, 255, 255), 3)  # connecting line
        cv2.circle(img, (centerX, centerY), 8, (0, 255, 0), cv2.FILLED)  # center

        # Calculate length
        length = math.hypot(thumbX - pointX, thumbY - pointY)
        if length < 60: cv2.circle(img, (centerX, centerY), 8, (0, 0, 255), cv2.FILLED)  # center

        # Rate conversions
        height = np.interp(length, [60, 600], [600, 50])
        precentage = np.interp(length, [60, 600], [0, 100])

        # Active volume bar
        if precentage > 80:
            cv2.rectangle(img, (72, int(height)), (103, 600), (0, 0, 255), cv2.FILLED)
        else:
            cv2.rectangle(img, (72, int(height)), (103, 600), (0, 255, 0), cv2.FILLED)

        # Change the volume
        osascript.osascript(f"set volume output volume {precentage}")

    cv2.rectangle(img, (70, 50), (105, 600), (255, 255, 255), 3)  # Empty volume box

    # Flip the image to set the correct way
    img_mirrored = cv2.flip(img, 1)

    # Print the precentage text
    cv2.putText(img_mirrored, f'{int(precentage)}%', (1805, 640), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Print the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img_mirrored, f"FPS: {str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img_mirrored)

    if cv2.waitKey(1) == ord('q'):
        break
