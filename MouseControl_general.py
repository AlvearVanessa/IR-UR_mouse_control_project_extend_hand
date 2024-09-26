""""
AI Virtual Mouse - General cameras and screen
Thur 26/09/24 08:43 CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/

"""

import cv2
import numpy as np
import HandTrackingModule_general as htm
import time
import autopy
import mouse
# import tkinter as tk

# Define color variables regarding OpenCV
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
LIGHT_PINK_COLOR = (255, 153, 255)
MAGENTA_COLOR = (255, 0, 255)
LIGHT_BLUE_COLOR = (255, 255, 0)
YELLOW_COLOR = (0, 255, 255)
NEON_GREEN_COLOR = (0, 255, 0)
ORANGE_COLOR = (0, 128, 255)
LAVENDER_COLOR = (255, 0, 128)

# Define variables
frame_reduction = 100  # Frame Reduction
p_time = 0
ploc_x, ploc_y = 0, 0
cloc_x, cloc_y = 0, 0
threshold_click = 40
threshold_scroll_up = 25 # 50
threshold_scroll_down = 35
smoothing = 5

# Define screen size
w_screen, h_screen = autopy.screen.size()
print("w_screen, h_screen", w_screen, h_screen)

# Caption of the camera in real-time
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Find 1 hand detected (closer to the camera)
detector = htm.HandDetector(max_hands=1)

# Define loop to interact with mouse functions using the hand detected
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    # Define height and width of the camera based on image shape (camera size)
    h_cam, w_cam, _ = img.shape
    print("h_cam, w_cam", h_cam, w_cam)
    # 2. Define the interactions: move, scroll-up, scroll-down, click
    if success:
        hands, img = detector.find_hands(img)  # with draw
        lm_list, bbox = detector.find_position(img)

        if hands:
            # 3. Select 1 hand
            hand1 = hands[0]
            # lm_list1 = hand1["lm_list"]  # List of 21 Landmark points in x, y, and z pixel position
            fingers1 = detector.fingers_up(hand1)
            # 4. Select x and y pixel position od landmarks 8 and 12 (index and middle tips)
            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]
                x2, y2 = lm_list[12][1:]

                # 5. Only Index Finger : Moving Mode, if index up and middle finger is down
                if fingers1[1] == 1 and fingers1[2] == 0:
                    # 6. Convert Coordinates to resize screen and makes the move short in real, long in screen
                    x3 = np.interp(x1, (frame_reduction, w_cam-frame_reduction), (0, w_screen))
                    y3 = np.interp(y1, (frame_reduction, h_cam-frame_reduction), (0, h_screen))
                    # 7. Smoothen Values (to avoid the mouse checking)
                    cloc_x = ploc_x + (x3 - ploc_x) / smoothing
                    cloc_y = ploc_y + (y3 - ploc_y) / smoothing

                    # 8. Move Mouse
                    x_move_screen = w_screen - cloc_x
                    y_move_screen = cloc_y
                    autopy.mouse.move(x_move_screen, cloc_y)
                    # time.sleep(0.002)
                    cv2.circle(img, (x1, y1), 20, NEON_GREEN_COLOR, cv2.FILLED)
                    ploc_x, ploc_y = cloc_x, cloc_y

            # 9. Find distance between fingers landmark 12 and 0
            length, _, lineInfo = detector.find_distance(12, 0, img)
            # print("line info 12 and 0", lineInfo)

            # 10. Clicking Mode if distance greater than a value and all fingers are up
            if fingers1[0] == 1 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 1:
                if length > 33:
                    # 11. Find distance between fingers landmark 12 and 0
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, NEON_GREEN_COLOR, cv2.FILLED)
                    autopy.mouse.click()
                    time.sleep(0.75)  # 0.75

            # 12. Mouse scrolling bottom up
            if fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0:
                time.sleep(0.04)  # time.sleep(0.05)
                # 13. Find distance between fingers
                length2, img, lineInfo2 = detector.find_distance(8, 12, img)
                # 14. Clock mouse if distance short
                if length2 < threshold_scroll_up:
                    cv2.circle(img, (lineInfo2[4], lineInfo2[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=1)

            # 15. Mouse scrolling bottom DOWN
            if fingers1[1] == 0 and fingers1[2] == 0:
                time.sleep(0.05)  # time.sleep(0.07)
                # 16. Find distance between fingers
                length3, img, lineInfo3 = detector.find_distance(8, 12, img)
                print("length3=", length3)
                # 17. Clock mouse if distance short
                if length3 < threshold_scroll_down:
                    cv2.circle(img, (lineInfo3[4], lineInfo3[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=-1)

            # 18. x and y position for landmark 5 and 17
            x1, y1 = lm_list[5][1:]
            x2, y2 = lm_list[17][1:]

        # 19. Print rectangle to resize screen (Edge)
        cv2.rectangle(img, (frame_reduction, frame_reduction), (w_cam-frame_reduction, h_cam-frame_reduction),
                      MAGENTA_COLOR, 4)

        # 20. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - p_time)
        p_time = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    BLUE_COLOR, 3)

        # 21. Display
        # cv2.namedWindow("Big Screen")  # Create a named window
        # cv2.moveWindow("Big Screen", int(w_screen/3), int(h_screen/4))  # Move it to (x,y) (0, 110)
        img_flip = cv2.flip(img, 1)
        # cv2.imshow("Big Screen", img_flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

