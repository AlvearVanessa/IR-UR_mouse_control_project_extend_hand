""""
AI Virtual Mouse
Mon 26/08/24 08:37 CET
@uthor: maalvear
Source: https://www.computervision.zone/courses/ai-virtual-mouse/

"""

import cv2
import numpy as np
import HandTrackingModule_blackBackground as htm
import time
import autopy
import mouse

##########################
w_cam, h_cam = 1280, 720
frame_reduction = 80  # Frame Reduction
#########################

p_time = 0
ploc_x, ploc_y = 0, 0
cloc_x, cloc_y = 0, 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# time.sleep(2)
cap.set(3, w_cam)
cap.set(4, h_cam)

detector = htm.HandDetector(max_hands=1)
w_screen, h_screen = autopy.screen.size()
# print(w_screen, h_screen)

threshold_click = 40
threshold_scroll_up = 15 # 50
threshold_scroll_down = 35
smoothing = 7

# Find Function
# x is the raw distance y is the value in cm
# x_ = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
# y_ = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# coff = np.polyfit(x_, y_, 2)  # y = Ax^2 + Bx + C

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

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()

    if success:
        hands, img = detector.find_hands(img)  # with draw
        lm_list, bbox = detector.find_position(img)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lm_list1 = hand1["lm_list"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            center_point_1 = hand1['center']  # center of the hand cx,cy
            hand_type_1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingers_up(hand1)
            # print(fingers1[1])

            # 8. Check if we are in clicking mode: Both index and middle fingers are up: Clicking Mode
            # length, img, lineInfo = detector.findDistance(lm_list[12][0:2], lm_list[0][0:2], img)
            # print(length)

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]
                x2, y2 = lm_list[12][1:]

                # 4. Only Index Finger : Moving Mode
                if fingers1[1] == 1 and fingers1[2] == 0:

                    # 5. Convert Coordinates
                    x3 = np.interp(x1, (frame_reduction, w_cam - 11 * frame_reduction), (0, w_screen))
                    # (0, h_screen + 200)
                    y3 = np.interp(y1, (frame_reduction, h_cam - 5 * frame_reduction), (0, h_screen + 200))
                    # 6. Smoothen Values (to avoid the mouse checking)
                    cloc_x = ploc_x + (x3 - ploc_x) / smoothing
                    cloc_y = ploc_y + (y3 - ploc_y) / smoothing
                    # print("Coordinates ", cloc_x, cloc_y)

                    # 7. Move Mouse
                    x_move_screen = w_screen - cloc_x
                    y_move_screen = cloc_y

                    autopy.mouse.move(x_move_screen, y_move_screen)
                    # time.sleep(0.05)
                    # print(autopy.mouse.move(w_screen - cloc_x, cloc_y), (w_screen - cloc_x, cloc_y))
                    cv2.circle(img, (x1, y1), 20, NEON_GREEN_COLOR, cv2.FILLED)
                    ploc_x, ploc_y = cloc_x, cloc_y

                    # AA = autopy.mouse.move(x_move_screen, y_move_screen)
                    # if AA is not None:
                        # autopy.mouse.move(355, 488)
                    # else:
                        # AA
                    
                    #while x_move_screen < 10 and y_move_screen > 1278:
                    #    autopy.mouse.move(355, 488)
                    #    break
                    #while x_move_screen < 363 and y_move_screen > 1278:
                    #    autopy.mouse.move(355, 488)
                    #    break
                    #while x_move_screen < 631 and y_move_screen > 1230:
                    #    autopy.mouse.move(355, 488)
                    #    break
            # 8. Check if we are in clicking mode: Both index and middle fingers are up: Clicking Mode
            length, _, lineInfo = detector.find_distance(12, 0, img)
            print("CLICK = ", length)

            # 10. Click mouse if distance short
            if fingers1[0] == 1 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 1 and fingers1[4] == 1:
                if length > 60:
                    # 9. Find distance between fingers
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, NEON_GREEN_COLOR, cv2.FILLED)
                    autopy.mouse.click()
                    time.sleep(0.75)  # 0.75


            # 12. Mouse scrolling bottom up
            if fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0:
                time.sleep(0.02)  # time.sleep(0.05)
                # 13. Find distance between fingers
                length2, img, lineInfo2 = detector.find_distance(8, 12, img)
                # print("Mouse scrolling UP= ", length2)
                # 14. Clock mouse if distance short
                if length2 < threshold_scroll_up:
                    cv2.circle(img, (lineInfo2[4], lineInfo2[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=1)

            # 12. Mouse scrolling bottom DOWN
            if fingers1[1] == 0 and fingers1[2] == 0:
                time.sleep(0.005)  # time.sleep(0.07)
                # 13. Find distance between fingers
                length3, img, lineInfo3 = detector.find_distance(8, 12, img)
                # print("Mouse scrolling DOWN= ", length3)
                # 14. Clock mouse if distance short
                if length3 < threshold_scroll_down:
                    cv2.circle(img, (lineInfo3[4], lineInfo3[5]),
                               15, BLUE_COLOR, cv2.FILLED)
                    mouse.wheel(delta=-1)

            lm_list, bbox = detector.find_position(img)
            x, y, w, h = bbox
            # print(lm_list[5][1:])
            # x and y position for landmk 5 and 17
            x1, y1 = lm_list[5][1:]
            x2, y2 = lm_list[17][1:]
            # print(x1, x2, y1, y2)
            # distance = int(math.sqrt((abs(x2 - x1)) ** 2 + (abs(y2 - y1)) ** 2))
            # print(distance)

            # A, B, C = coff
            # distanceCM1 = A * distance ** 2 + B * distance + C
            # print(distanceCM1, distance)

        # Print rectangle to resize screen
        # cv2.rectangle(img, (frame_reduction, frame_reduction), (w_cam - 43 * frame_reduction, h_cam - (18 * frame_reduction + 400)),
        #                       (255, 0, 255), 5)

        # Print rectangle to resize screen (Edge)
        cv2.rectangle(img, (frame_reduction-20, frame_reduction-20), (w_cam - 11 * frame_reduction, h_cam - (5 * frame_reduction)),
                                                                MAGENTA_COLOR, 5)

        #cv2.putText(img, f'{int(distanceCM1)} cm', (x + 5, y - 10), cv2.FONT_HERSHEY_PLAIN, 3,
        #            (0, 255, 128), 3)

        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - p_time)
        p_time = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    BLUE_COLOR, 3)

        # 12. Display
        cv2.namedWindow("Big Screen")  # Create a named window
        cv2.moveWindow("Big Screen", 900, 400)  # Move it to (x,y) (0, 110)
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Big Screen", img_flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
