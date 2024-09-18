"""
Detection Module with mediapipe
Based on: Teachable Machine, and Hand Sign Detection course for vowels of the American Sign Language
from Computer Vision Zone
Websites:
https://teachablemachine.withgoogle.com/
https://www.computervision.zone/courses/hand-sign-detection-asl/

Thur 22/08/24 15:46 CET

"""

# Import libraries
import cv2
import mediapipe as mp
import math
import numpy as np
import time

global fingers

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


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, min_track_con=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param max_hands: Maximum number of hands to detect
        :param detection_con: Minimum Detection Confidence Threshold
        :param min_track_con: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_con,
                                         min_tracking_confidence=self.min_track_con)
        self.results = self.hands.process
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lm_list = []

    def find_hands(self, img, draw=True, flip_type=True):
        """
        Find hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []
        h, w, c = img.shape
        black = np.zeros((400, 470, 3), dtype=np.uint8)  #np.zeros((1640, 3060, 3), dtype=np.uint8)
        annotated_image = black
        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                my_hand = {}
                # lm_list
                my_lm_list = []
                x_list = []
                y_list = []
                for id_, lm in enumerate(hand_lms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    my_lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                # bbox
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                box_w, box_h = x_max - x_min, y_max - y_min
                bbox = x_min, y_min, box_w, box_h
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                my_hand["lm_list"] = my_lm_list
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                if flip_type:
                    if hand_type.classification[0].label == "Right":
                        my_hand["type"] = "Left"
                    else:
                        my_hand["type"] = "Right"
                else:
                    my_hand["type"] = hand_type.classification[0].label
                all_hands.append(my_hand)

                # drawing landmarks
                hand_found = bool(self.results.multi_hand_landmarks)
                if hand_found:
                    for hand_landmarks in self.results.multi_hand_landmarks:
                        # draw
                        if draw:
                            self.mp_draw.draw_landmarks(annotated_image, hand_landmarks,
                                                        self.mp_hands.HAND_CONNECTIONS)
                            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                          (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                          LIGHT_PINK_COLOR, 2)
                            cv2.putText(img, my_hand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                        2, MAGENTA_COLOR, 2)

        if draw:
            return all_hands, annotated_image
        else:
            return all_hands, annotated_image

    def fingers_up(self, my_hand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        my_hand_type = my_hand["type"]
        my_lm_list = my_hand["lm_list"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if my_hand_type == "Right":
                if my_lm_list[self.tip_ids[0]][0] > my_lm_list[self.tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if my_lm_list[self.tip_ids[0]][0] < my_lm_list[self.tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id_ in range(1, 5):
                if my_lm_list[self.tip_ids[id_]][1] < my_lm_list[self.tip_ids[id_] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def find_position(self, img, handNo=0, draw=False):
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            # print(my_hand)
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, MAGENTA_COLOR, cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            # bbox = x_min, y_min, x_max, y_max
            bbox.append(x_min)
            bbox.append(y_min)
            bbox.append(x_max)
            bbox.append(y_max)

            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                              NEON_GREEN_COLOR, 2)

        return self.lm_list, bbox

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), LIGHT_BLUE_COLOR, t)
            cv2.circle(img, (x1, y1), r, LIGHT_BLUE_COLOR, cv2.FILLED)
            cv2.circle(img, (x2, y2), r, LIGHT_BLUE_COLOR, cv2.FILLED)
            cv2.circle(img, (cx, cy), r, BLUE_COLOR, cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def find_hands_draw(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        # results1 = hand_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        hand_found = bool(self.results.multi_hand_landmarks)
        if hand_found:

            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    black = np.zeros((400, 470, 3), dtype=np.uint8)
                    # black = np.zeros((1640, 3060, 3), dtype=np.uint8)
                    annotated_image = black
                    self.mp_draw.draw_landmarks(annotated_image, hand_lms,
                                                self.mp_hands.HAND_CONNECTIONS)

                    return annotated_image


def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(detection_con=0.5, max_hands=1)
    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.find_hands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
        lm_list, bbox = detector.find_position(img)

        # Find Distance between two Landmarks. Could be same hand or different hands
        length, img, info = detector.find_distance(0, 0, img)  # with draw
        # print(length)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lm_list1 = hand1["lm_list"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingers_up(hand1)
            # print(fingers1[1]==1)

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lm_list2 = hand2["lm_list"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingers_up(hand2)
                #print(lm_list1[12][0:2], lm_list1[0][0:2])

                # length, info = detector.find_distance(lm_list1[8], lm_list2[8])  # with draw

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    MAGENTA_COLOR, 3)

        # Display
        img_flip = cv2.flip(img, 1)
        cv2.imshow("Image", img_flip)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
