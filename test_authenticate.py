import argparse

import cv2
import imutils

import authenticate_user
import common
import data_handler

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="Image on which to run recognition.", required=True)
args = parser.parse_args()

frame = cv2.imread(args.image)
frame = imutils.resize(frame, width=750)

authenticate_user.run_face_recognition(frame, data_handler.load_database(common.DATABASE_LOC), draw_rectangles=True)

while cv2.waitKey(1) & 0xFF != ord('q') and cv2.getWindowProperty(common.FRAME_NAME, 0) >= 0:
    pass
cv2.destroyAllWindows()
