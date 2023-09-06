import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

vide0 = cv2.VideoCapture(0)

while True:
    ret, frame = vide0.read()
    box, label, confi = cv.detect_common_objects(frame)
    output = draw_bbox(frame, box, label, confi)

    cv2.imshow("Object Detection", output)

    if cv2.waitKey(1) & 0xFF == ord(" "):
        break
