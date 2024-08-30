"""Many thanks to https://stackoverflow.com/a/59906154/19648688."""

import sys

import cv2 as cv
import numpy as np


def nothing(x):
    pass


if __name__ == "__main__":
    # load image
    if len(sys.argv) != 2:
        print("Usage: python hls_selector.py <path_to_image>")
        exit(1)
    path = sys.argv[1]
    img = cv.imread(path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", path)
        exit(1)

    # Create a window
    cv.namedWindow("image")

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv.createTrackbar("HMin", "image", 0, 179, nothing)
    cv.createTrackbar("LMin", "image", 0, 255, nothing)
    cv.createTrackbar("SMin", "image", 0, 255, nothing)
    cv.createTrackbar("HMax", "image", 0, 179, nothing)
    cv.createTrackbar("LMax", "image", 0, 255, nothing)
    cv.createTrackbar("SMax", "image", 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv.setTrackbarPos("HMax", "image", 179)
    cv.setTrackbarPos("LMax", "image", 255)
    cv.setTrackbarPos("SMax", "image", 255)

    # Initialize HSV min/max values
    hMin = lMin = sMin = hMax = lMax = sMax = 0
    phMin = plMin = psMin = phMax = plMax = psMax = 0

    while True:
        # Get current positions of all trackbars
        hMin = cv.getTrackbarPos("HMin", "image")
        lMin = cv.getTrackbarPos("LMin", "image")
        sMin = cv.getTrackbarPos("SMin", "image")
        hMax = cv.getTrackbarPos("HMax", "image")
        lMax = cv.getTrackbarPos("LMax", "image")
        sMax = cv.getTrackbarPos("SMax", "image")

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, lMin, sMin])
        upper = np.array([hMax, lMax, sMax])

        # Convert to HSV format and color threshold
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HLS)
        mask = cv.inRange(hsv, lower, upper)
        # result = cv.bitwise_and(img, img, mask=mask)
        result = img.copy()
        result[mask > 0] = [255, 0, 0, 255] if result.shape[2] == 4 else [255, 0, 0]

        # Display result image
        cv.imshow("image", result)
        if cv.waitKey(10) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()
