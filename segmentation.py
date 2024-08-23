from typing import Optional

import cv2 as cv
import numpy as np


def segment_TB_pixels(
    img: np.ndarray,
    gray_img: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Segments the Trypan Blue-positive (TB) pixels from the corneal image (with the
    cornea already segmented out) via the Watershed algorithm.

    Parameters
    ----------
    img : np.ndarray
        A 3- or 4-channel image containing the cornea. If the image has 4 channels,
        areas outside of the cornea should be transparent. If the image has 3 channels,
        areas outside of the cornea should be black.
    gray_img : np.ndarray, optional
        The grayscale version of the image. If not provided, it is computed by
        converting the image to grayscale.

    Returns
    -------
    tuple of array and float
        The watershed-segmented regions that are believed to be stained by the TB dye,
        and the viability index as the ratio of number of healthy pixels (i.e.,
        TB-negative) to the total number of pixels in the cornea.
    """
    # convert to grayscale, remove noise and smooth image, and apply a first threshold
    # to coarsely extract all suspected TB-positive pixels
    has_four_channels = img.shape[2] == 4
    if gray_img is None:
        gray_img = cv.cvtColor(
            img, cv.COLOR_BGRA2GRAY if has_four_channels else cv.COLOR_BGR2GRAY
        )
    blurred_img = cv.GaussianBlur(
        gray_img,
        (9, 9),  # NOTE: tunable (!) - pairs of positive integers
        0,  # NOTE: tunable (!) - nonnegative float
    )
    thresholded_img = cv.adaptiveThreshold(
        blurred_img,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY_INV,
        101,  # NOTE: tunable (!!) - odd integer
        3.0,  # NOTE: tunable (!!!) - float
    )

    # segment the image via the Watershed algorithm, according to the tutorial found at
    # https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html.

    # first, perform an opening to get rid of small thresholded nonzero regions
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (3, 3),  # NOTE: tunable (!!) - pairs of positive integers (odd?)
    )
    mask = cv.morphologyEx(thresholded_img, cv.MORPH_OPEN, kernel)

    # then, for each nonzero pixel compute the distance to the nearest zero pixel, and
    # use it to discern foreground, background and unknown pixels
    distance = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_3)
    _, foreground = cv.threshold(
        distance,
        0.1 * distance.max(),  # NOTE: tunable (!!!) - threshold multiplier
        255,
        cv.THRESH_BINARY,
    )
    foreground = foreground.astype(np.uint8)
    background = cv.dilate(
        mask,
        kernel=kernel,
        iterations=3,  # NOTE: tunable (?) - nonnegative integer
    )
    unknown = cv.subtract(background, foreground)

    # finally, apply the watershed algorithm with the foreground as seed
    _, labels = cv.connectedComponents(foreground)
    labels = labels + 1  # shift all the labels up by one
    labels[unknown > 0] = 0  # and assign label 0 to the unknown region
    markers = cv.watershed(img[..., :3], labels)  # remove alpha channel if present
    # segmentation = img[..., :3].copy()
    # segmentation[markers == -1] = [255, 0, 0]  # boundaries are marked with -1
    # plt.imshow(segmentation)
    # plt.axis("off")
    # plt.show()

    # now that the watershed segmentation is available, we need to extract the contours
    # (i.e., boundaries) of all the nonzero regions. We find the contours of the
    # segmented regions via thresholding + findContours (see
    # https://stackoverflow.com/a/50889494/19648688), but we additionally take care to
    # remove contours that extend outside the cornea
    _, contours_mask = cv.threshold(
        markers.astype(np.uint8), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )

    # compute the mask associated to only those pixels belonging to the cornea
    corneal_mask = img[..., -1] > 0 if has_four_channels else gray_img > 0
    contours_mask = np.where(corneal_mask, contours_mask, np.uint8(0))
    # corneal_mask = np.where(
    #     img[..., -1] > 0 if has_four_channels else gray_img > 0,
    #     np.uint8(255),
    #     np.uint8(0),
    # )
    # size = min(corneal_mask.shape) // 100
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
    # shrunk_corneal_mask = cv.erode(corneal_mask, kernel)
    # contours_mask = cv.bitwise_and(
    #     contours_mask, contours_mask, mask=shrunk_corneal_mask
    # )

    # finally, find all the contours (hierarchy information is disregarded)
    contours, _ = cv.findContours(contours_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # create a mask by filling all found contours, thus highlighting all the pixels that
    # we believe to be positive to the TB stain
    tb_positive_mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(len(contours)):
        cv.drawContours(tb_positive_mask, contours, i, 255, cv.FILLED)
    # img[tb_positive_mask > 0] = (255, 0, 0, 255) if has_four_channels else (255, 0, 0)
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
    # plt.axis("off")
    # plt.show()

    # lastly, compute the viability index
    viability = 1 - cv.countNonZero(tb_positive_mask) / corneal_mask.sum()
    return tb_positive_mask, viability


def calculate_enclosing_circle(gray_img: np.ndarray) -> tuple[int, int, int]:
    """Computes the circle that encloses the corneal segmented image.

    Parameters
    ----------
    gray_img : np.ndarray, optional
        The grayscale version of the corneal segmented image.

    Returns
    -------
    tuple of 3 ints
        The x, y, and radius of the circle that encloses the corneal segmented image.
    """
    # extract the contours in the image
    blurred_image = cv.GaussianBlur(gray_img, (71, 71), 11)
    _, thresholded_img = cv.threshold(
        blurred_image, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )
    cnts, _ = cv.findContours(thresholded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 1, "Expected only one contour."

    (x, y), r = cv.minEnclosingCircle(cnts[0])
    x, y, r = int(x), int(y), int(r)
    # cv.circle(img, (x, y), r // 4, (255, 0, 0), 2)
    # cv.circle(img, (x, y), r // 2, (255, 0, 0), 2)
    # cv.circle(img, (x, y), r * 3 // 4, (255, 0, 0), 2)
    # cv.circle(img, (x, y), r, (255, 0, 0), 2)
    # cv.circle(img, (x, y), 7, (0, 0, 0), -1)
    # cv.putText(
    #     img,
    #     "center",
    #     (x - 20, y - 20),
    #     cv.FONT_HERSHEY_SIMPLEX,
    #     1.5,
    #     (0, 0, 0),
    #     2,
    # )
    # plt.imshow(img)
    # plt.show()
    return x, y, r


if __name__ == "__main__":
    path = r"test_imgs/VCHECK_52_cornea.png"
    img = cv.imread(path, cv.IMREAD_UNCHANGED)  # BGR or BGRA
    if img is None:
        print("Could not open or find the image:", path)
        exit(0)
    gray_img = cv.cvtColor(
        img, cv.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv.COLOR_BGR2GRAY
    )
    tb_positive_mask, viability = segment_TB_pixels(img, gray_img)
    # circles = calculate_enclosing_circle(gray_img)
    print("Viability index:", viability)
