import cv2
import numpy
import random

IMAGE_SIZE = 1024

image = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=numpy.uint8)

cv2.line(
    image,
    pt1=(100, 100),
    pt2=(100, 100),
    color=1,
    thickness=30
)

cv2.imwrite("datasets/test.png", image * 255)

def get_random_position(image_size, thickness):
    x = random.randint(0 + thickness // 2, image_size - 1 - thickness // 2)
    y = random.randint(0 + thickness // 2, image_size - 1 - thickness // 2)
    return x, y

def connect_two_points(in_image, out_image, thickness):
    image_size = in_image.shape[0]
    x1, y1 = get_random_position(image_size, thickness)
    x2, y2 = get_random_position(image_size, thickness)
    cv2.line(
        in_image,
        pt1=(x1, y1),
        pt2=(x1, y1),
        color=1,
        thickness=thickness
    )
    cv2.line(
        in_image,
        pt1=(x2, y2),
        pt2=(x2, y2),
        color=1,
        thickness=thickness
    )
    cv2.line(
        out_image,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=1,
        thickness=thickness
    )

in_image = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=numpy.uint8)
out_image = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=numpy.uint8)
thickness = 30

connect_two_points(in_image, out_image, thickness)

cv2.imwrite("datasets/in_image.png", in_image * 255)
cv2.imwrite("datasets/out_image.png", out_image * 255)
