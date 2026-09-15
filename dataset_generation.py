import cv2
import numpy
import os
from pathlib import Path
import random
import shutil

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

def generate_dataset(path, parts, image_size, thickness):

    if os.path.exists(path):
        shutil.rmtree(path)
    Path(path).mkdir()

    for part_name, part_size in parts:
        part_path = f"{path}/{part_name}"
        inputs_path = f"{part_path}/inputs"
        targets_path = f"{part_path}/targets"
        Path(part_path).mkdir()
        Path(inputs_path).mkdir()
        Path(targets_path).mkdir()
        for i in range(part_size):
            in_image = numpy.zeros((image_size, image_size), dtype=numpy.uint8)
            out_image = numpy.zeros((image_size, image_size), dtype=numpy.uint8)
            connect_two_points(in_image, out_image, thickness)
            nb_digits = len(str(part_size))
            image_name = f"{i + 1:0{nb_digits}d}.png"
            in_image_path = f"{inputs_path}/{image_name}"
            out_image_path = f"{targets_path}/{image_name}"
            cv2.imwrite(in_image_path, in_image * 255)
            cv2.imwrite(out_image_path, out_image * 255)
