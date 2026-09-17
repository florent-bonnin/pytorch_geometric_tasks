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

def draw_dotted_segment(in_image, out_image, thickness, nb_steps):
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
        pt2=(x1, y1),
        color=1,
        thickness=thickness
    )
    cv2.line(
        out_image,
        pt1=(x2, y2),
        pt2=(x2, y2),
        color=1,
        thickness=thickness
    )
    x_segment = x2 - x1
    y_segment = y2 - y1
    x_step = x_segment / (nb_steps + 1)
    y_step = y_segment / (nb_steps + 1)
    x_dot = x1
    y_dot = y1
    for i in range(nb_steps):
        x_dot += x_step
        y_dot += y_step
        cv2.line(
            out_image,
            pt1=(round(x_dot), round(y_dot)),
            pt2=(round(x_dot), round(y_dot)),
            color=1,
            thickness=thickness
        )

def connect_horizontally(in_image, out_image, thickness, nb_points):
    image_size = in_image.shape[0]
    points = []
    for i in range(nb_points):
        x, y = get_random_position(image_size, thickness)
        points.append((x, y))
        cv2.line(
            in_image,
            pt1=(x, y),
            pt2=(x, y),
            color=1,
            thickness=thickness
        )
    points = sorted(points)
    for i in range(nb_points - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        cv2.line(
            out_image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=1,
            thickness=thickness
        )

def generate_dataset(path, parts, image_size, thickness, task, task_parameters):

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
            task(in_image, out_image, thickness, **task_parameters)
            nb_digits = len(str(part_size))
            image_name = f"{i + 1:0{nb_digits}d}.png"
            in_image_path = f"{inputs_path}/{image_name}"
            out_image_path = f"{targets_path}/{image_name}"
            cv2.imwrite(in_image_path, in_image * 255)
            cv2.imwrite(out_image_path, out_image * 255)
