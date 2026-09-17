from dataset_generation import connect_horizontally
from dataset_generation import connect_two_points
from dataset_generation import draw_circle_from_two_points
from dataset_generation import draw_convex_hull
from dataset_generation import draw_dotted_segment
from dataset_generation import draw_minimum_spanning_tree
from dataset_generation import generate_dataset

path = "datasets/draw_minimum_spanning_tree"
parts = (
    ("train", 8000),
    ("val", 1000),
    ("test", 1000)
)
image_size = 512
thickness = 4
task = draw_minimum_spanning_tree
task_parameters = {"nb_points": 8}
generate_dataset(path, parts, image_size, thickness, task, task_parameters)
