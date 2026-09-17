from dataset_generation import connect_horizontally
from dataset_generation import connect_two_points
from dataset_generation import draw_dotted_segment
from dataset_generation import generate_dataset

path = "datasets/connect_horizontally"
parts = (
    ("train", 8000),
    ("val", 1000),
    ("test", 1000)
)
image_size = 512
thickness = 8
task = connect_horizontally
task_parameters = {"nb_points": 8}
generate_dataset(path, parts, image_size, thickness, task, task_parameters)
