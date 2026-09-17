from dataset_generation import connect_two_points
from dataset_generation import draw_dotted_segment
from dataset_generation import generate_dataset

path = "datasets/draw_dotted_segment"
parts = (
    ("train", 8000),
    ("val", 1000),
    ("test", 1000)
)
image_size = 512
thickness = 15
task = draw_dotted_segment
task_parameters = {"nb_steps": 2}
generate_dataset(path, parts, image_size, thickness, task, task_parameters)
