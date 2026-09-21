from dataset_generation import connect_horizontally
from dataset_generation import connect_to_k_nearest_neighbors
from dataset_generation import connect_two_points
from dataset_generation import draw_circle_from_three_points
from dataset_generation import draw_circle_from_two_points
from dataset_generation import draw_convex_hull
from dataset_generation import draw_delaunay_triangulation
from dataset_generation import draw_dotted_segment
from dataset_generation import draw_greedy_matching
from dataset_generation import draw_minimum_spanning_tree
from dataset_generation import generate_dataset

path = "datasets/draw_delaunay_triangulation"
parts = (
    ("train", 8000),
    ("val", 1000),
    ("test", 1000)
)
image_size = 512
thickness = 8
task = draw_delaunay_triangulation
task_parameters = {"nb_points": 16}
generate_dataset(path, parts, image_size, thickness, task, task_parameters)
