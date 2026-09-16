from dataset_generation import generate_dataset

path = "datasets/dataset1"
parts = (
    ("train", 8000),
    ("val", 1000),
    ("test", 1000)
)
image_size = 512
thickness = 15
generate_dataset(path, parts, image_size, thickness)
