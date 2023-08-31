from pathlib import Path
from typing import List
from PIL.ImageDraw import ImageDraw
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.celeba import csv
from torchvision.io import read_image
import torchvision.transforms as T

# from pillow


def read_csv(path: Path) -> list[tuple[Path, Path]]:
    image_dir: Path = path.parent.joinpath("images")
    label_dir: Path = path.parent.joinpath("labels")
    out = []

    with path.open("r") as f:
        reader = csv.reader(f)
        for line in reader:
            image_filename, label_filename = line
            image_file = image_dir.joinpath(image_filename)
            label_file = label_dir.joinpath(label_filename)
            out.append((image_file, label_file))
    return out


def center_to_rect(x, y, w, h):
    x1 = x - (w / 2)
    y1 = y - (h / 2)
    x2 = x + (w / 2)
    y2 = y + (h / 2)
    return x1, y1, x2, y2


def im_show(label_file: Path, image_file: Path):
    # Convert the tensor to a PIL Image
    img = Image.open(image_file)
    img = img.resize((448, 448))
    one_part = img.width / 7

    shapes = []
    for j in range(7):
        shapes.append(((0, one_part * j), (447, one_part * j)))
    for i in range(7):
        shapes.append(((i * one_part, 0), (i * one_part, 447)))

    for shape in shapes:
        img1 = ImageDraw(img)
        img1.line(shape, fill="white", width=0)

    with label_file.open("r") as lf:
        for line in lf.readlines():
            x, y, w, h = map(float, line.split()[1:])

            points = map(lambda x: int(448 * x), [x, y, w, h])
            x, y, w, h = points

            i, j = x / one_part, y / one_part
            print(i, j)
            img1 = ImageDraw(img)
            img1.point((x, y), fill="blue")

            img1.rectangle(center_to_rect(*[x, y, w, h]), outline="red")

    img.show()


class VocDataset(Dataset):
    def __init__(
        self,
        record_file: Path,
        transform=None,
        target_transform=None,
        imaginary_div: int = 7,
        classes: int = 20,
    ):
        self.record_file = record_file
        line_count = 0
        with record_file.open("r") as f:
            for _ in f:
                line_count += 1
        self.line_count = line_count
        self.img_labels = read_csv(self.record_file)
        self.transform = transform
        self.target_transform = target_transform
        self.imaginary_div = imaginary_div
        self.classes = classes

    def __len__(self):
        return self.line_count

    def __getitem__(self, idx):
        image = read_image(str(self.img_labels[idx][0])).type(torch.float32)
        labels = self.read_label_file(self.img_labels[idx][1])
        labels = create_target_vec(self.imaginary_div, self.classes, labels)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image / 255, labels

    def read_label_file(self, file: Path) -> List[List[str]]:
        with file.open("r") as lf:
            return [line.split() for line in lf.readlines()]


def create_target_vec(
    imaginary_divisions: int, n_classes: int, objects: List[List[str]]
):
    target = torch.zeros((imaginary_divisions, imaginary_divisions, n_classes + 5))

    for obj in objects:
        one_cell_width = 1 / imaginary_divisions
        class_, x, y, w, h = map(float, obj)
        y_dim = int(y // one_cell_width)
        x_dim = int(x // one_cell_width)
        cell_origin = (one_cell_width * x_dim, one_cell_width * y_dim)
        wrt_cell = ((x - cell_origin[0]) / one_cell_width), (
            (y - cell_origin[1]) / one_cell_width
        )  # ratio wrt to the cell
        k = torch.nn.functional.one_hot(
            torch.tensor(int(class_)), num_classes=n_classes
        )
        target[y_dim, x_dim, :] = torch.concat(
            [
                k,
                torch.tensor([1.0]),
                torch.tensor([wrt_cell[0]]),
                torch.tensor([wrt_cell[1]]),
                torch.tensor([w]),
                torch.tensor([h]),
            ]
        )
    return target
