from pathlib import Path
from .loss import loss_fn
from torch.utils.data import DataLoader
from .model import Yolo
from .data import VocDataset
import torch


import torchvision.transforms as T

from utils import nms_for_all_class


def main():
    model = Yolo(7, 2, 20)
    dl = DataLoader(
        VocDataset(
            Path("/home/akarshj/Programming/object_detection/train_data/train.csv"),
            transform=torch.nn.Sequential(
                *[
                    T.Resize((448, 448), antialias=True),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mean_loss = []
    for epoch in range(15):
        for image, labels in dl:
            preds = model(image)
            loss = loss_fn(preds, labels, 20, 2, 5, 0.5, 7)
            mean_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output = nms_for_all_class(
                preds.reshape(len(preds), -1, 30),
                0.5,
                0.5,
                divisions=7,
                img_dim=448,
            )
        print(sum(mean_loss) / len(mean_loss))


if __name__ == "__main__":
    main()
