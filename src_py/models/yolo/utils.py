from collections import defaultdict
import torch
from torch import Tensor
import torchvision


def xywh_to_x1y1x2y2(t: Tensor):
    x1 = t[..., 0:1] - t[..., 2:3] / 2
    y1 = t[..., 1:2] - t[..., 3:4] / 2
    x2 = t[..., 0:1] + t[..., 2:3] / 2
    y2 = t[..., 1:2] + t[..., 3:4] / 2
    return torch.concat([x1, y1, x2, y2], dim=-1)


def calculate_ious(boxes_preds: Tensor, boxes_labels: Tensor, box_format: str):
    """
    pred_bboxes: Mx4 [x1, y1, x2, y2]
    target_bboxes: Mx4 [x1, y1, x2, y2]
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = ((box1_x2 - box1_x1) * (box1_y2 - box1_y1)).abs()
    box2_area = ((box2_x2 - box2_x1) * (box2_y2 - box2_y1)).abs()

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def nms_for_all_class(
    boxes_preds: Tensor,
    prob_score: float,
    iou_threshold: float,
    divisions: int,
    img_dim: float = 448.0,
):
    """
    class: {
                bboxes: tensor[n, 4]
            }
    Returns: in the format x1, x2, y1, y2 with relative to cell
    (batch_size, 49, 30)
    """

    classes = boxes_preds[:, :, :20].argmax(-1)  # (2, 49, 5)

    shape = boxes_preds.shape
    boxes_preds = boxes_preds[..., 20:].reshape(
        shape[0], shape[1], 2, 5
    )  # (2, 49, 2, 5)
    _, max_indices = boxes_preds[..., 0].max(-1)  # (2, 49)

    gather_index = max_indices.view(shape[0], shape[1], 1, 1).expand(
        shape[0], shape[1], 1, 5
    )

    max_iou_preds = boxes_preds.gather(2, gather_index)

    combined = torch.concat([classes.unsqueeze(-1), max_iou_preds.squeeze(-2)], dim=-1)
    combined, _ = combined.sort(dim=1, descending=True)
    x1y1x2y2 = torch.concat(
        [combined[..., 0:2], xywh_to_x1y1x2y2(combined[..., 2:])], dim=-1
    )

    class_bbox_dict = defaultdict(Tensor)

    for img in x1y1x2y2:  # 0, 2
        class_ = img[:, 0]  # for class
        proba = img[:, 1]
        bbox = img[:, 2:]
        for c in class_.unique():
            # we also need to append only the results whose probablility score is higher than threshold
            for_class = (class_ == c) & (proba > prob_score)
            bboxes = bbox[for_class]
            scores = proba[for_class]
            cells = for_class.nonzero().squeeze()

            res = torchvision.ops.nms(bboxes, scores, iou_threshold)
            if len(res) > 0:
                bboxes = bboxes.index_select(0, res)
                cells = cells.index_select(0, res)
                size = x1y1x2y2_relative_to_img(
                    img_dim, divisions, bboxes, cells.unsqueeze(-1)
                )
                class_bbox_dict[c] = size
    return class_bbox_dict


def x1y1x2y2_relative_to_img(
    img_dim: float, divisions: int, tensor: Tensor, cell_ids: Tensor
):
    """
    tensor: dim [nx4]
    """
    one_cell_width = img_dim / divisions  # 448 / 7 = 64

    i = cell_ids // divisions
    j = cell_ids % divisions

    x1 = j + tensor[:, 0:1]
    y1 = i + tensor[:, 1:2]
    x2 = j + tensor[:, 2:3]
    y2 = i + tensor[:, 3:4]
    return torch.concat([x1, y1, x2, y2], dim=-1) * one_cell_width


if __name__ == "__main__":
    boxes_preds = torch.rand((2, 49, 30))
    nms_for_all_class(boxes_preds, 0.5, 0.5, 5)

    # print(
    #     x1y1x2y2_relative_to_img(
    #         5, 5, torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]), torch.tensor([[6], [4]])
    #     )
    # )
