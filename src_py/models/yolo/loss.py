from torch import concat
from torch.autograd._functions import torch
from torch.functional import Tensor

from .utils import calculate_ious


def loss_fn(
    predictions: Tensor,
    targets: Tensor,
    n_classes: int,
    n_bbox_predictors: int,
    lambdacoord: float,
    lambdanoobj: float,
    split_size: int,
):
    """
    prediction: 30 x 7 x 7
    target: 25 x 7 x 7
    """

    batch_size = len(predictions)  # 2

    # # for predictions
    predictions = predictions.reshape(
        -1, split_size**2, n_classes + n_bbox_predictors * 5
    )  # (2, 49, 30)
    targets = targets.reshape(-1, split_size**2, n_classes + 5)  # (2, 49, 25)

    pred_bboxes = predictions[..., n_classes:]  # (2, 49, 10)
    pred_bboxes = pred_bboxes.reshape(
        *pred_bboxes.shape[:-1], n_bbox_predictors, 5
    )  # (2, 49, 2, 5)
    pred_class_probabilities = predictions[..., :n_classes]  # (2, 49, 20)

    # for targets
    targets_bboxes = targets[..., n_classes:].unsqueeze(2)  # 2, 49, 1, 5
    targets_class_probabilities = targets[..., :n_classes]  # (2, 49, 20)

    ious = calculate_ious(
        pred_bboxes[..., 0, 1:].reshape(-1, 4),
        targets_bboxes[..., 0, 1:].reshape(-1, 4),
        "midpoint",
    )  # (2 * 49, 1)

    for i in range(1, n_bbox_predictors):
        iou = calculate_ious(
            pred_bboxes[..., i, 1:].reshape(-1, 4),  # (98, 4)
            targets_bboxes[..., 0, 1:].reshape(-1, 4),  # (98, 4)
            "midpoint",
        )  # (2 * 49, 1)
        ious = concat([ious, iou], dim=1)

    _, best_iou_indices = ious.max(dim=1)  # best_iou_indices.shape = (98,)
    object_containing_indices = targets_bboxes.reshape(-1, 5)[:, 0] == 1  # (98,)

    temp = pred_bboxes.reshape(-1, n_bbox_predictors, 5)  # (98, 2, 5)
    gather_index = best_iou_indices.view(-1, 1, 1).expand(
        -1, 1, temp.size(-1)
    )  # (98, 1, 5)
    max_iou_preds = temp.gather(1, gather_index)

    max_iou_preds = max_iou_preds.squeeze(1)  # (98, 5)

    obj_max_iou_preds_bbox = max_iou_preds[object_containing_indices]  # (3, 5)
    obj_target_bbox = targets_bboxes.view(-1, 5)[object_containing_indices]  # (3, 5)
    obj_pred_max_iou_class_proba = pred_class_probabilities.view(-1, n_classes)[
        object_containing_indices
    ]  # (3, 20)
    obj_target_class_proba = targets_class_probabilities.view(-1, n_classes)[
        object_containing_indices
    ]  # (3, 20) one hot

    noobj_bboxes = pred_bboxes.reshape(-1, n_bbox_predictors, 5)[
        ~object_containing_indices
    ].reshape(
        -1, 5
    )  # pred_bboxes.reshape(95, 2, 5) (95 * 2, 5)

    localization_loss = lambdacoord * (
        ((obj_target_bbox[:, 1:3] - obj_max_iou_preds_bbox[:, 1:3]) ** 2).sum()
        + (
            (obj_target_bbox[:, 3:5].sqrt() - obj_max_iou_preds_bbox[:, 3:5].sqrt())
            ** 2
        ).sum()
    )

    confidence_loss = (
        (obj_target_bbox[:, 0] - obj_max_iou_preds_bbox[:, 0]) ** 2
    ).sum()

    confidence_loss_no_obj = lambdanoobj * ((0 - noobj_bboxes[:, 0]) ** 2).sum()

    classification_loss = (
        (obj_target_class_proba - obj_pred_max_iou_class_proba) ** 2
    ).sum()

    total_loss = (
        localization_loss
        + confidence_loss
        + confidence_loss_no_obj
        + classification_loss
    )

    return total_loss / batch_size


def main():
    preds = torch.randn((30, 7, 7))
    targets = torch.randn((25, 7, 7))
    print(
        loss_fn(
            preds,
            targets,
            n_classes=20,
            n_bbox_predictors=2,
            lambdacoord=5,
            lambdanoobj=0.5,
            split_size=7,
        )
    )


if __name__ == "__main__":
    main()
