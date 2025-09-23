
import numpy as np


def calculate_iou(box1, box2):
    """
    두 개의 바운딩 박스 간의 Intersection over Union (IoU)를 계산합니다.

    Args:
        box1 (list): 첫 번째 바운딩 박스. 형식: [x1, y1, x2, y2]
        box2 (list): 두 번째 바운딩 박스. 형식: [x1, y1, x2, y2]

    Returns:
        float: 0과 1 사이의 IoU 값.
    """
    # 교차 영역(intersection)의 좌표 계산
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    # 교차 영역의 너비와 높이 계산 (겹치지 않으면 0이 됨)
    intersection_width = max(0, inter_x2 - inter_x1)
    intersection_height = max(0, inter_y2 - inter_y1)

    # 교차 영역의 넓이 계산
    intersection_area = intersection_width * intersection_height

    # 각 바운딩 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합(union) 영역의 넓이 계산
    union_area = box1_area + box2_area - intersection_area

    # IoU 계산 (0으로 나누는 경우 방지)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou



def calculate_iou_with_union_of_boxes(gt_box, pred_boxes):
    all_boxes = [gt_box] + pred_boxes
    x_min = min(box[0] for box in all_boxes)
    y_min = min(box[1] for box in all_boxes)
    x_max = max(box[2] for box in all_boxes)
    y_max = max(box[3] for box in all_boxes)

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # 캔버스(마스크)를 생성합니다. (0으로 채워진 2D 배열)
    gt_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    pred_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 정답 박스를 캔버스에 그립니다. (좌표를 캔버스 기준으로 변환)
    gt_x1, gt_y1, gt_x2, gt_y2 = [c - offset for c, offset in zip(gt_box, [x_min, y_min, x_min, y_min])]
    gt_mask[gt_y1:gt_y2, gt_x1:gt_x2] = 1

    # 예측 박스들을 캔버스에 그립니다.
    for box in pred_boxes:
        pred_x1, pred_y1, pred_x2, pred_y2 = [c - offset for c, offset in zip(box, [x_min, y_min, x_min, y_min])]
        pred_mask[pred_y1:pred_y2, pred_x1:pred_x2] = 1

    # 픽셀 단위로 넓이를 계산합니다.
    intersection_area = np.sum(np.logical_and(gt_mask, pred_mask))
    
    # 두 마스크를 합쳐 합집합 영역을 구합니다.
    union_mask = np.logical_or(gt_mask, pred_mask)
    union_area = np.sum(union_mask)

    # IoU 계산 (0으로 나누는 경우 방지)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou
