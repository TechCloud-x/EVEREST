import json
import re
import numpy as np
import torch
import ast
import cv2
from math import exp
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional, Tuple, Union
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.rlvr.visual_primitives import (
    parse_stage1_state,
    parse_stage2_verification,
    merge_verification_with_stage1,
    stage1_state_to_bboxes,
    gt_instances_from_mask,
    count_reward as vp_count_reward,
    bbox_match_reward as vp_bbox_match_reward,
    coverage_reward as vp_coverage_reward,
    duplicate_penalty as vp_duplicate_penalty,
    stage1_format_reward as vp_stage1_format_reward,
    mask_iou as vp_mask_iou,
    instance_recall as vp_instance_recall,
    false_instance_penalty as vp_false_instance_penalty,
    point_validity_reward as vp_point_validity_reward,
    count_consistency_reward as vp_count_consistency_reward,
    stage2_format_reward as vp_stage2_format_reward,
)

def _batch_iou(boxes1, boxes2):
    """Calculates Intersection over Union (IoU) for batches of boxes."""
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)

    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / np.maximum(unionArea, 1e-6)
    return iou

def _batch_l1_distance(boxes1, boxes2):
    """Calculates mean L1 distance for batches of boxes."""
    boxes1 = boxes1[:, np.newaxis, :]
    boxes2 = boxes2[np.newaxis, :, :]
    return np.mean(np.abs(boxes1 - boxes2), axis=2)







_S1_KEYWORDS = ["SCAN", "IDENTIFY", "LOCATE", "VERIFY", "COMPARE", "SELECT"]
_S1_KEYWORD_WEIGHTS = {
    "SCAN": 0.10, "IDENTIFY": 0.15, "LOCATE": 0.20,
    "VERIFY": 0.15, "COMPARE": 0.15, "SELECT": 0.15,
}
_S1_PYTHON_KEYWORDS = ["def", "for", "if", "return"]




_S2_KEYWORDS = ["CHOOSE_DIR", "INSPECT", "COMPUTE_POS", "PROBE", "STOP", "SKIP", "EVIDENCE_SUFFICIENT"]
_S2_KEYWORD_WEIGHTS = {
    "CHOOSE_DIR": 0.15,
    "INSPECT": 0.15,
    "COMPUTE_POS": 0.15,
    "PROBE": 0.20,
    "STOP": 0.15,
    "SKIP": 0.10,
    "EVIDENCE_SUFFICIENT": 0.10,
}
_S2_PYTHON_KEYWORDS = ["def", "for", "if", "return"]


def _keyword_presence_score(text: str, keywords: list, weights: dict) -> float:
    """Compute weighted keyword presence score."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    score = 0.0
    for kw in keywords:
        if re.search(rf'\b{kw}\b', text, re.IGNORECASE):
            score += weights.get(kw, 0.0)
    return score / total_weight


def _keyword_order_score(text: str, keywords: list) -> float:
    """Check if keywords appear in canonical order."""
    positions = {}
    for kw in keywords:
        match = re.search(rf'\b{kw}\b', text, re.IGNORECASE)
        if match:
            positions[kw] = match.start()

    if len(positions) < 3:
        return 0.0

    ordered_kws = sorted(positions.keys(), key=lambda k: positions[k])
    canonical_indices = [keywords.index(k) for k in ordered_kws if k in keywords]
    correct_pairs = sum(
        1 for i in range(len(canonical_indices) - 1)
        if canonical_indices[i] < canonical_indices[i + 1]
    )
    total_pairs = max(len(canonical_indices) - 1, 1)
    return correct_pairs / total_pairs


def _python_structure_score(text: str, python_keywords: list) -> float:
    """Check for Python-style structural keywords."""
    found = sum(1 for kw in python_keywords if re.search(rf'\b{kw}\b', text))
    return found / max(len(python_keywords), 1)


def _pseudocode_s1_reward(think_text: str) -> float:
    """
    Evaluate the quality of Stage-1 pseudocode in the <think> block.
    Returns a score in [0.0, 1.0].

    Components:
      - Domain keyword presence (weighted): 0.50
      - Python structure keywords:          0.20
      - Coordinate consistency (LOCATE):    0.15
      - Keyword ordering:                   0.15
    """
    if not think_text or len(think_text.strip()) < 10:
        return 0.0

    score = 0.0


    score += 0.50 * _keyword_presence_score(think_text, _S1_KEYWORDS, _S1_KEYWORD_WEIGHTS)


    score += 0.20 * _python_structure_score(think_text, _S1_PYTHON_KEYWORDS)


    locate_with_coords = re.findall(
        r'LOCATE\s*\(.*?\)\s*->\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]',
        think_text, re.IGNORECASE
    )
    if locate_with_coords:
        score += 0.15


    score += 0.15 * _keyword_order_score(think_text, _S1_KEYWORDS)

    return min(score, 1.0)


def _pseudocode_s2_reward(think_text: str) -> float:
    """
    Evaluate the quality of Stage-2 pseudocode in the <think> block.
    Returns a score in [0.0, 1.0].

    Components:
      - Domain keyword presence (weighted): 0.50
      - Python structure keywords:          0.20
      - Quality assessment (INSPECT):       0.15
      - Keyword ordering:                   0.15
    """
    if not think_text or len(think_text.strip()) < 10:
        return 0.0

    score = 0.0


    score += 0.50 * _keyword_presence_score(think_text, _S2_KEYWORDS, _S2_KEYWORD_WEIGHTS)


    score += 0.20 * _python_structure_score(think_text, _S2_PYTHON_KEYWORDS)


    compute_pos_match = re.findall(
        r'COMPUTE_POS\s*\(.*?\)\s*->\s*\[\s*\d+\s*,\s*\d+\s*\]',
        think_text, re.IGNORECASE
    )
    if compute_pos_match:
        score += 0.15


    score += 0.15 * _keyword_order_score(think_text, _S2_KEYWORDS)

    return min(score, 1.0)






def _multi_s1_format_reward(predict_str: str) -> float:
    """Calculates the format reward for a single Stage-1 prediction string.

    Returns a value in [0.0, 2.0]:
      - Pseudocode quality score [0, 1.0] (replaces binary think-tag check)
      - Segmentation JSON format score [0, 1.0] (unchanged)
    """
    think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    answer_match = re.search(r'<answer>.*?</answer>', predict_str, re.DOTALL)

    if not think_match or not answer_match:
        thinking_format_reward = 0.0
    else:
        thinking_format_reward = _pseudocode_s1_reward(think_match.group(1))

    segmentation_format_reward = 0.0
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return thinking_format_reward

        data = json.loads(json_match.group(1))
        if not data:
            return thinking_format_reward

        data_cnt = len(data)
        total_cur_reward = 0.0

        for item in data:
            cur_reward = 0.0
            if item.keys() == {'bbox_2d'}:
                bbox_2d = item['bbox_2d']
                if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                    cur_reward += 1.0

            total_cur_reward += cur_reward

        segmentation_format_reward = total_cur_reward / data_cnt
    except Exception:
        pass

    return thinking_format_reward + segmentation_format_reward

def _multi_s2_format_reward(predict_str: str, bbox_text: str) -> float:
    """Stage-2 per-round format reward (multi-round exploration schema).

    Returns value in [0.0, 2.0]:
      - Pseudocode quality (0-1): uses _pseudocode_s2_reward with new keyword set
      - Round JSON schema validity (0-1): checks {"round", "actions": [...], "new_bbox_candidates": [...]}

    The response is round-t's output; bbox_text (Stage-1 output) is kept as arg for backward
    compatibility but only used for light sanity bounding (e.g. bbox_id range).
    """
    think_match = re.search(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)

    if not think_match or not answer_match:
        thinking_format_reward = 0.0
    else:
        thinking_format_reward = _pseudocode_s2_reward(think_match.group(1))

    schema_reward = 0.0
    if answer_match:
        try:
            data = json.loads(answer_match.group(1))
        except Exception:
            data = None
        if isinstance(data, dict):
            has_round = "round" in data
            actions = data.get("actions")
            has_actions = isinstance(actions, list)
            schema_reward = 0.4 if (has_round and has_actions) else (0.2 if has_round or has_actions else 0.0)
            if has_actions:
                try:
                    n_stage1 = len(json.loads(bbox_text.replace("'", '"'))) if bbox_text else 0
                except Exception:
                    n_stage1 = 0
                if len(actions) == 0:
                    schema_reward += 0.4
                else:
                    valid = 0
                    for a in actions:
                        if not isinstance(a, dict) or "bbox_id" not in a or "act" not in a:
                            continue
                        bid = a.get("bbox_id")
                        if not isinstance(bid, int):
                            continue
                        if n_stage1 and bid >= n_stage1:
                            continue
                        act = str(a.get("act", "")).upper()
                        if act in ("STOP", "SKIP"):
                            valid += 1
                        elif act == "PROBE":
                            dir_name = a.get("dir")
                            pos = a.get("pos")
                            if (
                                dir_name
                                and isinstance(pos, list)
                                and len(pos) == 2
                            ):
                                valid += 1
                    schema_reward += 0.4 * (valid / max(len(actions), 1))
            cands = data.get("new_bbox_candidates")
            if isinstance(cands, list):
                schema_reward += 0.2

    return thinking_format_reward + schema_reward

def _multi_s1_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """Calculates the accuracy reward using Hungarian matching."""
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120

    try:
        gt_data = json.loads(ground_truth.replace("'", '"'))
        gt_bboxes = np.array([item['bbox_2d'] for item in gt_data])

        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return 0.0
        pred_data = json.loads(json_match.group(1))
        if not pred_data:
            return 0.0

        pred_bboxes = np.array([item['bbox_2d'] for item in pred_data])


        if len(pred_bboxes) > MAX_OBJECTS:
            pred_bboxes = pred_bboxes[:MAX_OBJECTS]

        if len(gt_bboxes) > MAX_OBJECTS:
            gt_bboxes = gt_bboxes[:MAX_OBJECTS]

        if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            return 0.0


        iou_matrix = _batch_iou(pred_bboxes, gt_bboxes)
        l1_matrix = _batch_l1_distance(pred_bboxes, gt_bboxes)


        iou_reward = (iou_matrix > 0.5).astype(float)
        bbox_l1_reward = (l1_matrix < 10).astype(float)

        cost_matrix = 2.0 - iou_reward - bbox_l1_reward

        row_indices, col_indices = linear_sum_assignment(cost_matrix)


        total_reward = len(row_indices) - cost_matrix[row_indices, col_indices].sum()


        max_length = max(len(pred_bboxes), len(gt_bboxes))
        if max_length == 0: return 0.0

        max_accuracy_reward = total_reward / max_length

    except Exception:
        pass

    return max_accuracy_reward

def _multi_s2_accuracy_reward(mask: np.ndarray, gt_mask: np.ndarray) -> float:

    if not isinstance(mask, np.ndarray) or not isinstance(gt_mask, np.ndarray):
        return 0.0

    if mask.shape != gt_mask.shape:
        return 0.0

    mask = mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(mask, gt_mask).sum()
    union = np.logical_or(mask, gt_mask).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

def _batch_points_distance(points1, points2):
    """Calculates Euclidean distance for batches of points."""
    points1 = points1[:, np.newaxis, :]
    points2 = points2[np.newaxis, :, :]
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))
    return dist


def _multi_s1_length_reward(predict_str: str, ground_truth: str) -> float:
    try:
        gt_data = json.loads(ground_truth.replace("'", '"'))
        gt_bboxes = np.array([item['bbox_2d'] for item in gt_data])
        gt_length = len(gt_bboxes)

        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return 0.0

        pred_data = json.loads(json_match.group(1))
        pred_bboxes = np.array([item['bbox_2d'] for item in pred_data])
        pred_length = len(pred_bboxes)

        J = gt_length
        K = pred_length

        if J == 0 and K == 0:
            return 1.0
        elif J == 0 and K > 0:
            return 0.0
        else:
            return np.exp(-2 * abs(K - J) / J)

    except (json.JSONDecodeError, re.error, ValueError, IndexError, TypeError, SyntaxError, KeyError) as e:
        return 0.0

def _multi_s2_length_reward(text: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.MULTILINE)
    if not match:
        return 0
    reward = 0
    answer_content = match.group(1).strip()
    try:
        parsed_answer = json.loads(answer_content)
        for group in parsed_answer:
            if 'points' not in group:
                continue
            length = len(group['points'])
            ideal = 2
            sigma = 2
            reward += exp(-((length - ideal) ** 2) / (2 * sigma ** 2))

        reward = reward / len(parsed_answer) if parsed_answer else 0
        return reward
    except Exception:
        return 0






_FLOW_SQRT_HALF = 0.5 ** 0.5
_FLOW_DIR_UNIT: Dict[str, Tuple[str, Tuple[float, float]]] = {
    "TL_ORTH": ("TL", (-1.0, 0.0)),
    "TL_DIAG": ("TL", (-_FLOW_SQRT_HALF, -_FLOW_SQRT_HALF)),
    "TR_ORTH": ("TR", (0.0, -1.0)),
    "TR_DIAG": ("TR", (_FLOW_SQRT_HALF, -_FLOW_SQRT_HALF)),
    "BR_ORTH": ("BR", (1.0, 0.0)),
    "BR_DIAG": ("BR", (_FLOW_SQRT_HALF, _FLOW_SQRT_HALF)),
    "BL_ORTH": ("BL", (0.0, 1.0)),
    "BL_DIAG": ("BL", (-_FLOW_SQRT_HALF, _FLOW_SQRT_HALF)),
}


def _flow_corner_xy(bbox: List[int], corner: str) -> Tuple[int, int]:
    bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
    if corner == "TL":
        return bx1, by1
    if corner == "TR":
        return bx2, by1
    if corner == "BR":
        return bx2, by2
    if corner == "BL":
        return bx1, by2
    return bx1, by1


def _expected_probe_pos(bbox: List[int], dir_name: str, step: int) -> Tuple[int, int]:
    """Canonical (x, y) given (bbox, dir_name, step). Mirrors pipeline.compute_probe_pos."""
    corner, (dx, dy) = _FLOW_DIR_UNIT[dir_name]
    ox, oy = _flow_corner_xy(bbox, corner)
    bx1, by1, bx2, by2 = bbox
    short_side = max(1, min(bx2 - bx1, by2 - by1))
    step_size = max(8.0, short_side * 0.15)
    px = int(round(ox + step * step_size * dx))
    py = int(round(oy + step * step_size * dy))
    return px, py


def _gt_cc_bboxes(gt_mask_array: np.ndarray) -> List[List[int]]:
    """Return bbox [x1, y1, x2, y2] for each connected component (area > 10)."""
    if gt_mask_array is None:
        return []
    bin_mask = (np.asarray(gt_mask_array) > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area <= 10:
            continue
        out.append([int(x), int(y), int(x + w), int(y + h)])
    return out


def _bbox_iou_scalar(b1: List[int], b2: List[int]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = max(0, x12 - x11) * max(0, y12 - y11)
    a2 = max(0, x22 - x21) * max(0, y22 - y21)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _flow_probe_reward(flow_json: Dict[str, Any], gt_mask_array: np.ndarray) -> float:
    """R1: per-PROBE pointwise reward (queries gt_mask at pos).

      pos ∈ gt ∧ relevant=True  → +1.0 (correct discovery)
      pos ∉ gt ∧ relevant=True  → -1.0 (hallucination)
      pos ∉ gt ∧ relevant=False → +0.5 (correct verification of boundary)
      pos ∈ gt ∧ relevant=False → -0.5 (false negative)
      geometry mismatch (pos vs canonical 8-direction formula, tol 4px) → -0.5
    """
    if not isinstance(flow_json, dict) or gt_mask_array is None:
        return 0.0
    gt_bin = (np.asarray(gt_mask_array) > 0)
    H, W = gt_bin.shape[:2]
    total = 0.0
    for b in flow_json.get("bboxes", []) or []:
        bbox = b.get("bbox_2d")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        for s in b.get("steps", []) or []:
            dir_name = s.get("dir")
            pos = s.get("pos")
            if dir_name not in _FLOW_DIR_UNIT or not (isinstance(pos, list) and len(pos) == 2):
                total += -0.5
                continue
            try:
                exp_x, exp_y = _expected_probe_pos(bbox, dir_name, int(s.get("step", 1)))
                act_x, act_y = int(pos[0]), int(pos[1])
            except Exception:
                total += -0.5
                continue
            if abs(act_x - exp_x) > 4 or abs(act_y - exp_y) > 4:
                total += -0.5
                continue
            cx = max(0, min(W - 1, act_x))
            cy = max(0, min(H - 1, act_y))
            hit = bool(gt_bin[cy, cx])
            relevant = bool(s.get("relevant", False))
            if relevant and hit:
                total += 1.0
            elif relevant and not hit:
                total -= 1.0
            elif (not relevant) and (not hit):
                total += 0.5
            else:
                total -= 0.5
    return float(total)


def _flow_efficiency_penalty(flow_json: Dict[str, Any]) -> float:
    """R2: -0.1 per PROBE action taken across all rounds and bboxes."""
    if not isinstance(flow_json, dict):
        return 0.0
    total_probes = 0
    for b in flow_json.get("bboxes", []) or []:
        total_probes += len(b.get("steps", []) or [])
    return float(-0.1 * total_probes)


def _flow_correction_reward(flow_json: Dict[str, Any], gt_mask_array: np.ndarray) -> float:
    """R3: +1.0 per new_bbox_candidate with IoU ≥ 0.3 against some gt connected component;
    -0.5 per hallucinated candidate."""
    if not isinstance(flow_json, dict):
        return 0.0
    cands = flow_json.get("corrections", {}).get("new_bboxes", []) or []
    if not cands:
        return 0.0
    gt_ccs = _gt_cc_bboxes(gt_mask_array)
    if not gt_ccs:
        return float(-0.5 * sum(1 for c in cands if isinstance(c, (list, tuple)) and len(c) == 4))
    total = 0.0
    for nb in cands:
        if not (isinstance(nb, (list, tuple)) and len(nb) == 4):
            continue
        best = max((_bbox_iou_scalar(list(nb), cc) for cc in gt_ccs), default=0.0)
        total += 1.0 if best >= 0.3 else -0.5
    return float(total)


def _flow_anti_inertia_penalty(flow_json: Dict[str, Any], gt_mask_array: np.ndarray) -> float:
    """R4: For each stage-1 bbox whose IoU vs all gt components < 0.3 and who was STOPped
    (no SKIP) without a covering correction bbox → -1.0."""
    if not isinstance(flow_json, dict):
        return 0.0
    gt_ccs = _gt_cc_bboxes(gt_mask_array)
    if not gt_ccs:
        return 0.0
    penalty = 0.0
    new_bbs = flow_json.get("corrections", {}).get("new_bboxes", []) or []
    for b in flow_json.get("bboxes", []) or []:
        bbox = b.get("bbox_2d")
        status = b.get("status")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if status != "stopped":
            continue
        best_iou = max((_bbox_iou_scalar(bbox, cc) for cc in gt_ccs), default=0.0)
        if best_iou < 0.3:
            rescue = False
            for nb in new_bbs:
                if isinstance(nb, (list, tuple)) and len(nb) == 4 and _bbox_iou_scalar(bbox, list(nb)) > 0.1:
                    rescue = True
                    break
            if not rescue:
                penalty -= 1.0
    return float(penalty)


def _flow_telemetry(flow_json: Dict[str, Any], gt_mask_array: np.ndarray) -> Dict[str, float]:
    """Exploration Efficiency / Discovery Precision / Correction Recall metrics."""
    if not isinstance(flow_json, dict):
        return {"exploration_efficiency": 0.0, "discovery_precision": 0.0, "correction_recall": 0.0}

    steps_counts = [len(b.get("steps", []) or []) for b in flow_json.get("bboxes", []) or []]
    efficiency = float(np.mean(steps_counts)) if steps_counts else 0.0


    tp, fp = 0, 0
    if gt_mask_array is not None:
        gt_bin = (np.asarray(gt_mask_array) > 0)
        H, W = gt_bin.shape[:2]
        for b in flow_json.get("bboxes", []) or []:
            for s in b.get("steps", []) or []:
                if not bool(s.get("relevant", False)):
                    continue
                pos = s.get("pos")
                if not (isinstance(pos, list) and len(pos) == 2):
                    continue
                try:
                    x = max(0, min(W - 1, int(pos[0])))
                    y = max(0, min(H - 1, int(pos[1])))
                except Exception:
                    continue
                if gt_bin[y, x]:
                    tp += 1
                else:
                    fp += 1
    discovery_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0


    correction_recall = 0.0
    if gt_mask_array is not None:
        gt_ccs = _gt_cc_bboxes(gt_mask_array)
        s1_bboxes = [b.get("bbox_2d") for b in flow_json.get("bboxes", []) or [] if isinstance(b.get("bbox_2d"), list)]
        missed = []
        for cc in gt_ccs:
            covered = any(_bbox_iou_scalar(list(sb), cc) >= 0.3 for sb in s1_bboxes if isinstance(sb, list) and len(sb) == 4)
            if not covered:
                missed.append(cc)
        if missed:
            new_bbs = flow_json.get("corrections", {}).get("new_bboxes", []) or []
            recovered = 0
            for m in missed:
                for nb in new_bbs:
                    if isinstance(nb, (list, tuple)) and len(nb) == 4 and _bbox_iou_scalar(list(nb), m) >= 0.3:
                        recovered += 1
                        break
            correction_recall = recovered / len(missed)

    return {
        "exploration_efficiency": float(efficiency),
        "discovery_precision": float(discovery_precision),
        "correction_recall": float(correction_recall),
    }


class SocioSegRuleRewardWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.format_pattern = self.worker_config.format_pattern

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards_split(self, data: DataProto):
        """
        Computes rewards for Multi-Instance Visual Primitive Reasoning (MIVPR).

        Stage-1 (map_responses, VisualPrimitiveState enumeration):
            format + count + bbox_match + coverage - duplicate
        Stage-2 (sat_responses, per-instance keep/adjust/drop verification):
            format + mask_iou + instance_recall - false_instance
                   + point_validity + count_consistency

        Returns a DataProto with reward tensors and metrics for both stages.
        ``seg_iou_rewards`` stays the final-mask IoU (used for advantage & eval).
        """

        map_response_text_list = self.tokenizer.batch_decode(data.batch["map_responses"], skip_special_tokens=False)
        sat_response_text_list = self.tokenizer.batch_decode(data.batch["sat_responses"], skip_special_tokens=False)

        map_pred_mask_list = data.non_tensor_batch['map_mask']
        sat_pred_mask_list = data.non_tensor_batch['sat_mask']
        gt_mask_list = data.non_tensor_batch["gt_mask"]

        def _strip(r: str) -> str:
            return r.replace("<|endoftext|>", "").replace("<|im_end|>", "").replace("<pad>", "")


        map_format_rewards = []
        map_count_rewards = []
        map_bbox_match_rewards = []
        map_coverage_rewards = []
        map_duplicate_penalties = []
        map_seg_iou_accuracies = []


        sat_format_rewards = []
        sat_mask_iou_rewards = []
        sat_instance_recall_rewards = []
        sat_false_instance_penalties = []
        sat_point_validity_rewards = []
        sat_count_consistency_rewards = []

        pred_count_errors = []
        duplicate_rate_samples = []

        for map_resp, sat_resp, map_pred_mask, sat_pred_mask, gt_mask in zip(
            map_response_text_list, sat_response_text_list,
            map_pred_mask_list, sat_pred_mask_list, gt_mask_list,
        ):
            map_resp = _strip(map_resp)
            sat_resp = _strip(sat_resp)

            gt_mask_np = np.array(gt_mask.convert("L"))
            gt_bboxes = gt_instances_from_mask(gt_mask_np)
            gt_count = len(gt_bboxes)


            s1_state = parse_stage1_state(map_resp)
            pred_bboxes = stage1_state_to_bboxes(s1_state)
            declared_count = s1_state["declared_count"] if s1_state else 0

            map_format_rewards.append(vp_stage1_format_reward(map_resp))
            map_count_rewards.append(vp_count_reward(declared_count, gt_count))
            map_bbox_match_rewards.append(vp_bbox_match_reward(pred_bboxes, gt_bboxes))
            map_coverage_rewards.append(vp_coverage_reward(pred_bboxes, gt_bboxes))
            map_duplicate_penalties.append(vp_duplicate_penalty(pred_bboxes))
            map_seg_iou_accuracies.append(vp_mask_iou(map_pred_mask, gt_mask_np))


            verif = parse_stage2_verification(sat_resp)
            merged = merge_verification_with_stage1(s1_state, verif)
            kept_count = len(merged)
            verified_count = verif.get("verified_count", kept_count) if isinstance(verif, dict) else kept_count

            sat_format_rewards.append(vp_stage2_format_reward(sat_resp))
            miou = vp_mask_iou(sat_pred_mask, gt_mask_np)
            sat_mask_iou_rewards.append(miou)
            sat_instance_recall_rewards.append(vp_instance_recall(sat_pred_mask, gt_bboxes))
            sat_false_instance_penalties.append(vp_false_instance_penalty(kept_count, gt_count))
            sat_point_validity_rewards.append(vp_point_validity_reward(verif, gt_mask_np))
            sat_count_consistency_rewards.append(vp_count_consistency_reward(verified_count, kept_count, gt_count))

            pred_count_errors.append(abs(kept_count - gt_count))
            duplicate_rate_samples.append(-vp_duplicate_penalty(pred_bboxes))


        sat_format_rewards = torch.tensor(sat_format_rewards, dtype=torch.float16)
        sat_mask_iou_rewards = torch.tensor(sat_mask_iou_rewards, dtype=torch.float16)
        sat_instance_recall_rewards = torch.tensor(sat_instance_recall_rewards, dtype=torch.float16)
        sat_false_instance_penalties = torch.tensor(sat_false_instance_penalties, dtype=torch.float16)
        sat_point_validity_rewards = torch.tensor(sat_point_validity_rewards, dtype=torch.float16)
        sat_count_consistency_rewards = torch.tensor(sat_count_consistency_rewards, dtype=torch.float16)
        sat_sum_rewards = (
            sat_format_rewards
            + sat_mask_iou_rewards
            + sat_instance_recall_rewards
            + sat_false_instance_penalties
            + sat_point_validity_rewards
            + sat_count_consistency_rewards
        )


        map_format_rewards = torch.tensor(map_format_rewards, dtype=torch.float16)
        map_count_rewards = torch.tensor(map_count_rewards, dtype=torch.float16)
        map_bbox_match_rewards = torch.tensor(map_bbox_match_rewards, dtype=torch.float16)
        map_coverage_rewards = torch.tensor(map_coverage_rewards, dtype=torch.float16)
        map_duplicate_penalties = torch.tensor(map_duplicate_penalties, dtype=torch.float16)
        map_seg_iou_accuracies = torch.tensor(map_seg_iou_accuracies, dtype=torch.float16)
        map_sum_rewards = (
            map_format_rewards
            + map_count_rewards
            + map_bbox_match_rewards
            + map_coverage_rewards
            + map_duplicate_penalties
        )

        metrics = {

            "sat_format_reward_mean": sat_format_rewards.mean().item(),
            "sat_mask_iou_reward_mean": sat_mask_iou_rewards.mean().item(),
            "sat_instance_recall_mean": sat_instance_recall_rewards.mean().item(),
            "sat_false_instance_penalty_mean": sat_false_instance_penalties.mean().item(),
            "sat_point_validity_mean": sat_point_validity_rewards.mean().item(),
            "sat_count_consistency_mean": sat_count_consistency_rewards.mean().item(),
            "sat_seg_iou_accuracy_mean": sat_mask_iou_rewards.mean().item(),

            "map_format_reward_mean": map_format_rewards.mean().item(),
            "map_count_reward_mean": map_count_rewards.mean().item(),
            "map_bbox_match_reward_mean": map_bbox_match_rewards.mean().item(),
            "map_coverage_reward_mean": map_coverage_rewards.mean().item(),
            "map_duplicate_penalty_mean": map_duplicate_penalties.mean().item(),
            "map_seg_iou_accuracy_mean": map_seg_iou_accuracies.mean().item(),

            "pred_count_error_mean": float(np.mean(pred_count_errors)) if pred_count_errors else 0.0,
            "duplicate_rate_mean": float(np.mean(duplicate_rate_samples)) if duplicate_rate_samples else 0.0,
        }

        output = DataProto.from_dict(
            tensors={
                "seg_iou_rewards": sat_mask_iou_rewards,
                "sat_response_level_rewards": sat_sum_rewards,
                "map_response_level_rewards": map_sum_rewards,
            },
            meta_info={"metrics": metrics}
        )

        return output
