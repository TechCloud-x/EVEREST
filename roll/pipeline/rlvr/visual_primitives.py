"""
Multi-Instance Visual Primitive Reasoning (MIVPR).

This module implements the core, side-effect-free logic for the
"Multi-Instance Visual Primitive Reasoning" innovation, which replaces the
legacy Stage-2 corner-probe ``flow_json`` exploration.

Motivation (the *Reference Gap*): in multi-instance segmentation the model loses
track of *which* instance it is reasoning about, because bounding boxes are
emitted as an anonymous list with no stable identity. We instead make the model
maintain bbox/point visual primitives as *instance-level reference anchors*:

  Stage-1 (full-image enumeration) emits a ``VisualPrimitiveState``:
      {
        "target": "...",
        "declared_count": 3,
        "instances": [
          {"id": 0, "bbox_2d": [x1,y1,x2,y2], "center_point": [cx,cy],
           "evidence": "map_label|sat_shape|both", "confidence": "high|medium|low"},
          ...
        ]
      }

  Stage-2 (single-round per-instance verification) emits a verification result:
      {
        "verified_count": 3,
        "instances": [
          {"id": 0, "action": "keep|adjust|drop",
           "bbox_2d": [x1,y1,x2,y2], "positive_points": [[x,y], ...]},
          ...
        ]
      }

All coordinates are pixel coordinates ``[x1, y1, x2, y2]`` consistent with the
existing SAM2 input (no 0-999 normalized coordinates).

Everything here is pure Python + numpy (cv2 optional). It is unit-testable
offline without ray / SAM2 / GPU.
"""

from __future__ import annotations

import json
import re
from math import exp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np





S1_KEYWORDS = ["SCAN_GRID", "ANCHOR_WITH_BOX", "DEDUPLICATE", "COUNT_CONFIRM"]
S1_KEYWORD_WEIGHTS = {
    "SCAN_GRID": 0.25,
    "ANCHOR_WITH_BOX": 0.30,
    "DEDUPLICATE": 0.25,
    "COUNT_CONFIRM": 0.20,
}

S2_KEYWORDS = ["INSPECT", "DECIDE", "PLACE_POINT", "VERIFY_COUNT"]
S2_KEYWORD_WEIGHTS = {
    "INSPECT": 0.30,
    "DECIDE": 0.30,
    "PLACE_POINT": 0.20,
    "VERIFY_COUNT": 0.20,
}
PYTHON_KEYWORDS = ["def", "for", "if", "return"]

VALID_ACTIONS = ("keep", "adjust", "drop")
_GT_MIN_AREA = 10





def extract_answer_block(text: str) -> Optional[str]:
    """Return the raw string inside the first <answer>...</answer> block, or None."""
    if not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def extract_think_block(text: str) -> Optional[str]:
    """Return the raw string inside the first <think>...</think> block, or None."""
    if not text:
        return None
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not m:
        return None
    return m.group(1)


def _safe_json_loads(s: str) -> Optional[Any]:
    if s is None:
        return None
    try:
        return json.loads(s)
    except Exception:

        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return None


def _as_int_bbox(bbox: Any) -> Optional[List[int]]:
    """Validate + coerce a 4-number bbox to ints. Returns None if invalid."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    out = []
    for v in bbox:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        out.append(int(round(v)))
    x1, y1, x2, y2 = out

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _as_int_point(pt: Any) -> Optional[List[int]]:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return None
    out = []
    for v in pt:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        out.append(int(round(v)))
    return out


def _bbox_center(bbox: List[int]) -> List[int]:
    return [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]





def parse_stage1_state(text: str) -> Optional[Dict[str, Any]]:
    """Parse the Stage-1 <answer> block into a normalized VisualPrimitiveState.

    Returns None if the answer is missing or yields zero valid instances.
    Each returned instance is guaranteed to have an integer ``id`` and a valid
    integer ``bbox_2d``; ``center_point`` is filled from the bbox when missing.
    Malformed individual instances are skipped (graceful degradation), so a
    response with some good and some bad instances still yields a usable state.
    """
    raw = extract_answer_block(text)
    data = _safe_json_loads(raw)
    if data is None:
        return None


    if isinstance(data, dict):
        instances_raw = data.get("instances", [])
        target = data.get("target", "")
        declared_count = data.get("declared_count", None)
    elif isinstance(data, list):
        instances_raw = data
        target = ""
        declared_count = None
    else:
        return None

    if not isinstance(instances_raw, list):
        return None

    instances: List[Dict[str, Any]] = []
    for i, obj in enumerate(instances_raw):
        if not isinstance(obj, dict):
            continue

        bbox = _as_int_bbox(obj.get("bbox_2d", obj.get("bbox")))
        if bbox is None:
            continue
        cp = _as_int_point(obj.get("center_point")) or _bbox_center(bbox)
        iid = obj.get("id", i)
        if not isinstance(iid, int) or isinstance(iid, bool):
            iid = i
        instances.append({
            "id": iid,
            "bbox_2d": bbox,
            "center_point": cp,
            "evidence": obj.get("evidence", ""),
            "confidence": obj.get("confidence", ""),
        })

    if not instances:
        return None

    if not isinstance(declared_count, int) or isinstance(declared_count, bool):
        declared_count = len(instances)

    return {
        "target": target if isinstance(target, str) else "",
        "declared_count": declared_count,
        "instances": instances,
    }


def stage1_state_to_bboxes(state: Optional[Dict[str, Any]]) -> List[List[int]]:
    """Extract ordered list of bbox_2d from a parsed Stage-1 state."""
    if not isinstance(state, dict):
        return []
    return [inst["bbox_2d"] for inst in state.get("instances", []) if "bbox_2d" in inst]


def stage1_state_to_bboxs_text(state: Optional[Dict[str, Any]]) -> str:
    """Serialize a Stage-1 state's bboxes into the legacy bboxs_text JSON.

    Compatible with render_image() and reward worker which expect a JSON list of
    {"bbox_2d": [...]} items.
    """
    items = [{"bbox_2d": list(b)} for b in stage1_state_to_bboxes(state)]
    return json.dumps(items)


def stage1_state_to_render_items(state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Items for rendering: bbox + id label + center point."""
    if not isinstance(state, dict):
        return []
    out = []
    for inst in state.get("instances", []):
        out.append({
            "id": inst.get("id"),
            "bbox_2d": inst.get("bbox_2d"),
            "center_point": inst.get("center_point"),
        })
    return out





def parse_stage2_verification(text: str) -> Optional[Dict[str, Any]]:
    """Parse the Stage-2 <answer> block into a normalized verification dict.

    Returns None if missing/unparseable. Individual malformed instances are
    skipped. An instance with an invalid action defaults to "keep" (conservative:
    we would rather keep an instance than silently drop a real one).
    """
    raw = extract_answer_block(text)
    data = _safe_json_loads(raw)
    if data is None:
        return None

    if isinstance(data, dict):
        instances_raw = data.get("instances", [])
        verified_count = data.get("verified_count", None)
    elif isinstance(data, list):
        instances_raw = data
        verified_count = None
    else:
        return None

    if not isinstance(instances_raw, list):
        return None

    instances: List[Dict[str, Any]] = []
    for i, obj in enumerate(instances_raw):
        if not isinstance(obj, dict):
            continue
        action = obj.get("action", "keep")
        if not isinstance(action, str) or action.lower() not in VALID_ACTIONS:
            action = "keep"
        action = action.lower()
        bbox = _as_int_bbox(obj.get("bbox_2d", obj.get("bbox")))
        iid = obj.get("id", i)
        if not isinstance(iid, int) or isinstance(iid, bool):
            iid = i
        pos_pts = []
        for p in obj.get("positive_points", []) or []:
            ip = _as_int_point(p)
            if ip is not None:
                pos_pts.append(ip)
        instances.append({
            "id": iid,
            "action": action,
            "bbox_2d": bbox,
            "positive_points": pos_pts,
        })

    if not instances:
        return None

    if not isinstance(verified_count, int) or isinstance(verified_count, bool):
        verified_count = sum(1 for x in instances if x["action"] != "drop")

    return {"verified_count": verified_count, "instances": instances}


def merge_verification_with_stage1(
    s1_state: Optional[Dict[str, Any]],
    s2_verif: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fuse Stage-2 verification onto Stage-1 instances to produce the final
    kept instance list for segmentation.

    Rules:
      - Match by ``id``. If Stage-2 omits an id present in Stage-1, that instance
        is kept with its Stage-1 bbox (no points).
      - action == "drop"  -> instance removed.
      - action == "adjust"-> use Stage-2 bbox if valid, else Stage-1 bbox.
      - action == "keep"  -> use Stage-1 bbox (or Stage-2 bbox if S1 missing).
      - positive_points carried through.

    Returns a list of {"bbox_2d": [...], "positive_points": [[x,y],...], "id": id}.
    If Stage-1 is missing, falls back to Stage-2 instances alone.
    """
    s1_by_id: Dict[Any, Dict[str, Any]] = {}
    if isinstance(s1_state, dict):
        for inst in s1_state.get("instances", []):
            s1_by_id[inst.get("id")] = inst

    out: List[Dict[str, Any]] = []

    if isinstance(s2_verif, dict):
        seen_ids = set()
        for v in s2_verif.get("instances", []):
            vid = v.get("id")
            seen_ids.add(vid)
            if v.get("action") == "drop":
                continue
            s1_inst = s1_by_id.get(vid)
            bbox = None
            if v.get("action") == "adjust":
                bbox = v.get("bbox_2d") or (s1_inst.get("bbox_2d") if s1_inst else None)
            else:
                bbox = (s1_inst.get("bbox_2d") if s1_inst else None) or v.get("bbox_2d")
            if bbox is None:
                continue
            out.append({
                "id": vid,
                "bbox_2d": list(bbox),
                "positive_points": [list(p) for p in v.get("positive_points", [])],
            })

        for iid, inst in s1_by_id.items():
            if iid in seen_ids:
                continue
            out.append({
                "id": iid,
                "bbox_2d": list(inst["bbox_2d"]),
                "positive_points": [],
            })
        return out


    for iid, inst in s1_by_id.items():
        out.append({
            "id": iid,
            "bbox_2d": list(inst["bbox_2d"]),
            "positive_points": [],
        })
    return out


def instances_to_sam_prompts(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert merged kept-instance dicts into SAM2 prompt dicts.

    Each output element: {"box": ndarray(4,),
                          "point_coords": ndarray(K,2)?,
                          "point_labels": ndarray(K,)?}.
    All positive_points get label 1 (v1 has no negative points).
    """
    out: List[Dict[str, Any]] = []
    for inst in instances:
        bbox = inst.get("bbox_2d")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        prompt: Dict[str, Any] = {"box": np.array(bbox)}
        pts = [p for p in inst.get("positive_points", []) if isinstance(p, (list, tuple)) and len(p) == 2]
        if pts:
            prompt["point_coords"] = np.array(pts)
            prompt["point_labels"] = np.ones(len(pts), dtype=int)
        out.append(prompt)
    return out





def _connected_component_stats(bin_mask: np.ndarray):
    """Return (num_labels, labels, stats) where stats[i] = [x, y, w, h, area].

    Prefers cv2 (production), falls back to scipy.ndimage (offline tests).
    Label 0 is background in both paths.
    """
    try:
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask.astype(np.uint8), connectivity=8
        )
        return num_labels, labels, stats
    except Exception:
        from scipy import ndimage
        structure = np.ones((3, 3), dtype=int)
        labels, n = ndimage.label(bin_mask, structure=structure)

        stats = np.zeros((n + 1, 5), dtype=int)
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if xs.size == 0:
                continue
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
            stats[i] = [x, y, w, h, int(xs.size)]
        return n + 1, labels, stats


def gt_instances_from_mask(mask_array: np.ndarray, min_area: int = _GT_MIN_AREA) -> List[List[int]]:
    """Return one bbox [x1,y1,x2,y2] per connected component with area > min_area."""
    if mask_array is None:
        return []
    arr = np.asarray(mask_array)
    if arr.ndim > 2:
        arr = arr[..., 0]
    bin_mask = (arr > 0)
    if not bin_mask.any():
        return []
    num_labels, _, stats = _connected_component_stats(bin_mask)
    out: List[List[int]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area <= min_area:
            continue
        out.append([int(x), int(y), int(x + w), int(y + h)])
    return out


def gt_count_from_mask(mask_array: np.ndarray, min_area: int = _GT_MIN_AREA) -> int:
    return len(gt_instances_from_mask(mask_array, min_area=min_area))





def bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_iou_matrix(preds: List[List[int]], gts: List[List[int]]) -> np.ndarray:
    """(P, G) IoU matrix."""
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)), dtype=float)
    M = np.zeros((len(preds), len(gts)), dtype=float)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            M[i, j] = bbox_iou(p, g)
    return M


def _center(b: List[int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)





def count_reward(declared_count: int, gt_count: int) -> float:
    """Soft count-matching reward in [0, 1]: exp(-|pred-gt| / max(gt,1))."""
    if gt_count <= 0:
        return 1.0 if declared_count == 0 else 0.0
    return float(exp(-abs(int(declared_count) - int(gt_count)) / max(gt_count, 1)))


def bbox_match_reward(pred_bboxes: List[List[int]], gt_bboxes: List[List[int]],
                      iou_weight: float = 0.7, center_weight: float = 0.3) -> float:
    """Hungarian-matched mean quality of pred vs gt in [0, 1].

    Per matched pair: iou_weight * IoU + center_weight * center_closeness,
    where center_closeness = exp(-dist / diag). Normalized by max(P, G) so
    over/under-prediction is penalized.
    """
    from scipy.optimize import linear_sum_assignment
    if not pred_bboxes or not gt_bboxes:
        return 0.0
    P, G = len(pred_bboxes), len(gt_bboxes)
    iou = bbox_iou_matrix(pred_bboxes, gt_bboxes)
    quality = np.zeros((P, G), dtype=float)
    for i, p in enumerate(pred_bboxes):
        pcx, pcy = _center(p)
        for j, g in enumerate(gt_bboxes):
            gcx, gcy = _center(g)
            diag = max(1.0, ((g[2] - g[0]) ** 2 + (g[3] - g[1]) ** 2) ** 0.5)
            dist = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5
            closeness = exp(-dist / diag)
            quality[i, j] = iou_weight * iou[i, j] + center_weight * closeness
    row, col = linear_sum_assignment(-quality)
    matched = quality[row, col].sum()
    return float(matched / max(P, G))


def coverage_reward(pred_bboxes: List[List[int]], gt_bboxes: List[List[int]],
                    iou_thr: float = 0.3) -> float:
    """Fraction of gt instances covered by at least one pred bbox (IoU >= thr)."""
    if not gt_bboxes:
        return 1.0 if not pred_bboxes else 0.0
    if not pred_bboxes:
        return 0.0
    M = bbox_iou_matrix(pred_bboxes, gt_bboxes)
    covered = (M.max(axis=0) >= iou_thr).sum()
    return float(covered / len(gt_bboxes))


def duplicate_penalty(pred_bboxes: List[List[int]], iou_thr: float = 0.7) -> float:
    """Penalty in [-1, 0] for mutually overlapping predictions (same instance
    predicted multiple times). Penalty = -overlapping_pairs / total_pairs."""
    n = len(pred_bboxes)
    if n < 2:
        return 0.0
    dup_pairs = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_iou(pred_bboxes[i], pred_bboxes[j]) >= iou_thr:
                dup_pairs += 1
    return float(-dup_pairs / total_pairs)


def _keyword_presence_score(text: str, keywords: List[str], weights: Dict[str, float]) -> float:
    total = sum(weights.values())
    if total == 0 or not text:
        return 0.0
    s = 0.0
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            s += weights.get(kw, 0.0)
    return s / total


def _python_structure_score(text: str) -> float:
    if not text:
        return 0.0
    found = sum(1 for kw in PYTHON_KEYWORDS if re.search(rf"\b{kw}\b", text))
    return found / len(PYTHON_KEYWORDS)


def stage1_format_reward(text: str) -> float:
    """Stage-1 format reward in [0, 2].

    Component A (pseudocode quality, 0-1):
      0.6 * keyword presence + 0.2 * python structure + 0.2 * has <think>/<answer>.
    Component B (schema validity, 0-1), with graceful degradation:
      0.3 if any valid instance parsed,
      + up to 0.4 for fraction of instances carrying a valid bbox,
      + 0.3 if declared_count present and equals number of instances.
    """
    think = extract_think_block(text)
    answer = extract_answer_block(text)

    quality = 0.0
    if think:
        quality += 0.6 * _keyword_presence_score(think, S1_KEYWORDS, S1_KEYWORD_WEIGHTS)
        quality += 0.2 * _python_structure_score(think)
    if think and answer:
        quality += 0.2

    schema = 0.0
    data = _safe_json_loads(answer)
    if isinstance(data, (dict, list)):
        if isinstance(data, dict):
            instances_raw = data.get("instances", [])
            declared = data.get("declared_count", None)
        else:
            instances_raw = data
            declared = None
        if isinstance(instances_raw, list) and instances_raw:
            valid_bbox = 0
            for obj in instances_raw:
                if isinstance(obj, dict) and _as_int_bbox(obj.get("bbox_2d", obj.get("bbox"))) is not None:
                    valid_bbox += 1
            if valid_bbox > 0:
                schema += 0.3
                schema += 0.4 * (valid_bbox / len(instances_raw))
                if isinstance(declared, int) and not isinstance(declared, bool) and declared == len(instances_raw):
                    schema += 0.3
    return float(min(quality, 1.0) + min(schema, 1.0))





def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if not isinstance(pred_mask, np.ndarray) or not isinstance(gt_mask, np.ndarray):
        return 0.0
    if pred_mask.shape != gt_mask.shape:
        return 0.0
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def instance_recall(pred_mask: np.ndarray, gt_bboxes: List[List[int]],
                    cover_thr: float = 0.5) -> float:
    """Fraction of gt connected-component bboxes whose region is sufficiently
    covered by the predicted mask (pred pixels inside bbox / bbox area >= thr)."""
    if not gt_bboxes:
        return 1.0
    if not isinstance(pred_mask, np.ndarray):
        return 0.0
    p = pred_mask.astype(bool)
    H, W = p.shape[:2]
    covered = 0
    for (x1, y1, x2, y2) in gt_bboxes:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        region = p[y1c:y2c, x1c:x2c]
        area = (x2c - x1c) * (y2c - y1c)
        if area > 0 and region.sum() / area >= cover_thr:
            covered += 1
    return float(covered / len(gt_bboxes))


def false_instance_penalty(kept_count: int, gt_count: int) -> float:
    """Penalty in [-1, 0] when more instances are kept than gt has.
    Penalty = -(kept-gt)/kept (only over-prediction is penalized here;
    under-prediction is handled by coverage/recall rewards)."""
    if kept_count <= 0 or kept_count <= gt_count:
        return 0.0
    return float(-(kept_count - gt_count) / kept_count)


def point_validity_reward(verif: Optional[Dict[str, Any]], gt_mask: np.ndarray) -> float:
    """Mean over all positive points of (1 if inside gt mask else 0), in [0, 1].
    Returns 0.0 if there are no positive points (no signal)."""
    if not isinstance(verif, dict) or not isinstance(gt_mask, np.ndarray):
        return 0.0
    g = gt_mask.astype(bool)
    H, W = g.shape[:2]
    hits, total = 0, 0
    for inst in verif.get("instances", []):
        if inst.get("action") == "drop":
            continue
        for p in inst.get("positive_points", []) or []:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                continue
            x = max(0, min(W - 1, int(p[0])))
            y = max(0, min(H - 1, int(p[1])))
            total += 1
            if g[y, x]:
                hits += 1
    if total == 0:
        return 0.0
    return float(hits / total)


def count_consistency_reward(verified_count: int, kept_count: int, gt_count: int) -> float:
    """Reward in [0, 1] that verified_count is both internally consistent
    (matches actual kept_count) and externally correct (matches gt_count).
      0.5 * (verified_count == kept_count) + 0.5 * soft(kept_count, gt_count)."""
    internal = 1.0 if int(verified_count) == int(kept_count) else 0.0
    external = count_reward(kept_count, gt_count)
    return float(0.5 * internal + 0.5 * external)


def stage2_format_reward(text: str) -> float:
    """Stage-2 format reward in [0, 2].

    Component A (pseudocode quality, 0-1): same shape as Stage-1 with S2 keywords.
    Component B (schema validity, 0-1) with graceful degradation:
      0.3 if any instance has a valid action,
      + 0.4 for fraction of instances with valid action,
      + 0.3 if verified_count present.
    """
    think = extract_think_block(text)
    answer = extract_answer_block(text)

    quality = 0.0
    if think:
        quality += 0.6 * _keyword_presence_score(think, S2_KEYWORDS, S2_KEYWORD_WEIGHTS)
        quality += 0.2 * _python_structure_score(think)
    if think and answer:
        quality += 0.2

    schema = 0.0
    data = _safe_json_loads(answer)
    if isinstance(data, (dict, list)):
        if isinstance(data, dict):
            instances_raw = data.get("instances", [])
            has_count = isinstance(data.get("verified_count", None), int) and not isinstance(data.get("verified_count"), bool)
        else:
            instances_raw = data
            has_count = False
        if isinstance(instances_raw, list) and instances_raw:
            valid_action = 0
            for obj in instances_raw:
                if isinstance(obj, dict):
                    a = obj.get("action", "")
                    if isinstance(a, str) and a.lower() in VALID_ACTIONS:
                        valid_action += 1
            if valid_action > 0:
                schema += 0.3
                schema += 0.4 * (valid_action / len(instances_raw))
                if has_count:
                    schema += 0.3
    return float(min(quality, 1.0) + min(schema, 1.0))
