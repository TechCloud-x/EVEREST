import numpy as np
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw
import warnings
import ray
import torch
import cv2
import datasets
from collections import defaultdict
from transformers import ProcessorMixin, AutoConfig
from transformers.image_utils import load_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers import BatchFeature, ProcessorMixin
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from datasets import load_from_disk, load_dataset
from ray.util.timer import _Timer
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.collator import DataCollatorWithPaddingForMultiSeg
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import GenerateScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import SocioSegConfig
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import GENERATE_SCHEDULER_NAME, RAY_NAMESPACE
from roll.utils.functionals import (
    reduce_metrics,
    RunningMoments,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

from roll.datasets.dataset import SocioSegDataset
from roll.pipeline.multi_utils import parse_points_text_from_content
from roll.pipeline.rlvr.visual_primitives import (
    parse_stage1_state,
    parse_stage2_verification,
    merge_verification_with_stage1,
    stage1_state_to_bboxs_text,
    stage1_state_to_render_items,
)

logger = get_logger()

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) for binary masks. Range: [0, 1]."""
    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0

    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union = np.logical_or(pred_mask_bool, gt_mask_bool).sum()

    if union == 0:

        return 1.0

    return float(intersection / union)


def _mask_bbox(mask: np.ndarray):
    """Returns (x1, y1, x2, y2) of the smallest axis-aligned bbox enclosing
    all nonzero pixels of the mask, or None if the mask is empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compute_giou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Generalized IoU for binary masks.

    gIoU = IoU - (|C| - |A ∪ B|) / |C|
    where C is the smallest axis-aligned bbox enclosing A ∪ B.
    Range: [-1, 1].
    """
    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0

    pred_area = pred_mask_bool.sum()
    gt_area = gt_mask_bool.sum()

    if pred_area == 0 and gt_area == 0:
        return 1.0

    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union_mask = np.logical_or(pred_mask_bool, gt_mask_bool)
    union = union_mask.sum()

    iou = intersection / union if union > 0 else 0.0

    ys, xs = np.where(union_mask)
    if ys.size == 0:
        return float(iou)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    c_area = (x2 - x1) * (y2 - y1)

    if c_area == 0:
        return float(iou)

    return float(iou - (c_area - union) / c_area)


def compute_ciou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Complete IoU on the enclosing bboxes of the two masks.

    cIoU = IoU_bbox - ρ²(center_pred, center_gt)/c² - α·v
    where c is the diagonal of the smallest enclosing bbox, v measures
    aspect-ratio consistency, and α is the trade-off weight.
    Range: [-1, 1].
    """
    pred_bbox = _mask_bbox(pred_mask)
    gt_bbox = _mask_bbox(gt_mask)

    if pred_bbox is None and gt_bbox is None:
        return 1.0
    if pred_bbox is None or gt_bbox is None:
        return -1.0

    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox


    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter_area = iw * ih
    p_area = (px2 - px1) * (py2 - py1)
    g_area = (gx2 - gx1) * (gy2 - gy1)
    union_area = p_area + g_area - inter_area
    iou_bbox = inter_area / union_area if union_area > 0 else 0.0


    cx1, cy1 = min(px1, gx1), min(py1, gy1)
    cx2, cy2 = max(px2, gx2), max(py2, gy2)
    c_diag_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    if c_diag_sq == 0:
        return float(iou_bbox)


    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
    g_cx, g_cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    center_dist_sq = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2


    pw, ph = px2 - px1, py2 - py1
    gw, gh = gx2 - gx1, gy2 - gy1
    if pw > 0 and ph > 0 and gw > 0 and gh > 0:
        v = (4.0 / (np.pi ** 2)) * (np.arctan(gw / gh) - np.arctan(pw / ph)) ** 2
        denom = (1.0 - iou_bbox) + v
        alpha = v / denom if denom > 0 else 0.0
    else:
        v, alpha = 0.0, 0.0

    return float(iou_bbox - center_dist_sq / c_diag_sq - alpha * v)


def compute_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes pixel-level F1 score for binary segmentation masks. Range: [0, 1]."""
    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0

    tp = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    fp = np.logical_and(pred_mask_bool, ~gt_mask_bool).sum()
    fn = np.logical_and(~pred_mask_bool, gt_mask_bool).sum()

    if tp == 0 and fp == 0 and fn == 0:

        return 1.0
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def format_prompt_1(prompt, processor, use_image=True, prompt_image_token=None):


    question_template = (
        "You will be given two images. The first is a map and the second is a corresponding satellite image. "
        "Find ALL instances of '{prompt}'. There may be ONE OR MANY instances; do not stop at the first one. "
        "Scan the whole image left-to-right, top-to-bottom, and anchor EVERY distinct instance with its own bounding box. "
        "Compare map and satellite cues to confirm each instance, and remove duplicates of the same instance. "
        "Output your reasoning as structured pseudocode in <think> </think> tags, "
        "and the final answer in <answer> </answer> tags. Please use English. "
        "Your <think> </think> block MUST follow this pseudocode format: "
        "def enumerate_instances(image_map, image_sat, target): "
        "    candidates = SCAN_GRID(image_map, image_sat, target) "
        "    instances = [] "
        "    for region in candidates: "
        "        bbox = ANCHOR_WITH_BOX(region) -> [x1, y1, x2, y2] "
        "        instances.append(bbox) "
        "    instances = DEDUPLICATE(instances) "
        "    count = COUNT_CONFIRM(instances) -> N "
        "    return instances, count "
        "Then output a VisualPrimitiveState in JSON: a 'target' string, a 'declared_count' integer "
        "equal to the number of instances, and an 'instances' list where each item has an integer 'id' "
        "(0-based), a 'bbox_2d' [x1,y1,x2,y2], a 'center_point' [cx,cy], an 'evidence' "
        "('map_label'|'sat_shape'|'both') and a 'confidence' ('high'|'medium'|'low'). "
        "i.e., <think>"
        "def enumerate_instances(image_map, image_sat, target): "
        "    candidates = SCAN_GRID(image_map, image_sat, target) "
        "    instances = [] "
        "    for region in candidates: "
        "        bbox = ANCHOR_WITH_BOX(region) -> [100, 200, 300, 400] "
        "        instances.append(bbox) "
        "    instances = DEDUPLICATE(instances) "
        "    count = COUNT_CONFIRM(instances) -> 2 "
        "    return instances, count "
        "</think>"
        "<answer>{answer}</answer>"
    )
    answer = (
        "{\"target\": \"<target>\", \"declared_count\": 2, \"instances\": ["
        "{\"id\": 0, \"bbox_2d\": [bx1,by1,bx2,by2], \"center_point\": [cx1,cy1], \"evidence\": \"both\", \"confidence\": \"high\"}, "
        "{\"id\": 1, \"bbox_2d\": [bx3,by3,bx4,by4], \"center_point\": [cx2,cy2], \"evidence\": \"sat_shape\", \"confidence\": \"medium\"}]}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": question_template.format(prompt=prompt, answer=answer)},
            ]
            if use_image and not prompt_image_token
            else [
                {"type": "text", "text": question_template.format(prompt=prompt, answer=answer)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text

def format_prompt_2(prompt, bboxs, processor, use_image=True, prompt_image_token=None):

    question_template = (
        "You will be given two images. The first is a map and the second is a corresponding satellite image. "
        "The Stage-1 instances for \"{prompt}\" are rendered on both images: each bounding box is drawn in blue "
        "with its integer id, and the coarse SAM mask is overlaid in red. "
        "The Stage-1 instances are: {bboxs}. "
        "Verify EACH instance independently by its id. For every instance decide one action: "
        "'keep' (correct), 'adjust' (correct object but box needs a better bbox_2d), or 'drop' "
        "(false positive or duplicate of another instance). For kept/adjusted instances, place 1-3 "
        "positive_points [x,y] that lie INSIDE the true target region to help refine the mask. "
        "Then report verified_count = number of instances you keep or adjust. "
        "Output your reasoning as structured pseudocode in <think> </think> tags, "
        "and the final answer in <answer> </answer> tags. "
        "Your <think> </think> block MUST follow this pseudocode format: "
        "def verify_instances(rendered_map, rendered_sat, instances, target): "
        "    kept = [] "
        "    for inst in instances: "
        "        quality = INSPECT(inst.id, inst.bbox, mask_overlay, target) -> good "
        "        action = DECIDE(quality) -> keep "
        "        if action != drop: "
        "            point = PLACE_POINT(inst.bbox, interior) -> [px, py] "
        "            kept.append(inst.id) "
        "    count = VERIFY_COUNT(kept) -> N "
        "    return kept, count "
        "i.e., <think>"
        "def verify_instances(rendered_map, rendered_sat, instances, target): "
        "    kept = [] "
        "    for inst in instances: "
        "        quality = INSPECT(inst.id, inst.bbox, mask_overlay, target) -> good "
        "        action = DECIDE(quality) -> keep "
        "        if action != drop: "
        "            point = PLACE_POINT(inst.bbox, interior) -> [150, 250] "
        "            kept.append(inst.id) "
        "    count = VERIFY_COUNT(kept) -> 2 "
        "    return kept, count "
        "</think>"
        "<answer>{answer}</answer>"
    )
    answer = (
        "{\"verified_count\": 2, \"instances\": ["
        "{\"id\": 0, \"action\": \"keep\", \"bbox_2d\": [bx1,by1,bx2,by2], \"positive_points\": [[px1,py1]]}, "
        "{\"id\": 1, \"action\": \"adjust\", \"bbox_2d\": [bx3,by3,bx4,by4], \"positive_points\": [[px2,py2],[px3,py3]]}]}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": question_template.format(prompt=prompt, bboxs=bboxs, answer=answer)},
            ]
            if use_image and not prompt_image_token
            else [
                {"type": "text", "text": question_template.format(prompt=prompt, bboxs=bboxs, answer=answer)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text

def process_image(images: List[Image.Image], processor: ProcessorMixin):

    image_processor = processor.image_processor
    factor = (
        image_processor.patch_size * image_processor.merge_size
        if "Qwen" in image_processor.image_processor_type
        else 28
    )
    def resize_image(image):
        height, width = image.height, image.width
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=image_processor.min_pixels,
            max_pixels=image_processor.max_pixels,
        )
        return image.resize((resized_width, resized_height), resample=image_processor.resample)
    return [resize_image(image) for image in images]

def count_components_opencv(image_list: List[Image.Image]) -> List[int]:
    counts = []
    for img in image_list:
        np_image = np.array(img)
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        _ , binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        counts.append(num_labels - 1)
    return counts

def get_bboxes(image_list: List[Image.Image]) -> str:
    all_bboxes_list = []

    for img in image_list:
        np_image = np.array(img)

        if len(np_image.shape) == 2:
            gray_image = np_image
        else:
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        _ , binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes_list = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)

                bbox_dict = {
                    "bbox_2d": list([x, y, x + w, y + h])
                }

                bboxes_list.append(bbox_dict)
        bboxes_list_str = json.dumps(bboxes_list)
        all_bboxes_list.append(bboxes_list_str)

    return all_bboxes_list

def encode_function(data_i, processor, id_key, prompt_key, label_key, image_map_key, image_sat_key):
    n = len(data_i[prompt_key])

    image_list = []
    image_flag = []
    for i in range(n):
        images = []
        ok = True
        for image in [data_i[image_map_key][i], data_i[image_sat_key][i]]:
            try:
                out = load_images([image], timeout=None)
                out = process_image(out, processor)
                images.append(out[0])
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
                ok = False
        image_list.append(images)
        image_flag.append(ok)

    id_list = [data_i.get(id_key, [f"id_{i}"])[i] for i in range(n)]

    question_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        question_list.append(instruct)

    sat_image_list = []
    map_image_list = []
    for sat_image, map_image in zip(data_i[image_sat_key], data_i[image_map_key]):
        try:
            sat_out = load_images(sat_image if isinstance(sat_image, (list, tuple)) else [sat_image], timeout=None)
            map_out = load_images(map_image if isinstance(map_image, (list, tuple)) else [map_image], timeout=None)
        except Exception:
            sat_out = [Image.new("RGB", (756,756))]
            map_out = [Image.new("RGB", (756,756))]

        sat_out = process_image(sat_out, processor)
        map_out = process_image(map_out, processor)

        sat_image_list.append(sat_out)
        map_image_list.append(map_out)

    try:
        label_out = load_images(data_i[label_key] if isinstance(data_i[label_key], (list, tuple)) else [data_i[label_key]], timeout=None)
        sat_seg_out = load_images(data_i[image_sat_key] if isinstance(data_i[image_sat_key], (list, tuple)) else [data_i[image_sat_key]], timeout=None)
    except Exception:
        label_out = [Image.new("RGB", (756,756))]
        sat_seg_out = [Image.new("RGB", (756,756))]

    object_gt = count_components_opencv(label_out)

    bbox_gt = get_bboxes(label_out)

    map_text_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        text = format_prompt_1(instruct, processor, use_image=image_flag[idx], prompt_image_token=None)
        map_text_list.append(text)

    encodings = {
        "id": id_list,
        "prompt_map": map_text_list,
        "question": question_list,
        "gt_mask": label_out,
        "gt_bbox": bbox_gt,
        "gt_object": object_gt,
        "image_sat": sat_image_list,
        "image_map": map_image_list,
        "seg_image": sat_seg_out,
        "image": image_list,
        "image_flag": image_flag,
        "tag": [""] * n
    }
    return encodings


FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


def get_dataset(data_args, encode_function, processor, features=None, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "test")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset
    data_path = None
    data_name = data_args.file_name
    data_files = []
    dataset_dir = getattr(data_args, "dataset_dir", ".")
    local_path: str = os.path.join(dataset_dir, data_name)


    dataset_builder = SocioSegDataset()
    test_path = os.path.join(local_path, "test")
    dataset = datasets.Dataset.from_generator(
        dataset_builder._generate_examples,
        gen_kwargs={"data_dir": test_path},
        features=dataset_builder.info.features
    )




    remove_columns = list(dataset.features.keys() - features.keys())

    id_key = getattr(data_args, "id") if getattr(data_args, "id", None) else "id"
    prompt_key = getattr(data_args, "prompt") if getattr(data_args, "prompt", None) else "problem"
    label_key = getattr(data_args, "mask_label") if getattr(data_args, "mask_label", None) else "mask_label"
    image_map_key = getattr(data_args, "map_image") if getattr(data_args, "map_image", None) else "map_image"
    image_sat_key = getattr(data_args, "sat_image") if getattr(data_args, "sat_image", None) else "sat_image"
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(data, processor, id_key, prompt_key, label_key, image_map_key, image_sat_key),
        batched=True,
        batch_size=100,
        num_proc=32,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def get_dataloader(dataset, batch_size, data_collator):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        collate_fn=data_collator,
    )
    return dataloader


def get_extra_data_provider(model_name_or_path: str, processor=None):
    model_name_or_path = download_model(model_name_or_path)
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    except (ValueError, OSError):

        mca_config_path = os.path.join(model_name_or_path, "mca_config.json")
        if os.path.isfile(mca_config_path):
            with open(mca_config_path, "r") as f:
                mca_config = json.load(f)
            hf_config_json = mca_config.get("hf_config_json")
            if hf_config_json:
                hf_config_dict = json.loads(hf_config_json)
                config = AutoConfig.for_model(**hf_config_dict)
            else:
                raise
        else:
            raise
    if "qwen2" in config.model_type:
        import types

        from transformers import BatchFeature
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel

        dummy_self = BatchFeature(
            {
                "config": BatchFeature(
                    {
                        "vision_config": BatchFeature({"spatial_merge_size": processor.image_processor.merge_size}),
                        "image_token_id": processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                        "video_token_id": processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                        "vision_start_token_id": processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    }
                )
            }
        )
        if hasattr(Qwen2VLForConditionalGeneration, "get_rope_index"):
            get_rope_index = types.MethodType(Qwen2VLForConditionalGeneration.get_rope_index, dummy_self)
        else:
            get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, dummy_self)

        def extra_data_provider(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            rope_index = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)[0]


            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider

    def default_extra_data_provider(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen = input_ids.shape
        position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if attention_mask is not None:
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return {"position_ids": position_ids}

    return default_extra_data_provider

def render_image(
    bboxes_json: str,
    images: List[Image.Image],
    mask: Union[np.ndarray, Image.Image]
) -> List[Image.Image]:
    rendered_images = []
    processed_mask_overlay = None
    try:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert('L'))
        else:
            mask_array = np.array(mask)

        if images:
            first_image_width, first_image_height = images[0].size
            overlay_np = np.zeros((first_image_height, first_image_width, 4), dtype=np.uint8)
            mask_array = cv2.resize(mask_array, (first_image_width, first_image_height), interpolation=cv2.INTER_NEAREST)
            mask_array = mask_array > 0
            alpha_value = int(255 * 0.4)
            mask_color = [255, 0, 0, alpha_value]
            overlay_np[mask_array] = mask_color
            processed_mask_overlay = Image.fromarray(overlay_np, 'RGBA')
        else:
            print("warning: images is empty")

    except Exception as e:

        processed_mask_overlay = None

    bboxes = []
    try:
        bbox_data: List[Dict[str, Any]] = json.loads(bboxes_json)
        if isinstance(bbox_data, list):
            for item in bbox_data:
                if isinstance(item, dict) and 'bbox_2d' in item and len(item['bbox_2d']) == 4:
                    bboxes.append(item['bbox_2d'])
                else:
                    print(f"warning: item is not a dict or bbox_2d is not in item")
    except (json.JSONDecodeError, TypeError) as e:

        bboxes = []

    for i, image in enumerate(images):
        current_rendered_image = image.copy().convert("RGBA")
        if bboxes:
            draw = ImageDraw.Draw(current_rendered_image)
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                try:
                    shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
                    draw.rectangle(shape, outline="blue", width=2)
                except Exception as e:
                    continue

        if processed_mask_overlay:
            try:
                if current_rendered_image.size != processed_mask_overlay.size:
                     resized_mask = processed_mask_overlay.resize(current_rendered_image.size, Image.Resampling.LANCZOS)
                     current_rendered_image = Image.alpha_composite(current_rendered_image, resized_mask)
                else:
                     current_rendered_image = Image.alpha_composite(current_rendered_image, processed_mask_overlay)
            except ValueError as e:

                pass

        final_image = current_rendered_image.convert("RGB")
        rendered_images.append(final_image)

    return rendered_images


def render_image_with_ids(
    render_items: List[Dict[str, Any]],
    images: List[Image.Image],
    mask: Union[np.ndarray, Image.Image],
) -> List[Image.Image]:
    """Render Stage-1 visual primitives (bbox + id label + center point) plus the
    coarse SAM mask overlay, for Stage-2 per-instance verification. Mirrors the
    training-side render_image_with_ids."""
    rendered_images = []
    processed_mask_overlay = None
    try:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert('L'))
        else:
            mask_array = np.array(mask)
        if images:
            w, h = images[0].size
            overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
            mask_array = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay_np[mask_array > 0] = [255, 0, 0, int(255 * 0.4)]
            processed_mask_overlay = Image.fromarray(overlay_np, 'RGBA')
    except Exception:
        processed_mask_overlay = None

    for image in images:
        current = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(current)
        for item in render_items or []:
            bbox = item.get("bbox_2d")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            try:
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="blue", width=2)
                draw.text((bbox[0] + 2, max(0, bbox[1] - 12)), str(item.get("id", "")), fill="yellow")
                cp = item.get("center_point")
                if isinstance(cp, (list, tuple)) and len(cp) == 2:
                    draw.ellipse([(cp[0] - 3, cp[1] - 3), (cp[0] + 3, cp[1] + 3)], fill="lime", outline="green")
            except Exception:
                continue
        if processed_mask_overlay is not None:
            try:
                overlay = processed_mask_overlay
                if current.size != processed_mask_overlay.size:
                    overlay = processed_mask_overlay.resize(current.size, Image.Resampling.LANCZOS)
                current = Image.alpha_composite(current, overlay)
            except ValueError:
                pass
        rendered_images.append(current.convert("RGB"))
    return rendered_images






_SQRT_HALF = 0.5 ** 0.5

EXPLORE_DIRECTIONS: Dict[str, Tuple[str, Tuple[float, float]]] = {
    "TL_ORTH": ("TL", (-1.0, 0.0)),
    "TL_DIAG": ("TL", (-_SQRT_HALF, -_SQRT_HALF)),
    "TR_ORTH": ("TR", (0.0, -1.0)),
    "TR_DIAG": ("TR", (_SQRT_HALF, -_SQRT_HALF)),
    "BR_ORTH": ("BR", (1.0, 0.0)),
    "BR_DIAG": ("BR", (_SQRT_HALF, _SQRT_HALF)),
    "BL_ORTH": ("BL", (0.0, 1.0)),
    "BL_DIAG": ("BL", (-_SQRT_HALF, _SQRT_HALF)),
}

MAX_STEPS_PER_BBOX = 4
EXPLORATION_ROUNDS = 4
EARLY_STOP_ACTIVE_FRAC = 0.2


def _corner_xy(bbox: List[int], corner: str) -> Tuple[int, int]:
    bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
    if corner == "TL":
        return bx1, by1
    if corner == "TR":
        return bx2, by1
    if corner == "BR":
        return bx2, by2
    if corner == "BL":
        return bx1, by2
    raise ValueError(f"Unknown corner: {corner}")


def compute_probe_pos(
    bbox: List[int],
    dir_name: str,
    step: int,
    image_wh: Optional[Tuple[int, int]] = None,
) -> List[int]:
    if dir_name not in EXPLORE_DIRECTIONS:
        raise ValueError(f"Unknown direction: {dir_name}")
    corner, (dx, dy) = EXPLORE_DIRECTIONS[dir_name]
    ox, oy = _corner_xy(bbox, corner)
    bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
    short_side = max(1, min(bx2 - bx1, by2 - by1))
    step_size = max(8.0, short_side * 0.15)
    px = int(round(ox + step * step_size * dx))
    py = int(round(oy + step * step_size * dy))
    if image_wh is not None:
        W, H = image_wh
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
    return [px, py]


def _parse_bboxs_text(bboxs_text: str) -> List[List[int]]:
    if not bboxs_text:
        return []
    try:
        data = json.loads(bboxs_text.replace("'", '"'))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[List[int]] = []
    for item in data:
        if isinstance(item, dict) and "bbox_2d" in item and isinstance(item["bbox_2d"], list) and len(item["bbox_2d"]) == 4:
            try:
                out.append([int(v) for v in item["bbox_2d"]])
            except Exception:
                continue
    return out


def init_flow_json(bboxs_text: str) -> Dict[str, Any]:
    boxes = _parse_bboxs_text(bboxs_text)
    return {
        "round": 0,
        "bboxes": [
            {
                "bbox_id": i,
                "bbox_2d": boxes[i],
                "status": "exploring",
                "steps_used": 0,
                "steps": [],
            }
            for i in range(len(boxes))
        ],
        "corrections": {"new_bboxes": []},
    }


def parse_round_actions(response_text: str) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    actions: List[Dict[str, Any]] = []
    new_cands: List[List[int]] = []
    m = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    if not m:
        return actions, new_cands
    try:
        data = json.loads(m.group(1).strip())
    except Exception:
        return actions, new_cands
    if not isinstance(data, dict):
        return actions, new_cands
    raw_actions = data.get("actions", [])
    if isinstance(raw_actions, list):
        for a in raw_actions:
            if isinstance(a, dict) and "bbox_id" in a and "act" in a:
                actions.append(a)
    raw_cands = data.get("new_bbox_candidates", [])
    if isinstance(raw_cands, list):
        for bb in raw_cands:
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                try:
                    new_cands.append([int(v) for v in bb])
                except Exception:
                    pass
    return actions, new_cands


def update_flow_json(
    flow_json: Dict[str, Any],
    actions: List[Dict[str, Any]],
    new_cands: List[List[int]],
    round_id: int,
) -> Dict[str, Any]:
    flow_json["round"] = round_id
    bbox_lookup = {b["bbox_id"]: b for b in flow_json["bboxes"]}
    for a in actions:
        bid = a.get("bbox_id")
        if bid not in bbox_lookup:
            continue
        b = bbox_lookup[bid]
        if b["status"] in ("stopped", "skipped"):
            continue
        act = str(a.get("act", "")).upper()
        if act == "STOP":
            b["status"] = "stopped"
        elif act == "SKIP":
            b["status"] = "skipped"
        elif act == "PROBE":
            pos = a.get("pos", [])
            dir_name = a.get("dir")
            step_idx = a.get("step", 1)
            try:
                step_idx_int = int(step_idx)
            except Exception:
                step_idx_int = 1
            if (
                dir_name in EXPLORE_DIRECTIONS
                and isinstance(pos, (list, tuple))
                and len(pos) == 2
            ):
                try:
                    pos_int = [int(pos[0]), int(pos[1])]
                except Exception:
                    continue
                b["steps"].append(
                    {
                        "round": round_id,
                        "dir": dir_name,
                        "step": step_idx_int,
                        "pos": pos_int,
                        "relevant": bool(a.get("relevant", False)),
                    }
                )
                b["steps_used"] = b["steps_used"] + 1
                if b["steps_used"] >= MAX_STEPS_PER_BBOX:
                    b["status"] = "stopped"
    for bb in new_cands:
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            flow_json["corrections"]["new_bboxes"].append(list(bb))
    return flow_json


def active_bbox_count(flow_json: Dict[str, Any]) -> int:
    return sum(1 for b in flow_json.get("bboxes", []) if b.get("status") == "exploring")


def flow_json_to_sam_prompts(flow_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in flow_json.get("bboxes", []):
        bbox = b.get("bbox_2d")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        prompt: Dict[str, Any] = {"box": np.array(bbox)}
        valid_steps = [s for s in b.get("steps", []) if isinstance(s.get("pos"), list) and len(s["pos"]) == 2]
        if valid_steps:
            prompt["point_coords"] = np.array([s["pos"] for s in valid_steps])
            prompt["point_labels"] = np.array([1 if s.get("relevant") else 0 for s in valid_steps])
        out.append(prompt)
    for bb in flow_json.get("corrections", {}).get("new_bboxes", []):
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            out.append({"box": np.array(list(bb))})
    return out


def flow_json_to_bboxs_text(flow_json: Dict[str, Any]) -> str:
    items: List[Dict[str, Any]] = []
    for b in flow_json.get("bboxes", []):
        bb = b.get("bbox_2d")
        if isinstance(bb, list) and len(bb) == 4:
            items.append({"bbox_2d": list(bb)})
    for bb in flow_json.get("corrections", {}).get("new_bboxes", []):
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            items.append({"bbox_2d": list(bb)})
    return json.dumps(items)


def flow_json_to_summary(flow_json: Dict[str, Any]) -> str:
    """Compact text summary of flow_json for prompt injection."""
    try:
        return json.dumps(flow_json, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


def format_prompt_2_round(
    prompt: str,
    flow_summary: str,
    round_id: int,
    processor,
    use_image: bool = True,
    prompt_image_token: Optional[str] = None,
    max_rounds: int = 4,
) -> str:
    """Build the VLM prompt for round-t of Stage-2 multi-round exploration.

    flow_summary is a compact text representation of the cumulative flow_json state.
    Must stay in sync with the training-side format_prompt_2_round in rlvr_socioseg_vlm_pipeline.py.
    """
    question_template = (
        "You will be given two images: a rendered map and a rendered satellite image. "
        "Both show the Stage-1 bbox outlines (blue) and segmentation mask overlays (red) for '{prompt}'. "
        "This is round {round_id} of K={K} multi-round exploration. "
        "Your goal is to refine each Stage-1 bbox via symbolic probing along 8 outward rays from its corners.\n"
        "Current flow state: {flow_summary}\n"
        "Action space (per bbox, 8 directions, all pointing OUTward from the bbox):\n"
        "  TL_ORTH origin=(bx1,by1) dir=left;       TL_DIAG origin=(bx1,by1) dir=up-left-45\n"
        "  TR_ORTH origin=(bx2,by1) dir=up;         TR_DIAG origin=(bx2,by1) dir=up-right-45\n"
        "  BR_ORTH origin=(bx2,by2) dir=right;      BR_DIAG origin=(bx2,by2) dir=down-right-45\n"
        "  BL_ORTH origin=(bx1,by2) dir=down;       BL_DIAG origin=(bx1,by2) dir=down-left-45\n"
        "step_size = max(8, min(bbox_w, bbox_h) * 0.15); pos = corner + step * step_size * unit_vec.\n"
        "For each bbox whose status='exploring', choose exactly ONE action:\n"
        "  PROBE: advance one step along an unused (dir, step). Report pos and relevant as true/false.\n"
        "  STOP : stop exploring this bbox (enough evidence).\n"
        "  SKIP : skip exploration (high confidence, no probing needed).\n"
        "Optionally propose new_bbox_candidates as 4-tuples [x1,y1,x2,y2] if you spot a missed instance.\n"
        "Each bbox can take AT MOST 4 PROBE steps across all rounds. Use them sparingly.\n"
        "Output reasoning as pseudocode in <think>...</think>; final JSON in <answer>...</answer>.\n"
        "Your <think> MUST follow this pseudocode format:\n"
        "def explore_round(rendered_sat, rendered_map, flow_json, target):\n"
        "    actions = []\n"
        "    for bbox in flow_json.bboxes:\n"
        "        if bbox.status in [stopped, skipped]: continue\n"
        "        if bbox.steps_used >= 4: STOP(bbox); continue\n"
        "        if EVIDENCE_SUFFICIENT(bbox): STOP(bbox); continue\n"
        "        dir = CHOOSE_DIR(bbox, unexplored) -> 'BR_DIAG'\n"
        "        step = NEXT_STEP_INDEX(bbox, dir) -> 2\n"
        "        pos = COMPUTE_POS(bbox, dir, step) -> [316, 416]\n"
        "        relevant = INSPECT(pos, rendered_sat, rendered_map, target) -> false\n"
        "        actions.append(bbox_id, act=PROBE, dir, step, pos, relevant)\n"
        "    new_bbox_candidates = DETECT_MISSED(images, target) -> []\n"
        "    return actions, new_bbox_candidates\n"
        "</think>\n"
        "<answer>{answer}</answer>"
    )
    answer_literal = (
        '{"round": ' + str(round_id) + ', "actions": ['
        '{"bbox_id": 0, "act": "PROBE", "dir": "BR_DIAG", "step": 1, "pos": [bx, by], "relevant": true}, '
        '{"bbox_id": 1, "act": "STOP"}'
        '], "new_bbox_candidates": []}'
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": question_template.format(
                    prompt=prompt,
                    round_id=round_id,
                    K=max_rounds,
                    flow_summary=flow_summary,
                    answer=answer_literal,
                )},
            ]
            if use_image and not prompt_image_token
            else [
                {"type": "text", "text": question_template.format(
                    prompt=prompt,
                    round_id=round_id,
                    K=max_rounds,
                    flow_summary=flow_summary,
                    answer=answer_literal,
                )},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text


def draw_visual_prompt(image: Image.Image, mask: Union[np.ndarray, Image.Image], visual_prompt) -> Image.Image:

    prompt_data = visual_prompt

    rendered_image = image.copy().convert("RGBA")
    width, height = rendered_image.size

    try:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert('L'))
        else:
            mask_array = np.array(mask)

        overlay_np = np.zeros((height, width, 4), dtype=np.uint8)

        if mask_array.shape[0] != height or mask_array.shape[1] != width:
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)

        bool_mask = mask_array > 0

        alpha_value = int(255 * 0.4)
        mask_color = [255, 0, 0, alpha_value]

        overlay_np[bool_mask] = mask_color
        mask_overlay = Image.fromarray(overlay_np, 'RGBA')
        rendered_image = Image.alpha_composite(rendered_image, mask_overlay)

    except Exception as e:

        pass

    draw = ImageDraw.Draw(rendered_image)

    if "box" in prompt_data:
        bbox = prompt_data["box"]
        bbox = bbox.tolist()
        if isinstance(bbox, list) and len(bbox) == 4:
            shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
            draw.rectangle(shape, outline="blue", width=2)

    if "point_coords" in prompt_data and "point_labels" in prompt_data:
        points = prompt_data.get("point_coords", [])
        labels = prompt_data.get("point_labels", [])
        radius = 5

        for point, label in zip(points, labels):
            point = point.tolist()
            if isinstance(point, list) and len(point) == 2:
                x, y = point
                point_bbox = [x - radius, y - radius, x + radius, y + radius]
                if label == 1:
                    fill_color = "green"
                else:
                    fill_color = "red"
                draw.ellipse(point_bbox, fill=fill_color, outline=None)
    return rendered_image.convert("RGB")

class SocioSegInferPipeline(BasePipeline):
    def __init__(self, pipeline_config: SocioSegConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)

        self.processor.image_processor.max_pixels, self.processor.image_processor.min_pixels = (
            getattr(self.pipeline_config.actor_train.model_args, "max_pixels", 768 * 768),
            getattr(self.pipeline_config.actor_train.model_args, "min_pixels", 56 * 56),
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        features = datasets.Features(
            {

                "id": datasets.Value("string"),
                "prompt_map": datasets.Value("string"),
                "question": datasets.Value("string"),
                "gt_mask": datasets.Image(decode=True),
                "seg_image": datasets.Image(decode=True),
                "gt_object": datasets.Value("int32"),
                "gt_bbox": datasets.Value("string"),
                "image_sat": datasets.Sequence(datasets.Image(decode=True)),
                "image_map": datasets.Sequence(datasets.Image(decode=True)),
                "image": datasets.Sequence(datasets.Image(decode=True)),

                "image_flag": datasets.Value("bool"),

                "tag": datasets.Value("string"),
            }
        )
        dataset = get_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, features, get_eval=False
        )

        self.extra_data_provider = get_extra_data_provider(
            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
        )
        data_collator = DataCollatorWithPaddingForMultiSeg(
            tokenizer=self.tokenizer,
            processor=self.processor,
            extra_data_provider=self.extra_data_provider,
            max_length=self.pipeline_config.prompt_length,
            image_key="image",
            padding="max_length",
            gt_object_key="gt_object",
            gt_bbox_key="gt_bbox",

        )
        self.dataloader = get_dataloader(dataset, self.pipeline_config.rollout_batch_size, data_collator)

        max_steps = len(self.dataloader) * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        self.seg_infer: Any = Cluster(
            name=self.pipeline_config.seg_infer.name,
            worker_cls=self.pipeline_config.seg_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.seg_infer,
        )

        self.actor_train: Any = Cluster(
            name="actor_train_actor",
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )

        self.actor_infer: Any = Cluster(
            name="actor_infer_actor",
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )


        self.generate_scheduler = GenerateScheduler.options(
            name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        refs.extend(self.seg_infer.initialize(pipeline_config=self.pipeline_config, blocking=False, tokenizer=self.tokenizer))
        ray.get(refs)


        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        self.running = RunningMoments()

    @torch.no_grad()
    def run(self):
        global_step = 0

        result_dir = os.path.join(self.pipeline_config.output_dir, "result")
        result_save_dirs = {
            "stage1": os.path.join(result_dir, "stage1"),
            "stage2": os.path.join(result_dir, "stage2"),
            "render1": os.path.join(result_dir, "render1"),
            "render2": os.path.join(result_dir, "render2"),
        }
        for save_dir in result_save_dirs.values():
            os.makedirs(save_dir, exist_ok=True)

        tps_timer = _Timer(window_size=5)
        actor_infer_timer1 = _Timer(window_size=5)
        actor_infer_response_timer1 = _Timer(window_size=5)
        actor_infer_timer2 = _Timer(window_size=5)
        actor_infer_response_timer2 = _Timer(window_size=5)
        seg_infer_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        _METRIC_NAMES = ("iou", "ciou", "giou", "f1")
        _CATEGORIES = ("name", "class", "function")
        all_metrics = {k: [] for k in _METRIC_NAMES}
        cat_metrics = {c: {k: [] for k in _METRIC_NAMES} for c in _CATEGORIES}
        for batch_dict in tqdm(self.dataloader):
            metrics = {}
            with tps_timer:


                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)

                batch_dict: Dict
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info = {
                    "global_step": global_step,

                    "_broadcast_non_tensor_batch": True,
                }

                with actor_infer_timer1, actor_infer_response_timer1:


                    gen_batch = batch.pop(
                        batch_keys=["map_input_ids", "map_attention_mask", "map_position_ids"],
                        non_tensor_batch_keys=(
                            ["multi_modal_map_data"] if "multi_modal_map_data" in batch.non_tensor_batch else []
                        ),
                    )
                    gen_batch.rename("map_input_ids", "input_ids")
                    gen_batch.rename("map_attention_mask", "attention_mask")
                    gen_batch.rename("map_position_ids", "position_ids")
                    gen_batch.non_tensor_batch["multi_modal_data"] = gen_batch.non_tensor_batch.pop("multi_modal_map_data")








                    gen_batch.meta_info = {"global_step": global_step}
                    gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                    generate_output: DataProto = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        ),
                        timeout=self.pipeline_config.rpc_timeout,
                    )
                    metrics.update(reduce_metrics(generate_output.meta_info.pop("metrics", {})))



                generate_output.rename(old_keys="input_ids", new_keys="map_input_ids")
                generate_output.rename(old_keys="attention_mask", new_keys="map_attention_mask")
                generate_output.rename(old_keys="position_ids", new_keys="map_position_ids")
                generate_output.rename(old_keys="responses", new_keys="map_responses")
                generate_output.rename(old_keys="response_mask", new_keys="map_response_mask")
                generate_output.rename(old_keys="prompts", new_keys="map_prompts")
                generate_output.rename(old_keys="prompt_mask", new_keys="map_prompt_mask")




                for key, value in batch.non_tensor_batch.items():
                    batch.non_tensor_batch[key] = np.repeat(
                        value, self.actor_infer.worker_config.generating_args.num_return_sequences
                    )

                batch.batch = generate_output.batch
                batch = batch.union(generate_output)

                with seg_infer_timer:
                    seg_batch = batch.pop(
                        batch_keys=["map_responses", "map_prompts"],
                        non_tensor_batch_keys=['seg_image']
                    )
                    seg_batch_refs: List[ray.ObjectRef] = self.seg_infer.segment_vp_stage1(seg_batch, blocking=False)

                seg_batch_out: DataProto = DataProto.materialize_concat(data_refs=seg_batch_refs)
                batch = batch.union(seg_batch_out)

                batch.non_tensor_batch["map_mask"] = batch.non_tensor_batch.pop("mask")
                batch.non_tensor_batch["map_visual_prompt"] = batch.non_tensor_batch.pop("visual_prompt")
                batch.non_tensor_batch.pop("response_text")


                response_list = self.tokenizer.batch_decode(batch.batch["map_responses"], skip_special_tokens=False)
                s1_state_list = [parse_stage1_state(r) for r in response_list]
                bboxs_text_list = [stage1_state_to_bboxs_text(s) for s in s1_state_list]
                render_items_list = [stage1_state_to_render_items(s) for s in s1_state_list]


                sat_text_list = []
                for instruct, bboxs_text in zip(batch.non_tensor_batch["question"], bboxs_text_list):
                    text = format_prompt_2(instruct, bboxs_text, self.processor)
                    sat_text_list.append(text)

                sat_padded_features = defaultdict(list)
                un_padded_features = defaultdict(list)
                mm_feature_keys = set()

                zipped_features = zip(
                    sat_text_list,
                    render_items_list,
                    batch.non_tensor_batch["image"],
                    batch.non_tensor_batch["map_mask"],
                )

                for text, render_items, image_sat, mask in zipped_features:
                    rd_image = render_image_with_ids(render_items, image_sat, mask)

                    sat_model_inputs: BatchFeature = self.processor(
                        images=rd_image,
                        text=text,
                    )
                    for key in ["prompt_sat"]:
                        if key in sat_model_inputs:
                            sat_model_inputs.pop(key)

                    padded_keys = ["input_ids", "attention_mask", "labels"]
                    for key in filter(lambda k: k in sat_model_inputs, padded_keys):
                        sat_padded_features[key].append(sat_model_inputs.pop(key)[0])

                    mm_feature_keys = mm_feature_keys.union(sat_model_inputs.keys())

                    sat_model_inputs.convert_to_tensors(tensor_type='pt')

                    un_padded_features["multi_modal_sat_inputs"].append(dict(sat_model_inputs))

                    un_padded_features["multi_modal_sat_data"].append(
                        {
                            "prompt_token_ids":
                            self.tokenizer.encode(text, add_special_tokens=False),
                            "multi_modal_data": {
                                "image": [rd_image] if not isinstance(rd_image, list) else rd_image,
                            },
                        }
                    )

                sat_batch = pad_without_fast_tokenizer_warning(
                    self.tokenizer,
                    sat_padded_features,
                    padding='max_length',
                    max_length=self.pipeline_config.prompt_length,
                    pad_to_multiple_of=None,
                    return_tensors='pt',
                )
                sat_batch.update(un_padded_features)

                sat_fun_params = ['input_ids', 'attention_mask', 'image_grid_thw']
                sat_kwargs = {}
                for key in sat_fun_params:
                    if key in sat_batch:
                        sat_kwargs[key] = sat_batch[key]
                    elif key in mm_feature_keys:
                        mm_inputs = [inputs[key] for inputs in sat_batch["multi_modal_sat_inputs"] if key in inputs]
                        if mm_inputs:
                            sat_kwargs[key] = torch.concat(mm_inputs, dim=0)
                        else:
                            print(f"Warning: {key} not found in any multi-modal inputs, using default value.")
                            exit()
                    else:
                        print(f"Warning: {key} not found in batch, using default value.")
                        exit()

                sat_extra_data = self.extra_data_provider(**sat_kwargs)
                sat_extra_data['position_ids'] = sat_extra_data.pop('position_ids')
                sat_extra_data['attention_mask'] = sat_kwargs.pop('attention_mask')
                sat_extra_data['input_ids'] = sat_kwargs.pop('input_ids')
                sat_batch.update(sat_extra_data)

                sat_batch['bboxs_text'] = bboxs_text_list

                for key in sat_batch:
                    if isinstance(sat_batch[key], (torch.Tensor, np.ndarray)):
                        assert sat_batch[key].shape[0] == sat_batch["input_ids"].shape[0]
                    else:
                        assert len(sat_batch[key]) == sat_batch["input_ids"].shape[0]
                        sat_batch[key] = np.array(sat_batch[key], dtype=object)

                sat_batch: DataProto = DataProto.from_single_dict(sat_batch)

                batch = batch.union(sat_batch)

                with actor_infer_timer2, actor_infer_response_timer2:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=(
                            ["multi_modal_sat_data"] if "multi_modal_sat_data" in batch.non_tensor_batch else []
                        ),
                    )
                    gen_batch.non_tensor_batch["multi_modal_data"] = gen_batch.non_tensor_batch.pop("multi_modal_sat_data")
                    ori_num_return_sequences = self.pipeline_config.actor_infer.generating_args.num_return_sequences
                    self.pipeline_config.actor_infer.generating_args.num_return_sequences = 1
                    gen_batch.meta_info = {"global_step": global_step}
                    gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                    generate_output = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        ),
                        timeout=self.pipeline_config.rpc_timeout,
                    )
                    self.pipeline_config.actor_infer.generating_args.num_return_sequences = ori_num_return_sequences
                    metrics.update(reduce_metrics(generate_output.meta_info.pop("metrics", {})))

                batch = batch.union(generate_output)


                sat_response_texts = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
                merged_instances_list = []
                for s1_state, resp_text in zip(s1_state_list, sat_response_texts):
                    verif = parse_stage2_verification(resp_text)
                    merged_instances_list.append(merge_verification_with_stage1(s1_state, verif))



                batch.non_tensor_batch["vp_instances"] = np.array(merged_instances_list, dtype=object)

                with seg_infer_timer:
                    seg_batch = batch.pop(
                        batch_keys=["responses", "prompts"],
                        non_tensor_batch_keys=['seg_image', 'vp_instances']
                    )
                    seg_batch_refs: List[ray.ObjectRef] = self.seg_infer.segment_vp_stage2(seg_batch, blocking=False)

                seg_batch_out: DataProto = DataProto.materialize_concat(data_refs=seg_batch_refs)
                seg_batch_out.meta_info.pop("metrics")
                batch = batch.union(seg_batch_out)
                batch.non_tensor_batch["sat_mask"] = batch.non_tensor_batch.pop("mask")
                batch.non_tensor_batch["sat_visual_prompt"] = batch.non_tensor_batch.pop("visual_prompt")

                batch_metrics = {k: [] for k in _METRIC_NAMES}
                map_response_list = self.tokenizer.batch_decode(batch.batch["map_responses"], skip_special_tokens=False)
                sat_response_list = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
                for i in range(len(batch)):
                    gt_mask = np.array(batch[i].non_tensor_batch["gt_mask"].convert("L"))
                    pred_mask = batch[i].non_tensor_batch["sat_mask"]

                    sample_scores = {
                        "iou": compute_iou(pred_mask, gt_mask),
                        "ciou": compute_ciou(pred_mask, gt_mask),
                        "giou": compute_giou(pred_mask, gt_mask),
                        "f1": compute_f1(pred_mask, gt_mask),
                    }
                    for k in _METRIC_NAMES:
                        batch_metrics[k].append(sample_scores[k])

                    sample_id = batch[i].non_tensor_batch["id"]
                    prefix = sample_id.split("_", 1)[0] if isinstance(sample_id, str) else None
                    if prefix in cat_metrics:
                        for k in _METRIC_NAMES:
                            cat_metrics[prefix][k].append(sample_scores[k])

                    try:
                        map_visual_prompt = batch[i].non_tensor_batch["map_visual_prompt"][0]
                        sat_visual_prompt = batch[i].non_tensor_batch["sat_visual_prompt"][0]
                    except:
                        map_visual_prompt = {}
                        sat_visual_prompt = {}
                    image_s1 = draw_visual_prompt(batch[i].non_tensor_batch["seg_image"], batch[i].non_tensor_batch["map_mask"], map_visual_prompt)
                    image_s2 = draw_visual_prompt(batch[i].non_tensor_batch["seg_image"], batch[i].non_tensor_batch["sat_mask"], sat_visual_prompt)
                    mask_s1 = batch[i].non_tensor_batch["map_mask"]
                    mask_s2 = batch[i].non_tensor_batch["sat_mask"]
                    save_id = sample_id

                    mask_s1 = mask_s1.astype(np.uint8) * 255
                    mask_s2 = mask_s2.astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(result_save_dirs["stage1"], f"{save_id}.png"), mask_s1)
                    cv2.imwrite(os.path.join(result_save_dirs["stage2"], f"{save_id}.png"), mask_s2)
                    image_s1.save(os.path.join(result_save_dirs["render1"], f"{save_id}.png"))
                    image_s2.save(os.path.join(result_save_dirs["render2"], f"{save_id}.png"))


                    with open(os.path.join(result_save_dirs["stage1"], f"{save_id}.txt"), "w") as f:
                        f.write(map_response_list[i])
                    with open(os.path.join(result_save_dirs["stage2"], f"{save_id}.txt"), "w") as f:
                        f.write(sat_response_list[i])

                print(
                    "batch: "
                    + ", ".join(f"{k}={np.mean(batch_metrics[k]):.4f}" for k in _METRIC_NAMES)
                )
                for k in _METRIC_NAMES:
                    all_metrics[k].extend(batch_metrics[k])

        os.makedirs(result_dir, exist_ok=True)

        def _write_result(path, md):
            n = len(md["iou"])
            with open(path, "w") as f:
                for k in _METRIC_NAMES:
                    mean_v = float(np.mean(md[k])) if md[k] else 0.0
                    f.write(f"{k}: {mean_v:.6f}\n")
                f.write(f"count: {n}\n")

        _write_result(os.path.join(result_dir, "result.txt"), all_metrics)
        for c in _CATEGORIES:
            _write_result(os.path.join(result_dir, f"result_{c}.txt"), cat_metrics[c])

        print(
            "overall: "
            + ", ".join(f"{k}={np.mean(all_metrics[k]) if all_metrics[k] else 0.0:.4f}" for k in _METRIC_NAMES)
        )
        for c in _CATEGORIES:
            md = cat_metrics[c]
            if md["iou"]:
                print(
                    f"[{c}] n={len(md['iou'])}, "
                    + ", ".join(f"{k}={np.mean(md[k]):.4f}" for k in _METRIC_NAMES)
                )
            else:
                print(f"[{c}] n=0 (no samples)")
