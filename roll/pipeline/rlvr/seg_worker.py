import torch
import re
import ast
import json
from functools import partial
from sam2.build_sam import build_sam2
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.configs.worker_config import WorkerConfig
from roll.utils.context_managers import state_offload_manger
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.offload_states import OffloadStateType
from roll.models.model_providers import sam2_seg_model_provider
from roll.pipeline.rlvr.visual_primitives import (
    parse_stage1_state,
    stage1_state_to_bboxes,
    instances_to_sam_prompts,
)

def parse_points_from_content(content):
    """
    从内容字符串中提取一个三层嵌套的2D点列表。

    Args:
        content (str): 包含点的字符串，嵌入在 <answer> 标签内。
                       示例: "一些文本 <answer>[[[10,20],[30,40]],[[50,60]]]</answer> 更多文本"

    Returns:
        list: 解析后的列表，格式为 [[[x,y],[x,y]], ...]。
              如果找不到模式或解析失败，则返回空列表。
    """

    answer_pattern = r"<answer>(.*?)</answer>"

    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        points_text = answer_match.group(1)
        try:

            parsed_data = ast.literal_eval(points_text)




            if not isinstance(parsed_data, list):

                return []





            is_valid_structure = all(
                isinstance(group, list) and all(
                    isinstance(point, list) and len(point) == 2 and all(
                        isinstance(coord, (int, float)) for coord in point
                    )
                    for point in group
                )
                for group in parsed_data
            )

            if not is_valid_structure:

                return []

            return parsed_data




        except (ValueError, SyntaxError) as e:

            return []
        except Exception as e:

            return []
    else:

        return []

def parse_points_from_content_v2(content: str) -> list:
    """
    从包含 <answer> 标签的字符串中，提取、解析JSON格式的“坐标点字典”列表，
    并将其转换为三层嵌套的2D点列表。

    Args:
        content (str): 包含JSON数据的字符串，该数据嵌入在 <answer> 标签内。
                       示例: "一些文本 <answer>[{"p1":[1,2], "p2":[3,4]}, {"p3":[5,6]}]</answer> 更多文本"

    Returns:
        list: 解析并转换成功后的列表，格式为 [[[x,y],[x,y]], ...]。
              如果找不到 <answer> 标签，或解析失败，或数据结构不符合要求，则返回空列表。
              示例输出: [[[1,2],[3,4]], [[5,6]]]
    """

    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)

    if answer_match:

        points_text = answer_match.group(1).strip()

        try:

            parsed_data = json.loads(points_text)




            if not isinstance(parsed_data, list):
                return []


            is_valid_structure = all(
                isinstance(obj, dict) and obj and all(
                    isinstance(point, list) and len(point) == 2 and all(
                        isinstance(coord, (int, float)) for coord in point
                    )
                    for point in obj.values()
                )
                for obj in parsed_data
            )

            if not is_valid_structure:
                return []




            return [list(obj.values()) for obj in parsed_data]

        except json.JSONDecodeError:

            return []
        except Exception:

            return []
    else:

        return []

def parse_visual_prompt_from_json_s1(content: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)

    if answer_match:

        points_text = answer_match.group(1).strip()
        try:

            data = json.loads(points_text)


            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []


            for obj in data:
                try:

                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = obj.get("bbox_2d", [])


                    if isinstance(box, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:

                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:

            print(f"Error parsing JSON string: {e}. Content: {content}")

    return parsed_objects

def flow_json_to_sam_prompts(flow_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a multi-round flow_json into SAM prompt list.

    Each element: {"box": ndarray, "point_coords": ndarray?, "point_labels": ndarray?}.
    Steps with relevant=True become positive points (label=1), else negative (label=0).
    corrections.new_bboxes are appended as box-only prompts.
    """
    out: List[Dict[str, Any]] = []
    for b in flow_json.get("bboxes", []):
        bbox = b.get("bbox_2d")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        prompt: Dict[str, Any] = {"box": np.array(bbox)}
        valid_steps = [
            s for s in b.get("steps", [])
            if isinstance(s.get("pos"), list) and len(s["pos"]) == 2
        ]
        if valid_steps:
            prompt["point_coords"] = np.array([s["pos"] for s in valid_steps])
            prompt["point_labels"] = np.array(
                [1 if s.get("relevant") else 0 for s in valid_steps]
            )
        out.append(prompt)
    for bb in flow_json.get("corrections", {}).get("new_bboxes", []):
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            out.append({"box": np.array(list(bb))})
    return out


def parse_visual_prompt_from_json_s2(content: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:

        points_text = answer_match.group(1).strip()
        try:

            data = json.loads(points_text)


            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []


            for obj in data:
                try:

                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = obj.get("bbox_2d", [])
                    point_data = obj.get("points", [])

                    points = [[p[0], p[1]] for p in point_data]
                    labels = np.ones(len(points), dtype=int)
                    labels = labels.tolist()



                    if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                            "points": points,
                            "labels": labels
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:

                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:

            print(f"Error parsing JSON string: {e}. Content: {content}")

    return parsed_objects

def parse_visual_prompt_from_json_s2_old(content: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:

        points_text = answer_match.group(1).strip()
        try:

            data = json.loads(points_text)


            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []


            for obj in data:
                try:

                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = obj.get("bbox_2d", [])
                    point_data = obj.get("point_2d", [])

                    points = [[p[0], p[1]] for p in point_data]
                    labels = [p[2] for p in point_data]


                    if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                            "points": points,
                            "labels": labels
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:

                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:

            print(f"Error parsing JSON string: {e}. Content: {content}")

    return parsed_objects

def parse_visual_prompt_from_json_s2_sat(content: str, bbox_text: str) -> List[Dict[str, Any]]:
    """
    从LLM生成的JSON字符串输出中解析出视觉提示信息。

    Args:
        content (str): 一个包含物体提示信息的JSON格式字符串。
                       例如: '[{"bbox_2d": [10,100,200,210], "point_2d": [[70,180,0],[20,200,1]]}, ...]'

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键，以适配后续处理。
    """
    parsed_objects = []
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:

        points_text = answer_match.group(1).strip()
        try:

            data = json.loads(points_text)
            bbox_data = json.loads(bbox_text)


            if not isinstance(data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []
            if not isinstance(bbox_data, list):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []
            if len(data) != len(bbox_data):
                print(f"Warning: JSON content is not a list. Content: {content}")
                return []

            for obj, bbox in zip(data, bbox_data):
                try:

                    if not isinstance(obj, dict):
                        print(f"Warning: Skipped malformed object in JSON. Object: {obj}")
                        continue

                    box = bbox.get("bbox_2d", [])
                    point_data = obj.get("point_2d", [])

                    points = [[p[0], p[1]] for p in point_data]
                    labels = [p[2] for p in point_data]


                    if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list) and len(box) == 4:
                        parsed_objects.append({
                            "box": box,
                            "points": points,
                            "labels": labels
                        })
                    else:
                        print(f"Warning: Skipped object with incorrect internal data types. Object: {obj}")
                except Exception as e:

                    print(f"Error parsing object: {e}. Object: {obj}")

        except json.JSONDecodeError as e:

            print(f"Error parsing JSON string: {e}. Content: {content}")

    return parsed_objects

def parse_visual_prompt_from_content_segr1(content: str) -> List[Dict[str, Any]]:
    """
    从LLM的文本输出中解析出SAM的提示信息（边界框、点、标签）。

    Args:
        content (str): LLM生成的包含视觉提示的字符串。
                       例如："<bbox>[...],<points>[[...]],<labels>[...]</bbox>, ..."

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典代表一个物体的提示，
                              包含 'box', 'points', 'labels' 键。
    """


    pattern = re.compile(r"<bbox>(.*?)</bbox>, <points>(.*?)</points>, <labels>(.*?)</labels>")

    matches = pattern.findall(content)

    parsed_objects = []
    for box_str, points_str, labels_str in matches:
        try:

            box = ast.literal_eval(box_str)
            points = ast.literal_eval(points_str)
            labels = ast.literal_eval(labels_str)


            if isinstance(box, list) and isinstance(points, list) and isinstance(labels, list):
                parsed_objects.append({
                    "box": box,
                    "points": points,
                    "labels": labels
                })
            else:

                print(f"Warning: Skipped malformed prompt data. Box: {box_str}, Points: {points_str}")

        except (ValueError, SyntaxError) as e:

            print(f"Error parsing prompt string: {e}. Content part: {box_str}, {points_str}, {labels_str}")
            continue

    return parsed_objects


def parse_visual_prompt_from_content_samr1(content: str) -> List[Dict[str, Any]]:
    """
    从LLM的文本输出中解析出SAM的提示信息。
    此函数会先从文本中提取被 <answer>...</answer> 标签包裹的JSON字符串，然后再进行解析。

    此版本处理的输入是一个包含JSON内容的文本块。
    "points"列表中的每个元素格式为 [x, y, label]。
    函数会将其拆分为独立的 "points" ([x, y]) 和 "labels" 列表以保持输出格式的统一。

    Args:
        content (str): LLM生成的包含视觉提示的字符串，其中JSON部分被<answer>标签包裹。
                       例如：'<think>...</think><answer>{"bbox": [248,218,395,300], "points": [[360,237,1]]}</answer>'

    Returns:
        List[Dict[str, Any]]: 一个字典列表。如果解析成功，列表中将包含一个字典，
                               该字典代表一个物体的提示，包含 'box', 'points', 'labels' 键。
                               如果解析失败，则返回空列表。
    """
    parsed_objects = []
    try:


        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if not match:

            print(f"Warning: Could not find <answer> tag in the content. Content: {content}")
            return []


        json_content = match.group(1).strip()


        data = json.loads(json_content)


        box = data.get("bbox")

        points_with_labels = data.get("points")



        if box is not None and not isinstance(box, list):
             print(f"Warning: Skipped malformed JSON data. 'bbox' is not a list. Content: {content}")
             return []
        if points_with_labels is not None and not isinstance(points_with_labels, list):
            print(f"Warning: Skipped malformed JSON data. 'points' is not a list. Content: {content}")
            return []


        points_xy = []
        labels = []


        if points_with_labels:
            for point_data in points_with_labels:
                if isinstance(point_data, list) and len(point_data) == 3:

                    points_xy.append([point_data[0], point_data[1]])

                    labels.append(point_data[2])
                else:
                    print(f"Warning: Skipped malformed point data. Expected [x, y, label]. Got: {point_data}")



        parsed_objects.append({
            "box": box if box is not None else [],
            "points": points_xy,
            "labels": labels
        })

    except json.JSONDecodeError as e:

        print(f"Error parsing JSON string from <answer> tag: {e}. Content: {content}")
    except (TypeError, AttributeError) as e:

        print(f"Error processing parsed JSON data: {e}. Content: {content}")

    return parsed_objects

class SegWorker(Worker):
    """
    一个专用于SAM（Segment Anything Model）模型推理的Worker。
    它只包含模型加载和推理的逻辑，没有训练过程。
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)

        self.strategy: Optional[InferenceStrategy] = None
        self.tokenizer = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config, tokenizer=None):
        """
        初始化Worker，加载SAM模型。
        此方法会被Cluster广播到所有SAMWorker实例上。
        """
        super().initialize(pipeline_config)

        self.tokenizer = tokenizer



        self.strategy = create_strategy(worker=self)



        self.strategy.initialize(model_provider=sam2_seg_model_provider)



        self.strategy.offload_states()
        self.logger.info(f"{self.worker_name} (SAMWorker) initialized")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment(self, data: DataProto) -> DataProto:
        """
        执行SAM模型的分割推理任务。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'prompts' 等
                              SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}

        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)




        points_list = []
        for response in response_text_list:



            coords_list = parse_points_from_content(response)

            sam_prompt_list = []
            for coords in coords_list:
                point_coords = np.array(coords)

                point_labels = np.ones(point_coords.shape[0], dtype=int)

                sam_prompt_dict = {
                    'point_coords': point_coords,
                    'point_labels': point_labels
                }
                sam_prompt_list.append(sam_prompt_dict)

            points_list.append(sam_prompt_list)


        data.non_tensor_batch["visual_prompt"] = np.array(points_list, dtype=object)

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data


    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v2(self, data: DataProto) -> DataProto:
        """
        执行SAM模型的分割推理任务。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'prompts' 等
                              SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}

        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)




        points_list = []
        for response in response_text_list:



            coords_list = parse_points_from_content_v2(response)

            sam_prompt_list = []
            for coords in coords_list:
                point_coords = np.array(coords)

                point_labels = np.ones(point_coords.shape[0], dtype=int)

                sam_prompt_dict = {
                    'point_coords': point_coords,
                    'point_labels': point_labels
                }
                sam_prompt_list.append(sam_prompt_dict)

            points_list.append(sam_prompt_list)


        data.non_tensor_batch["visual_prompt"] = np.array(points_list, dtype=object)

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v3(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}


        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        stage_flag_list = data.non_tensor_batch["stage_flag"]
        prompts_for_batch = []
        for response, stage_flag in zip(response_text_list, stage_flag_list):


            if stage_flag == 1:
                parsed_objects = parse_visual_prompt_from_json_s1(response)
            else:
                parsed_objects = parse_visual_prompt_from_json_s2(response)

            sam_prompt_list_for_image = []

            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:

                    if 'box' in obj_prompts and obj_prompts['box']:

                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])


                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])


                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass

                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)


        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)




        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v4_map(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}


        response_text_list = self.tokenizer.batch_decode(data.batch["map_responses"], skip_special_tokens=True)
        prompts_for_batch = []
        for response in response_text_list:


            parsed_objects = parse_visual_prompt_from_json_s2(response)

            sam_prompt_list_for_image = []

            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:

                    if 'box' in obj_prompts and obj_prompts['box']:

                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])


                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])


                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass

                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)


        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)




        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_v4_sat(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}

        flow_list = data.non_tensor_batch.get("flow_json") if hasattr(data, "non_tensor_batch") else None
        if flow_list is not None:
            try:
                flow_iterable = list(flow_list)
            except Exception:
                flow_iterable = []
            if "responses" in data.batch:
                try:
                    response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
                except Exception:
                    response_text_list = [""] * len(flow_iterable)
            else:
                response_text_list = [""] * len(flow_iterable)
            prompts_for_batch = [flow_json_to_sam_prompts(fj if isinstance(fj, dict) else {}) for fj in flow_iterable]
        else:

            response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
            prompts_for_batch = []
            for response in response_text_list:


                parsed_objects = parse_visual_prompt_from_json_s2(response)

                sam_prompt_list_for_image = []

                for obj_prompts in parsed_objects:

                    sam_prompt_dict = {}
                    try:

                        if 'box' in obj_prompts and obj_prompts['box']:

                            if len(obj_prompts['box']) == 4:
                                sam_prompt_dict['box'] = np.array(obj_prompts['box'])


                        if 'points' in obj_prompts and obj_prompts['points']:
                            point_coords = np.array(obj_prompts['points'])
                            point_labels = np.array(obj_prompts['labels'])


                            if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                                sam_prompt_dict['point_coords'] = point_coords
                                sam_prompt_dict['point_labels'] = point_labels
                            else:
                                print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                    except Exception as e:
                        pass

                    if sam_prompt_dict:
                        sam_prompt_list_for_image.append(sam_prompt_dict)

                prompts_for_batch.append(sam_prompt_list_for_image)


        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)




        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_segr1(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}

        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        prompts_for_batch = []
        for response in response_text_list:


            parsed_objects = parse_visual_prompt_from_content_segr1(response)

            sam_prompt_list_for_image = []

            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:

                    if 'box' in obj_prompts and obj_prompts['box']:

                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])


                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])


                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass

                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)


        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)




        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_vp_stage1(self, data: "DataProto") -> "DataProto":
        """Stage-1 coarse segmentation for Multi-Instance Visual Primitive Reasoning.

        Parses the Stage-1 VisualPrimitiveState from ``map_responses`` and builds
        one box-only SAM prompt per enumerated instance. The union of per-instance
        masks forms the coarse mask that is rendered for the Stage-2 verification.
        """
        metrics = {}
        response_text_list = self.tokenizer.batch_decode(data.batch["map_responses"], skip_special_tokens=True)

        prompts_for_batch = []
        for response in response_text_list:
            state = parse_stage1_state(response)
            bboxes = stage1_state_to_bboxes(state)
            sam_prompt_list_for_image = []
            for bbox in bboxes:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    sam_prompt_list_for_image.append({"box": np.array(bbox)})
            prompts_for_batch.append(sam_prompt_list_for_image)

        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            output_batch = self.strategy.segment(batch=data)
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_vp_stage2(self, data: "DataProto") -> "DataProto":
        """Final segmentation for Multi-Instance Visual Primitive Reasoning.

        Consumes the merged kept-instance list (``vp_instances``) produced by the
        pipeline (Stage-1 primitives fused with Stage-2 keep/adjust/drop verdicts)
        and builds box + positive-point SAM prompts per instance.
        """
        metrics = {}
        if "responses" in data.batch:
            try:
                response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
            except Exception:
                response_text_list = [""] * data.batch.batch_size[0]
        else:
            response_text_list = [""] * data.batch.batch_size[0]

        instances_list = data.non_tensor_batch["vp_instances"]
        prompts_for_batch = []
        for insts in instances_list:
            try:
                insts_list = list(insts) if insts is not None else []
            except Exception:
                insts_list = []
            prompts_for_batch.append(instances_to_sam_prompts(insts_list))

        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            output_batch = self.strategy.segment(batch=data)
            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list
            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def segment_samr1(self, data: "DataProto") -> "DataProto":
        """
        执行SAM模型的分割推理任务。
        此函数已更新，可以正确解析LLM输出并构建包含多种提示的SAM输入。

        Args:
            data (DataProto): 输入数据，其 batch 字段应包含 'images', 'responses' 等
                            SAM模型推理需要的信息。

        Returns:
            DataProto: 输出数据，其 batch 字段包含推理结果，如 'masks'。
        """
        metrics = {}

        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        prompts_for_batch = []

        for response in response_text_list:


            parsed_objects = parse_visual_prompt_from_content_samr1(response)

            sam_prompt_list_for_image = []

            for obj_prompts in parsed_objects:

                sam_prompt_dict = {}
                try:

                    if 'box' in obj_prompts and obj_prompts['box']:

                        if len(obj_prompts['box']) == 4:
                            sam_prompt_dict['box'] = np.array(obj_prompts['box'])


                    if 'points' in obj_prompts and obj_prompts['points']:
                        point_coords = np.array(obj_prompts['points'])
                        point_labels = np.array(obj_prompts['labels'])


                        if point_coords.shape[0] == point_labels.shape[0] and point_coords.shape[1] == 2 and len(point_labels.shape) == 1:
                            sam_prompt_dict['point_coords'] = point_coords
                            sam_prompt_dict['point_labels'] = point_labels
                        else:
                            print(f"Warning: Mismatch between points ({point_coords.shape[0]}) and labels ({point_labels.shape[0]}). Skipping points for this object.")
                except Exception as e:
                    pass

                if sam_prompt_dict:
                    sam_prompt_list_for_image.append(sam_prompt_dict)

            prompts_for_batch.append(sam_prompt_list_for_image)


        data.non_tensor_batch["visual_prompt"] = np.array(prompts_for_batch, dtype=object)




        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/segment",

            load_kwargs={"include": [OffloadStateType.model_params]},
        ):

            data = data.to("cuda")



            output_batch = self.strategy.segment(batch=data)

            data.non_tensor_batch['mask'] = output_batch['mask']
            data.non_tensor_batch['response_text'] = response_text_list


            data.to("cpu")

        data.meta_info = {"metrics": metrics}
        return data
