import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import BatchFeature, PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy


def collate_fn_to_dict_list(data_list: list[dict]) -> dict:
    """将list[dict]数据转成dict[list]"""
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.cat(val, dim=0)

    for key, val in non_tensors.items():
        tem_array = np.empty([len(val)], dtype=object)
        tem_array[:] = val
        non_tensors[key] = tem_array

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


@dataclass
class DataCollatorWithPaddingForPaddedKeys:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_features = [{k: v for k, v in feature.items() if k in self.padded_keys} for feature in features]
        un_padded_features = [{k: v for k, v in feature.items() if k not in self.padded_keys} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["position_ids"] = torch.clip(torch.cumsum(batch["attention_mask"], dim=-1) - 1, min=0, max=None)
        un_padded_batch = collate_fn_to_dict_list(un_padded_features)
        batch.update(un_padded_batch)
        return batch


@dataclass
class DataCollatorWithPaddingForMM:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    processor: Optional[ProcessorMixin] = None
    extra_data_provider: Optional[callable] = None
    prompt_key: str = "prompt"
    answer_key: Optional[str] = "ground_truth"
    image_key: Optional[str] = "image"
    image_flag_key: Optional[str] = "image_flag"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.tokenizer and self.processor

        padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:




            model_inputs: BatchFeature = self.processor(
                images=feature[self.image_key]
                if self.image_key and (not self.image_flag_key or feature[self.image_flag_key])
                else None,
                text=feature[self.prompt_key],
            )
            for key in ["prompt"]:
                if key in model_inputs:
                    model_inputs.pop(key)
            for key in filter(lambda k: k in model_inputs, self.padded_keys):
                padded_features[key].append(model_inputs.pop(key)[0])

            mm_feature_keys = mm_feature_keys.union(model_inputs.keys())

            model_inputs.convert_to_tensors(tensor_type=self.return_tensors)
            if self.image_key:




                un_padded_features["multi_modal_inputs"].append(dict(model_inputs))

                un_padded_features["multi_modal_data"].append(
                    {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                        "multi_modal_data": {
                            "image": [feature[self.image_key]]
                            if not isinstance(feature[self.image_key], list)
                            else feature[self.image_key]
                        },
                    }
                    if not self.image_flag_key or feature[self.image_flag_key]
                    else {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                    }
                )
            if self.answer_key:
                un_padded_features[self.answer_key].append(feature[self.answer_key])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update(un_padded_features)





        if self.extra_data_provider:
            fun_params = inspect.signature(self.extra_data_provider).parameters
            kwargs = {}
            for key in fun_params:
                if key in batch:
                    kwargs[key] = batch[key]
                elif key in mm_feature_keys:
                    mm_inputs = [inputs[key] for inputs in batch["multi_modal_inputs"] if key in inputs]
                    kwargs[key] = torch.concat(mm_inputs, dim=0) if mm_inputs else fun_params[key].default
                else:
                    kwargs[key] = fun_params[key].default
            extra_data = self.extra_data_provider(**kwargs)
            batch.update(extra_data)



        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                assert batch[key].shape[0] == batch["input_ids"].shape[0]
            else:
                assert len(batch[key]) == batch["input_ids"].shape[0]
                batch[key] = np.array(batch[key], dtype=object)
        return batch


@dataclass
class DataCollatorWithPaddingForSegZero:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    processor: Optional[ProcessorMixin] = None
    extra_data_provider: Optional[callable] = None
    prompt_key: str = "prompt"
    image_key: Optional[str] = "image"
    sat_image_key: Optional[str] = "seg_image"
    id_key: Optional[str] = "id"
    gt_key: Optional[str] = "gt"
    gt_object_key: Optional[str] = "gt_object"
    image_flag_key: Optional[str] = "image_flag"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.tokenizer and self.processor

        padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:




            model_inputs: BatchFeature = self.processor(
                images=feature[self.image_key]
                if self.image_key and (not self.image_flag_key or feature[self.image_flag_key])
                else None,
                text=feature[self.prompt_key],
            )
            for key in ["prompt"]:
                if key in model_inputs:
                    model_inputs.pop(key)
            for key in filter(lambda k: k in model_inputs, self.padded_keys):
                padded_features[key].append(model_inputs.pop(key)[0])

            mm_feature_keys = mm_feature_keys.union(model_inputs.keys())

            model_inputs.convert_to_tensors(tensor_type=self.return_tensors)
            if self.image_key:




                un_padded_features["multi_modal_inputs"].append(dict(model_inputs))

                un_padded_features["multi_modal_data"].append(
                    {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                        "multi_modal_data": {
                            "image": [feature[self.image_key]]
                            if not isinstance(feature[self.image_key], list)
                            else feature[self.image_key]
                        },
                    }
                    if not self.image_flag_key or feature[self.image_flag_key]
                    else {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                    }
                )
            if self.gt_key:
                un_padded_features[self.gt_key].append(feature[self.gt_key])
            if self.sat_image_key:
                un_padded_features[self.sat_image_key].append(feature[self.sat_image_key])
            if self.id_key:
                un_padded_features[self.id_key].append(feature[self.id_key])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update(un_padded_features)





        if self.extra_data_provider:
            fun_params = inspect.signature(self.extra_data_provider).parameters
            kwargs = {}
            for key in fun_params:
                if key in batch:
                    kwargs[key] = batch[key]
                elif key in mm_feature_keys:
                    mm_inputs = [inputs[key] for inputs in batch["multi_modal_inputs"] if key in inputs]
                    kwargs[key] = torch.concat(mm_inputs, dim=0) if mm_inputs else fun_params[key].default
                else:
                    kwargs[key] = fun_params[key].default
            extra_data = self.extra_data_provider(**kwargs)
            batch.update(extra_data)



        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                assert batch[key].shape[0] == batch["input_ids"].shape[0]
            else:
                assert len(batch[key]) == batch["input_ids"].shape[0]
                temp_array = np.empty([len(batch[key])], dtype=object)
                temp_array[:] = batch[key]
                batch[key] = temp_array

        return batch

@dataclass
class DataCollatorWithPaddingForSeg:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    processor: Optional[ProcessorMixin] = None
    extra_data_provider: Optional[callable] = None
    prompt_key: str = "prompt"
    image_key: Optional[str] = "image"
    sat_image_key: Optional[str] = "seg_image"
    id_key: Optional[str] = "id"
    gt_mask_key: Optional[str] = "gt_mask"
    gt_object_key: Optional[str] = None
    gt_center_key: Optional[str] = None
    gt_bbox_key: Optional[str] = None
    stage_flag_key: Optional[str] = None
    image_flag_key: Optional[str] = "image_flag"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.tokenizer and self.processor

        padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:




            model_inputs: BatchFeature = self.processor(
                images=feature[self.image_key]
                if self.image_key and (not self.image_flag_key or feature[self.image_flag_key])
                else None,
                text=feature[self.prompt_key],
            )
            for key in ["prompt"]:
                if key in model_inputs:
                    model_inputs.pop(key)
            for key in filter(lambda k: k in model_inputs, self.padded_keys):
                padded_features[key].append(model_inputs.pop(key)[0])

            mm_feature_keys = mm_feature_keys.union(model_inputs.keys())

            model_inputs.convert_to_tensors(tensor_type=self.return_tensors)
            if self.image_key:




                un_padded_features["multi_modal_inputs"].append(dict(model_inputs))

                un_padded_features["multi_modal_data"].append(
                    {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                        "multi_modal_data": {
                            "image": [feature[self.image_key]]
                            if not isinstance(feature[self.image_key], list)
                            else feature[self.image_key]
                        },
                    }
                    if not self.image_flag_key or feature[self.image_flag_key]
                    else {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_key], add_special_tokens=False),
                    }
                )
            if self.gt_mask_key:
                un_padded_features[self.gt_mask_key].append(feature[self.gt_mask_key])
            if self.gt_object_key:
                un_padded_features[self.gt_object_key].append(feature[self.gt_object_key])
            if self.gt_center_key:
                un_padded_features[self.gt_center_key].append(feature[self.gt_center_key])
            if self.sat_image_key:
                un_padded_features[self.sat_image_key].append(feature[self.sat_image_key])
            if self.id_key:
                un_padded_features[self.id_key].append(feature[self.id_key])
            if self.gt_bbox_key:
                un_padded_features[self.gt_bbox_key].append(feature[self.gt_bbox_key])
            if self.stage_flag_key:
                un_padded_features[self.stage_flag_key].append(feature[self.stage_flag_key])
            if self.prompt_key:
                un_padded_features[self.prompt_key].append(feature[self.prompt_key])


        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch.update(un_padded_features)





        if self.extra_data_provider:
            fun_params = inspect.signature(self.extra_data_provider).parameters
            kwargs = {}
            for key in fun_params:
                if key in batch:
                    kwargs[key] = batch[key]
                elif key in mm_feature_keys:
                    mm_inputs = [inputs[key] for inputs in batch["multi_modal_inputs"] if key in inputs]
                    kwargs[key] = torch.concat(mm_inputs, dim=0) if mm_inputs else fun_params[key].default
                else:
                    kwargs[key] = fun_params[key].default
            extra_data = self.extra_data_provider(**kwargs)
            batch.update(extra_data)



        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                assert batch[key].shape[0] == batch["input_ids"].shape[0]
            else:
                assert len(batch[key]) == batch["input_ids"].shape[0]
                temp_array = np.empty([len(batch[key])], dtype=object)
                temp_array[:] = batch[key]
                batch[key] = temp_array

        return batch


@dataclass
class DataCollatorWithPaddingForMultiSeg:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    processor: Optional[ProcessorMixin] = None
    extra_data_provider: Optional[callable] = None
    prompt_map_key: Optional[str] = "prompt_map"
    question_key: Optional[str] = "question"
    image_key: Optional[str] = None
    map_image_key: Optional[str] = "image_map"
    id_key: Optional[str] = "id"
    gt_mask_key: Optional[str] = "gt_mask"
    gt_point_key: Optional[str] = None
    seg_image_key: Optional[str] = "seg_image"
    gt_object_key: Optional[str] = None
    gt_center_key: Optional[str] = None
    gt_bbox_key: Optional[str] = None
    image_flag_key: Optional[str] = "image_flag"
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padded_keys: List[str] = field(default_factory=lambda: ["input_ids", "attention_mask", "labels"])
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.tokenizer and self.processor

        map_padded_features = defaultdict(list)
        un_padded_features = defaultdict(list)
        mm_feature_keys = set()
        for feature in features:





            map_model_inputs: BatchFeature = self.processor(
                images=feature[self.image_key]
                if self.image_key and (not self.image_flag_key or feature[self.image_flag_key])
                else None,
                text=feature[self.prompt_map_key] if self.prompt_map_key else None,
            )
            for key in ["prompt_map"]:
                if key in map_model_inputs:
                    map_model_inputs.pop(key)
            for key in filter(lambda k: k in map_model_inputs, self.padded_keys):
                map_padded_features[key].append(map_model_inputs.pop(key)[0])

            mm_feature_keys = mm_feature_keys.union(map_model_inputs.keys())

            map_model_inputs.convert_to_tensors(tensor_type=self.return_tensors)
            if self.image_key:




                un_padded_features["multi_modal_map_inputs"].append(dict(map_model_inputs))

                un_padded_features["multi_modal_map_data"].append(
                    {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_map_key], add_special_tokens=False)
                        if self.prompt_map_key else [],
                        "multi_modal_data": {
                            "image": [feature[self.image_key]]
                            if not isinstance(feature[self.image_key], list)
                            else feature[self.image_key]
                        },
                    }
                    if not self.image_flag_key or feature[self.image_flag_key]
                    else {
                        "prompt_token_ids":
                        self.tokenizer.encode(feature[self.prompt_map_key], add_special_tokens=False)
                        if self.prompt_map_key else [],
                    }
                )

            if self.question_key:
                un_padded_features[self.question_key].append(feature[self.question_key])
            if self.gt_mask_key:
                un_padded_features[self.gt_mask_key].append(feature[self.gt_mask_key])
            if self.gt_object_key:
                un_padded_features[self.gt_object_key].append(feature[self.gt_object_key])
            if self.gt_point_key:
                un_padded_features[self.gt_point_key].append(feature[self.gt_point_key])
            if self.gt_center_key:
                un_padded_features[self.gt_center_key].append(feature[self.gt_center_key])
            if self.seg_image_key:
                un_padded_features[self.seg_image_key].append(feature[self.seg_image_key])
            if self.map_image_key:
                un_padded_features[self.map_image_key].append(feature[self.map_image_key])
            if self.gt_bbox_key:
                un_padded_features[self.gt_bbox_key].append(feature[self.gt_bbox_key])
            if self.image_key:
                un_padded_features[self.image_key].append(feature[self.image_key])
            if self.id_key:
                un_padded_features[self.id_key].append(feature[self.id_key])


        map_batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            map_padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {}
        batch["map_input_ids"] = map_batch["input_ids"]
        batch["map_attention_mask"] = map_batch["attention_mask"]
        batch.update(un_padded_features)






        if self.extra_data_provider:
            map_fun_params = ['map_input_ids', 'map_attention_mask', 'image_grid_thw']
            map_kwargs = {}
            for key in map_fun_params:
                if key in batch:
                    map_kwargs[key] = batch[key]
                elif key in mm_feature_keys:
                    mm_inputs = [inputs[key] for inputs in batch["multi_modal_map_inputs"] if key in inputs]
                    map_kwargs[key] = torch.concat(mm_inputs, dim=0) if mm_inputs else map_fun_params[key].default
                else:
                    print(f"Warning: {key} not found in batch, using default value.")
                    exit()
            map_kwargs['input_ids'] = map_kwargs.pop('map_input_ids')
            map_kwargs['attention_mask'] = map_kwargs.pop('map_attention_mask')
            map_extra_data = self.extra_data_provider(**map_kwargs)
            map_extra_data['map_position_ids'] = map_extra_data.pop('position_ids')
            batch.update(map_extra_data)



        for key in batch:
            if isinstance(batch[key], (torch.Tensor, np.ndarray)):
                assert batch[key].shape[0] == batch["map_input_ids"].shape[0]
            else:
                assert len(batch[key]) == batch["map_input_ids"].shape[0]
                tem_array = np.empty([len(batch[key])], dtype=object)
                tem_array[:] = batch[key]
                batch[key] = tem_array
        return batch
