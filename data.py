import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import PIL
import os
import tqdm
from utils import json2token, token2json
from typing import Any, List, Optional, Union
import json
from transform import train_transform
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random

def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)


def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)


class GeoDataset(Dataset):
    def __init__(
        self,
        module: None,
        dataset_name_or_path: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
    ):
        super().__init__()
        self.module = module
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = "<s>"
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        print(dataset_name_or_path, self.split)
        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in tqdm.tqdm(self.dataset):
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]
            new_data = [
                    task_start_token
                    + json2token(
                        self.module,
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=False,
                    )
                    + self.module.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            self.gt_token_sequences.append(
                new_data
            )
        print(self.gt_token_sequences[-10:])   

        newly_added_num = self.module.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set([self.task_start_token, self.prompt_end_token]))})
        if newly_added_num > 0:
            self.module.model.decoder.resize_token_embeddings(len(self.module.tokenizer))

        self.prompt_end_token_id = self.module.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:

        img = img.convert("RGB")
        img = Image.fromarray(train_transform(img))
        
        return self.to_tensor(img)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        img_path = sample["image"].replace("/root/cv/lixumin/cv-geometric-dataset/","/root/cv/lixumin/data/cv-geometric-dataset/")
        input_tensor = self.prepare_input(Image.open(img_path), random_padding=self.split == "train")

        # input_ids
        processed_parse = random.choice(self.gt_token_sequences[idx]) 
        input_ids = self.module.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.module.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            # print(labels)
            # labels[
            #     : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            # ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            # print(labels)
            return input_tensor, input_ids, labels
        else:
            prompt = "<s_line>"
            prompt_tensors = self.module.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
            # prompt_tensors = prompt_tensors.to(device)
            # prompt_end_index = torch.nonzero(
            #     input_ids == self.prompt_end_token_id
            # ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_tensors, processed_parse