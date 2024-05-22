# -*- coding: utf-8 -*-
# @Time    :   2024/05/21 16:13:14
# @Author  :   lixumin1030@gmail.com
# @FileName:   lightning_module.py

import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, AutoTokenizer
from utils import json2token, token2json


class GeoVIEModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model_config = VisionEncoderDecoderConfig.from_pretrained(
            self.config.pretrained_model_name_or_path,
        )
        
        self.model_config.encoder.image_size = self.config.input_size # [512, 512]

        
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            config=self.model_config
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name_or_path)
        

        self.pytorch_lightning_version_is_1 = int(pl.__version__[0]) < 2
        self.num_of_loaders = len(self.config.dataset_name_or_paths)

        # ### TODO
        # self.model.decoder.prepare_inputs_for_generation = prepare_inputs_for_inference

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, decoder_labels = list(), list(), list()

        for batch_data in batch:
            # print(batch_data[1].shape)
            image_tensors.append(batch_data[0])
            decoder_input_ids.append(batch_data[1])
            decoder_labels.append(batch_data[2])
        image_tensors = torch.cat(image_tensors)
        decoder_input_ids = torch.cat(decoder_input_ids)
        decoder_labels = torch.cat(decoder_labels)
        loss = self.model(image_tensors, decoder_input_ids, labels=decoder_labels).loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        if not self.pytorch_lightning_version_is_1:
            self.log('loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.validation_step_outputs = [[] for _ in range(self.num_of_loaders)]
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image_tensors, decoder_input_ids, prompt_tensors, answers = batch

        # decoder_prompts = pad_sequence(
        #     [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_tensors)],
        #     batch_first=True,
        # ) 
        
        print(prompt_tensors.shape)
        print(image_tensors.shape)
        # TODO
        preds = self.model.generate(
            image_tensors, 
            decoder_input_ids=prompt_tensors,
            max_length=self.config.max_length, 
            num_beams=1,
        )
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        scores = []
        for pred, answer in zip(preds, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                self.print(f"Prediction: {pred}")
                self.print(f"    Answer: {answer}")
                self.print(f" Normed ED: {scores[0]}")

        self.validation_step_outputs[dataloader_idx].append(scores)

        return scores

    def on_validation_epoch_end(self):
        assert len(self.validation_step_outputs) == self.num_of_loaders
        cnt = [0] * self.num_of_loaders
        total_metric = [0] * self.num_of_loaders
        val_metric = [0] * self.num_of_loaders
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)

    def configure_optimizers(self):

        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert len(self.config.train_batch_sizes) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.config.max_epochs * self.config.num_training_samples_per_epoch) / (
                self.config.train_batch_sizes[0] * torch.cuda.device_count() * self.config.get("num_nodes", 1)
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = min(self.config.max_steps, max_iter) if max_iter is not None else self.config.max_steps

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, nesterov=True)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.config.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.result_path) / self.config.exp_name / self.config.exp_version
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


class GeoVIEDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                )
            )
        return loaders

    def val_dataloader(self):
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                )
            )
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

####################################################################################
# import pytorch_lightning as pl
# from transformers import VisionEncoderDecoderModel, AdamW
# class GeoVIEModule(pl.LightningModule):
#     def __init__(self, model_name, learning_rate=5e-5):
#         super().__init__()
#         self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
#         self.learning_rate = learning_rate

#     def forward(self, pixel_values, input_ids, attention_mask):
#         outputs = self.model(pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)
#         return outputs.loss

#     def training_step(self, batch, batch_idx):
#         pixel_values, input_ids, attention_mask = batch
#         loss = self.forward(pixel_values, input_ids, attention_mask)
#         self.log('train_loss', loss)
#         return loss

#     def configure_optimizers(self):
#         return AdamW(self.parameters(), lr=self.learning_rate)
    
#     def save_model(self, save_path):
#         self.model.save_pretrained(save_path)
#         self.tokenizer.save_pretrained(save_path)
#         self.feature_extractor.save_pretrained(save_path)

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torch.utils.data import DataLoader
# from transformers import ViTFeatureExtractor, AutoTokenizer

# # 参数设置
# img_folder = "/path/to/image/folder"
# captions_file = "/path/to/captions/file"
# batch_size = 8
# epochs = 5
# model_save_path = "/path/to/save/model"

# # 加载模型、特征提取器和分词器
# model_name = "/store/lixumin/xizhi_OCR/nougat_ocr/workspace/latex_ocr_mini_240130/"
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 创建数据集和数据加载器
# dataset = CustomDataset(img_folder, captions_file, feature_extractor, tokenizer)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 定义模型
# model = ImageCaptioningModel(model_name=model_name)

# # 定义ModelCheckpoint回调
# checkpoint_callback = ModelCheckpoint(
#     dirpath=model_save_path,
#     filename="{epoch}-{train_loss:.2f}",
#     save_top_k=-1,  # 保存所有检查点
#     verbose=True,
#     monitor="train_loss",
#     mode="min",
#     save_weights_only=False
# )

# # 定义训练器
# trainer = pl.Trainer(
#     max_epochs=epochs,
#     gpus=1 if torch.cuda.is_available() else 0,
#     callbacks=[checkpoint_callback]
# )

# # 开始训练
# trainer.fit(model, dataloader)