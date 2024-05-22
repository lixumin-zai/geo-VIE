# -*- coding: utf-8 -*-
# @Time    :   2024/05/21 11:51:48
# @Author  :   lixumin1030@gmail.com
# @FileName:   geo-IE.py
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch

class GeoVIE(VisionEncoderDecoderModel):
    def __init__(self, encoder_model_name, decoder_model_name):
        super().__init__(encoder_model_name, decoder_model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

    def inference(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Generate features using the encoder
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        outputs = self.generate(pixel_values=pixel_values)

        # Decode the output
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output