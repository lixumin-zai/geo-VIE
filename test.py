from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderConfig
from PIL import Image
import requests
import torch
from torchvision import transforms
# 加载预训练的模型
model_name = "/store/lixumin/xizhi_OCR/nougat_ocr/workspace/latex_ocr_mini_240130/"
# model_name = '/store/lixumin/xizhi_OCR/nougat_ocr/pretrain_model/small_model'
model = VisionEncoderDecoderModel.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数的总大小: {total_params} 个参数")
total_params_in_MB = total_params * 4 / (1024 ** 2)  # assuming 4 bytes per parameter (float32)
print(f"模型参数的总大小: {total_params_in_MB:.2f} MB")
model_config = VisionEncoderDecoderConfig.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载并预处理图像
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image
preprocess = transforms.Compose([
    # transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为Tensor
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])
image_path = "/home/lixumin/project/xizhi_OCR/xizhi-latex-beta/show.jpg"
image = load_image(image_path)

# image = preprocess(image).unsqueeze(0) / 255.0
# image = image.to(device)
# pixel_values = model.encoder(pixel_values=image).pixel_values

pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
print(pixel_values.shape)
pixel_values = pixel_values.to(device)



target_text = "your target text"
target_ids = tokenizer([target_text,target_text], return_tensors="pt").input_ids.to(device)

# 生成描述
max_length = 256
num_beams = 1

output_ids = model(
    pixel_values, 
    target_ids,
    labels=target_ids
)

# 准备prompt
prompt = "HasdA"
prompt_tensors = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
prompt_tensors = prompt_tensors.to(device)
print(prompt_tensors.shape)
output_ids = model.generate(
    torch.cat([pixel_values, pixel_values]), 
    max_length=max_length, 
    num_beams=num_beams,
    decoder_input_ids=prompt_tensors.repeat(2, 1)
)
print(output_ids.shape)
print(output_ids)
caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print("result:", caption)