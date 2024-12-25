# 匯入必要套件
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

# 確保輸出目錄存在
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 設定裝置 (GPU 或 CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# 1. 翻譯中文到英文
def translate_text(text):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    translated_text = translator(text)[0]['translation_text']
    print(f"Translated Text: {translated_text}")
    return translated_text

# 2. 圖像生成
def generate_image(prompt, output_file):
    # 使用 Stable Diffusion 模型
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    image = pipe(prompt).images[0]

    # 保存圖像
    image_path = os.path.join(output_dir, output_file)
    image.save(image_path)
    print(f"Image saved to {image_path}")
    return image_path

# 主流程
def main():
    # 輸入中文句子
    chinese_text = input("輸入中文句子: ")  # 例如：一隻狗和一隻貓
    translated_text = translate_text(chinese_text)

    # 根據翻譯結果生成圖像
    output_file = "generated_image.png"
    generate_image(translated_text, output_file)

    # 顯示結果
    img = Image.open(os.path.join(output_dir, output_file))
    img.show()

if __name__ == "__main__":
    main()
