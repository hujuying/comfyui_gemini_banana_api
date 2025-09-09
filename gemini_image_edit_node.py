import requests
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
import folder_paths
import re
import os

class GeminiImageEditNode:
    """
    ComfyUI节点：使用Gemini API进行图像编辑（支持多API Key轮询和密钥文件存储）
    支持1-3张图片输入和文本提示
    """
    
    def __init__(self):
        self.key_index = 0  # 用于轮询的索引
        self.plugin_dir = os.path.dirname(__file__)
        self.key_file = os.path.join(self.plugin_dir, "api_keys.txt")
    
    def load_keys_from_file(self):
        """从文件加载API密钥"""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return self.parse_api_keys(content)
        except Exception as e:
            print(f"[GeminiImageEditNode] 读取密钥文件失败: {e}")
        return []
    
    def save_keys_to_file(self, keys):
        """保存API密钥到文件"""
        try:
            with open(self.key_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(keys))
            print(f"[GeminiImageEditNode] API密钥已保存到文件")
        except Exception as e:
            print(f"[GeminiImageEditNode] 保存密钥文件失败: {e}")
    
    def parse_api_keys(self, api_keys_str):
        """解析API Keys，支持逗号或换行分隔"""
        if not api_keys_str:
            return []
        
        # 按换行符和逗号分割，然后清理每个key
        keys = re.split(r'[,，\n\r]+', api_keys_str)
        # 过滤空字符串并去除空白
        keys = [key.strip() for key in keys if key.strip()]
        return keys
    
    def get_api_keys(self, input_keys):
        """获取API密钥：优先使用输入的，否则使用文件中的"""
        # 解析输入的密钥
        input_keys_list = self.parse_api_keys(input_keys)
        
        if input_keys_list:
            # 如果有输入密钥，保存到文件
            self.save_keys_to_file(input_keys_list)
            return input_keys_list
        else:
            # 如果没有输入密钥，从文件加载
            file_keys = self.load_keys_from_file()
            if file_keys:
                return file_keys
            else:
                print("[GeminiImageEditNode] 未找到API密钥，请在输入框中输入或确保api_keys.txt文件存在")
                return []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit this image according to my request"
                }),
                "api_keys": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "base_url": ("STRING", {
                    "default": ""
                }),
                "model": ("STRING", {
                    "default": "gemini-2.5-flash-image-preview"
                }),
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 300
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 4000
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_images"
    CATEGORY = "image/AI"
    
    def tensor_to_base64(self, tensor):
        """
        将ComfyUI的tensor图像转换为base64字符串
        """
        # 转换tensor到PIL Image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除batch维度
        
        # 转换到0-255范围
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        tensor = tensor.clamp(0, 255).byte()
        
        # 转换为numpy数组
        numpy_image = tensor.cpu().numpy()
        
        # 创建PIL Image
        pil_image = Image.fromarray(numpy_image, mode='RGB')
        
        # 转换为base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        img_bytes = buffer.getvalue()
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_string
    
    def base64_to_tensor(self, base64_string):
        """
        将base64字符串转换为ComfyUI tensor
        """
        try:
            # 清理base64字符串
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # 解码base64
            img_bytes = base64.b64decode(base64_string)
            
            # 转换为PIL Image
            pil_image = Image.open(io.BytesIO(img_bytes))
            pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 转换为tensor
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)  # 添加batch维度
            
            return tensor
            
        except Exception as e:
            print(f"Error converting base64 to tensor: {e}")
            return None
    
    def create_gemini_request(self, prompt, images_base64, model):
        """
        创建Gemini API请求格式
        """
        parts = [{"text": prompt}]
        
        # 添加图像数据
        for i, img_base64 in enumerate(images_base64):
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_base64
                }
            })
        
        return {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
    
    def convert_to_openai_format(self, gemini_data, model):
        """
        将Gemini格式转换为OpenAI格式
        """
        messages = []
        
        for content in gemini_data.get("contents", []):
            role = content.get("role", "user")
            message_content = []
            
            for part in content.get("parts", []):
                if "text" in part:
                    message_content.append({
                        "type": "text",
                        "text": part["text"]
                    })
                elif "inline_data" in part:
                    inline_data = part["inline_data"]
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{inline_data['mime_type']};base64,{inline_data['data']}"
                        }
                    })
            
            messages.append({
                "role": role,
                "content": message_content
            })
        
        return {
            "model": model,
            "messages": messages,
            "max_tokens": 1000
        }
    
    def extract_image_from_response(self, response_text):
        """
        从API响应中提取base64图像数据
        """
        import re
        
        # 匹配各种可能的base64图像格式
        patterns = [
            r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)',  # 完整的data URL
            r'([A-Za-z0-9+/=]{100,})',  # 长的base64字符串
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                return matches[0]  # 返回第一个匹配的图像
        
        return None
    
    def edit_images(self, prompt, api_keys, base_url, model, image1, 
                   image2=None, image3=None, timeout=60, max_tokens=1000):
        """
        主要的图像编辑函数（支持API Key轮询和密钥文件存储）
        """
        try:
            # 获取API Keys
            keys = self.get_api_keys(api_keys)
            if not keys:
                print("No valid API keys found")
                return (image1,)
            
            print(f"[GeminiImageEditNode] 使用 {len(keys)} 个API密钥")
            
            # 循环尝试所有API Keys
            for attempt in range(len(keys)):
                current_key = keys[self.key_index]
                # 隐藏显示实际的密钥值
                display_key = current_key[:7] + "..." if len(current_key) > 10 else "***"
                print(f"[GeminiImageEditNode] 尝试使用API Key: {display_key}")
                
                try:
                    # 收集所有输入的图像
                    images = [image1]
                    if image2 is not None:
                        images.append(image2)
                    if image3 is not None:
                        images.append(image3)
                    
                    print(f"Processing {len(images)} image(s) with Gemini API...")
                    
                    # 转换图像为base64
                    images_base64 = []
                    for img in images:
                        base64_img = self.tensor_to_base64(img)
                        images_base64.append(base64_img)
                    
                    # 创建API请求数据
                    gemini_data = self.create_gemini_request(prompt, images_base64, model)
                    openai_data = self.convert_to_openai_format(gemini_data, model)
                    openai_data["max_tokens"] = max_tokens
                    
                    # 发送API请求
                    response = requests.post(
                        f"{base_url}/v1/chat/completions",
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {current_key}'
                        },
                        json=openai_data,
                        timeout=timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print("API call successful!")
                        
                        # 提取图像数据
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            base64_image = self.extract_image_from_response(content)
                            
                            if base64_image:
                                # 转换为tensor
                                output_tensor = self.base64_to_tensor(base64_image)
                                if output_tensor is not None:
                                    print("Image successfully generated and converted!")
                                    # 重置key索引以便下次从第一个开始
                                    self.key_index = 0
                                    return (output_tensor,)
                                else:
                                    print("Failed to convert base64 to tensor")
                            else:
                                print("No image found in API response")
                                print("Response content:", content[:500] + "..." if len(content) > 500 else content)
                        else:
                            print("Invalid API response structure")
                            print("Response:", json.dumps(result, indent=2)[:1000])
                        
                        # 如果解析失败但请求成功，继续下一个key
                        self.key_index = (self.key_index + 1) % len(keys)
                        continue
                    
                    elif response.status_code in [401, 403]:  # API Key相关错误
                        print(f"API Key access denied or invalid")
                        self.key_index = (self.key_index + 1) % len(keys)
                        continue
                    
                    else:
                        print(f"API call failed with status code: {response.status_code}")
                        print(f"Error: {response.text}")
                        # 对于其他错误，也尝试下一个key
                        self.key_index = (self.key_index + 1) % len(keys)
                        continue
                        
                except Exception as e:
                    print(f"Error with API Key: {e}")
                    self.key_index = (self.key_index + 1) % len(keys)
                    continue
            
            # 如果所有keys都失败，返回原始图像
            print("All API keys failed. Returning original image.")
            self.key_index = 0  # 重置索引
            return (image1,)
            
        except Exception as e:
            print(f"Error in edit_images: {e}")
            self.key_index = 0  # 重置索引
            return (image1,)

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiImageEditNode": GeminiImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEditNode": "Gemini Image Edit"
}
