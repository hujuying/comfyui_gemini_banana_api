import http.client
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
import requests
import re
import ssl
from urllib.parse import urlparse
import os

class GeminiImageEditV2Node:
    """
    ComfyUI节点：使用Gemini API进行图像编辑 V2（支持多API Key轮询和密钥文件存储）
    改进的图像URL检测和下载功能
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
            print(f"[GeminiImageEditV2Node] 读取密钥文件失败: {e}")
        return []
    
    def save_keys_to_file(self, keys):
        """保存API密钥到文件"""
        try:
            with open(self.key_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(keys))
            print(f"[GeminiImageEditV2Node] API密钥已保存到文件")
        except Exception as e:
            print(f"[GeminiImageEditV2Node] 保存密钥文件失败: {e}")
    
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
                print("[GeminiImageEditV2Node] 未找到API密钥，请在输入框中输入或确保api_keys.txt文件存在")
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
                    "default": "apis.kuai.host"
                }),
                "model": ("STRING", {
                    "default": "gemini-2.5-flash-image-hd"
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
    
    def tensor_to_image_url(self, tensor):
        """
        将ComfyUI的tensor图像转换为临时URL或base64格式
        这里我们直接转换为base64 data URL格式
        """
        try:
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
            
            # 返回data URL格式
            return f"data:image/jpeg;base64,{base64_string}"
            
        except Exception as e:
            print(f"Error converting tensor to image URL: {e}")
            return None
    
    def url_to_tensor(self, image_url):
        """
        从URL下载图像并转换为ComfyUI tensor
        """
        try:
            print(f"Downloading image from URL: {image_url}")
            
            # 下载图像
            response = requests.get(image_url, timeout=30, verify=False)
            response.raise_for_status()
            
            # 转换为PIL Image
            pil_image = Image.open(io.BytesIO(response.content))
            pil_image = pil_image.convert('RGB')
            
            print(f"Downloaded image size: {pil_image.size}")
            
            # 转换为numpy数组
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 转换为tensor
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)  # 添加batch维度
            
            return tensor
            
        except Exception as e:
            print(f"Error downloading/converting image from URL: {e}")
            return None
    
    def extract_image_urls_from_response(self, response_text):
        """
        从API响应中提取所有可能的图像URL
        支持多种格式的图像链接
        """
        image_urls = []
        
        # 更全面的URL匹配模式
        url_patterns = [
            # 标准图像URL
            r'https?://[^\s\)\]\}\"\']+\.(?:webp|jpg|jpeg|png|gif|bmp|tiff)',
            # Markdown格式的图像
            r'!\[.*?\]\((https?://[^\s\)]+\.(?:webp|jpg|jpeg|png|gif|bmp|tiff))\)',
            # HTML img标签
            r'<img[^>]+src=["\']([^"\']+\.(?:webp|jpg|jpeg|png|gif|bmp|tiff))["\'][^>]*>',
            # 任何以http开头，以图像扩展名结尾的URL
            r'(https?://[^\s\)\]\}\"\'<>]+\.(?:webp|jpg|jpeg|png|gif|bmp|tiff))',
            # 不带扩展名但可能是图像的URL（如google.datas.systems的链接）
            r'(https?://[^\s\)\]\}\"\'<>]*(?:image|img|photo|pic)[^\s\)\]\}\"\'<>]*)',
            # 特定域名的图像链接
            r'(https?://(?:filesystem\.site|google\.datas\.systems|[^/]+)/[^\s\)\]\}\"\'<>]*\.(?:webp|jpg|jpeg|png|gif|bmp|tiff))',
            r'(https?://(?:filesystem\.site|google\.datas\.systems)/[^\s\)\]\}\"\'<>]+)',
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    url = match[0] if match[0] else match[1] if len(match) > 1 else match[-1]
                else:
                    url = match
                
                if url and url not in image_urls:
                    image_urls.append(url)
        
        return image_urls
    
    def create_openai_request(self, prompt, image_urls, model, max_tokens):
        """
        创建OpenAI格式的API请求
        """
        content = [{"type": "text", "text": prompt}]
        
        # 添加图像URL
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        
        return {
            "model": model,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": max_tokens
        }
    
    def call_api_with_http_client(self, host, path, payload, headers, timeout):
        """
        使用http.client调用API
        """
        try:
            # 创建SSL上下文
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # 建立连接
            conn = http.client.HTTPSConnection(host, timeout=timeout, context=context)
            
            # 发送请求
            conn.request("POST", path, payload, headers)
            
            # 获取响应
            res = conn.getresponse()
            data = res.read()
            
            # 关闭连接
            conn.close()
            
            return res.status, data.decode("utf-8")
            
        except Exception as e:
            print(f"HTTP client error: {e}")
            return None, str(e)
    
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
            
            print(f"[GeminiImageEditV2Node] 使用 {len(keys)} 个API密钥")
            
            # 循环尝试所有API Keys
            for attempt in range(len(keys)):
                current_key = keys[self.key_index]
                # 隐藏显示实际的密钥值
                display_key = current_key[:7] + "..." if len(current_key) > 10 else "***"
                print(f"[GeminiImageEditV2Node] 尝试使用API Key: {display_key}")
                
                try:
                    # 收集所有输入的图像
                    images = [image1]
                    if image2 is not None:
                        images.append(image2)
                    if image3 is not None:
                        images.append(image3)
                    
                    print(f"Processing {len(images)} image(s) with Gemini API V2...")
                    
                    # 转换图像为URL格式
                    image_urls = []
                    for i, img in enumerate(images):
                        img_url = self.tensor_to_image_url(img)
                        if img_url:
                            image_urls.append(img_url)
                            print(f"Converted image {i+1} to data URL")
                        else:
                            print(f"Failed to convert image {i+1}")
                    
                    if not image_urls:
                        print("No valid images to process")
                        self.key_index = (self.key_index + 1) % len(keys)
                        continue
                    
                    # 创建API请求数据
                    request_data = self.create_openai_request(prompt, image_urls, model, max_tokens)
                    payload = json.dumps(request_data)
                    
                    # 设置请求头
                    headers = {
                        'Authorization': f'Bearer {current_key}',
                        'Content-Type': 'application/json'
                    }
                    
                    # 解析base_url
                    if base_url.startswith('http://') or base_url.startswith('https://'):
                        parsed_url = urlparse(base_url)
                        host = parsed_url.netloc
                        path = parsed_url.path if parsed_url.path else "/v1/chat/completions"
                    else:
                        host = base_url
                        path = "/v1/chat/completions"
                    
                    print(f"Calling API: {host}{path}")
                    
                    # 调用API
                    status_code, response_text = self.call_api_with_http_client(
                        host, path, payload, headers, timeout
                    )
                    
                    if status_code == 200:
                        print("API call successful!")
                        
                        try:
                            result = json.loads(response_text)
                            
                            # 提取响应内容
                            if 'choices' in result and len(result['choices']) > 0:
                                content = result['choices'][0]['message']['content']
                                print(f"API Response content: {content[:200]}...")
                                
                                # 提取图像URL
                                image_urls = self.extract_image_urls_from_response(content)
                                
                                if image_urls:
                                    print(f"Found {len(image_urls)} image URL(s):")
                                    for i, url in enumerate(image_urls):
                                        print(f"  {i+1}. {url}")
                                    
                                    # 尝试下载第一个图像
                                    for url in image_urls:
                                        output_tensor = self.url_to_tensor(url)
                                        if output_tensor is not None:
                                            print("Image successfully downloaded and converted!")
                                            # 重置key索引
                                            self.key_index = 0
                                            return (output_tensor,)
                                        else:
                                            print(f"Failed to download image from: {url}")
                                else:
                                    print("No image URLs found in API response")
                                    print("Full response content:", content)
                            else:
                                print("Invalid API response structure")
                                print("Response:", response_text[:1000])
                                
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON response: {e}")
                            print("Raw response:", response_text[:500])
                        
                        # 如果解析失败但请求成功，继续下一个key
                        self.key_index = (self.key_index + 1) % len(keys)
                        continue
                    
                    elif status_code in [401, 403]:  # API Key相关错误
                        print(f"API Key access denied or invalid")
                        self.key_index = (self.key_index + 1) % len(keys)
                        continue
                    
                    else:
                        print(f"API call failed with status code: {status_code}")
                        print(f"Error response: {response_text}")
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
            import traceback
            traceback.print_exc()
            self.key_index = 0  # 重置索引
            return (image1,)

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiImageEditV2Node": GeminiImageEditV2Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEditV2Node": "Gemini Image Edit V2"
}
