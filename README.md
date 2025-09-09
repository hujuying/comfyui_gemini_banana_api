# ComfyUI Gemini 图像编辑插件

[![GitHub](https://img.shields.io/github/license/yourusername/comfyui-gemini-image-edit)](https://github.com/yourusername/comfyui-gemini-image-edit/blob/main/LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-插件-blue)](https://github.com/comfyanonymous/ComfyUI)

一个强大的 ComfyUI 插件，支持使用 Gemini API 进行图像编辑和生成，具备多 API 密钥轮询和安全存储功能。

## 🌟 功能特点

- **Gemini 图像编辑**：支持使用 Gemini API 编辑 1-3 张图片
- **多 API 密钥支持**：当 API 限制达到时自动轮询多个 API 密钥
- **安全密钥存储**：自动将密钥保存到本地文件，避免明文显示
- **灵活的密钥输入**：支持逗号或换行分隔的 API 密钥
- **双版本支持**：提供两种不同 API 处理方式的节点版本
- **自动回退机制**：所有 API 密钥都失败时返回原始图像
- **高级图像解析**：从各种响应格式中提取图像

## 🚀 安装方法

1. 进入 ComfyUI 自定义节点目录：
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. 克隆此仓库：
   ```bash
   git clone https://github.com/yourusername/comfyui-gemini-image-edit.git
   ```

3. 安装依赖：
   ```bash
   pip install -r comfyui-gemini-image-edit/requirements.txt
   ```

4. 重启 ComfyUI

## 🔐 安全密钥管理

### 密钥存储机制

1. **首次使用**：在 `api_keys` 输入框中输入您的 API 密钥（支持逗号或换行分隔）
2. **自动保存**：插件会自动将密钥保存到 `api_keys.txt` 文件中
3. **后续使用**：输入框留空即可自动使用文件中保存的密钥
4. **更新密钥**：在输入框中输入新密钥将覆盖文件中的内容

### 密钥输入格式

支持以下格式：

**格式1：逗号分隔**
```
sk-key1,sk-key2,sk-key3
```

**格式2：换行分隔**
```
sk-key1
sk-key2
sk-key3
```

**格式3：混合分隔**
```
sk-key1,
sk-key2
sk-key3
```

## 📖 使用说明

### 节点类型

1. **Gemini Image Edit** - 标准版本，使用 requests 库
2. **Gemini Image Edit V2** - 增强版本，改进了图像 URL 检测功能

### 输入参数

- **prompt**：图像编辑的文本提示（支持多行）
- **api_keys**：API 密钥（首次使用时输入，后续可留空）
- **base_url**：API 端点 URL
- **model**：Gemini 模型名称（如 `gemini-2.5-flash-image-preview`）
- **image1/2/3**：输入图像（1 为必需，2 和 3 为可选）
- **timeout**：API 请求超时时间（10-300 秒）
- **max_tokens**：响应最大令牌数（100-4000）

### API 密钥轮询

当出现以下情况时，插件会自动轮询提供的 API 密钥：
- 达到速率限制
- API 密钥无效或过期
- 网络错误发生

## 🛠️ 配置示例

### 首次使用（输入密钥）
```
sk-your-first-key,
sk-your-second-key
sk-your-third-key
```

### 后续使用（输入框留空）
```
（留空）
```

### 工作流示例
```json
{
  "prompt": "增强颜色并添加更多细节",
  "api_keys": "",
  "base_url": "https://api.example.com",
  "model": "gemini-2.5-flash-image-preview",
  "timeout": 60,
  "max_tokens": 1000
}
```

## 📋 环境要求

- ComfyUI
- Python 3.8+
- requests >= 2.25.0
- pillow >= 8.0.0
- torch >= 1.9.0
- numpy >= 1.21.0

## 🔧 故障排除

### 常见问题

1. **找不到密钥文件**：确保插件目录有写入权限
2. **API 密钥错误**：检查 `api_keys.txt` 文件内容是否正确
3. **速率限制**：添加更多 API 密钥以进行轮询
4. **图像处理**：检查输入图像是否为支持的格式

### 日志查看
在 ComfyUI 控制台中查看详细的错误信息和 API 密钥轮询信息。

## 🤝 贡献代码

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 此仓库
2. 创建功能分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送到分支 (`git push origin feature/新功能`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 ComfyUI 社区提供的优秀框架
- 受各种 AI 图像编辑工具启发
- 本插件基于https://www.bilibili.com/video/BV1KeaRzhE8o/ 站长的插件魔改而成，特此鸣谢~
## 📞 支持

如有问题、疑问或反馈，请在 GitHub 上[提交 issue](https://github.com/yourusername/comfyui-gemini-image-edit/issues)。

---

**安全提醒**：此插件会将 API 密钥保存到本地文件，请确保您的计算机环境安全，避免密钥泄露。

```

