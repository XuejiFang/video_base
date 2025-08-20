# Video Base - 视频生成基础模型
<video src="https://jaggar-oss.oss-cn-shanghai.aliyuncs.com/video/video_base_250210.mp4" width="512" controls preload="metadata"></video>

## 项目简介

使用T2V-DiT模型进行文本到视频的生成。项目提供了完整的推理流程和示例视频展示。

## 功能特性

- 🎬 文本到视频生成
- 🚀 基于ComfyUI的易用界面
- 📱 支持多种视频格式输出
- 🎨 高质量视频生成效果

## 安装说明

### 1. 环境要求

- Python 3.8+
- ComfyUI
- 足够的GPU内存（推荐8GB+）

### 2. 模型下载

从 [ModelScope](https://modelscope.cn/models/HakimZJU/t2v_dit_dev_1) 下载模型文件，并保存到 `./models` 目录下。

### 3. 依赖安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 推理运行

1. 确保模型文件已正确放置在 `./models` 目录下
2. 启动ComfyUI应用：

```bash
python app.py
```

## 项目结构

```
video_base/
├── README.md          # 项目说明文档
├── app.py            # 主应用程序
├── models/           # 模型文件目录
├── assets/           # 示例视频目录
│   ├── media32.mp4
│   ├── media33.mp4
│   └── ...
└── requirements.txt  # 依赖包列表
```

## 技术架构

- **模型**: T2V-DiT (Text-to-Video Diffusion Transformer)
- **框架**: ComfyUI
- **后端**: Python
- **前端**: Web界面

---

**注意**: 请确保在使用本项目时遵守相关法律法规，不得生成违法或不当内容。
