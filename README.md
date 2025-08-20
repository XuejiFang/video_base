# Video Base - 视频生成基础模型

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

3. 在浏览器中打开ComfyUI界面
4. 配置文本提示词和生成参数
5. 点击运行开始视频生成

## 示例视频展示

### 示例 1
<video src="./assets/media32.mp4" width="384" controls preload="metadata"></video>

### 示例 2
<video src="assets/media33.mp4" width="384" controls preload="metadata"></video>

### 示例 3
<video src="assets/media34.mp4" width="384" controls preload="metadata"></video>

### 示例 4
<video src="assets/media35.mp4" width="384" controls preload="metadata"></video>

### 示例 5
<video src="assets/media36.mp4" width="384" controls preload="metadata"></video>

### 示例 6
<video src="assets/media37.mp4" width="384" controls preload="metadata"></video>

### 示例 7
<video src="assets/media38.mp4" width="384" controls preload="metadata"></video>

### 示例 8
<video src="assets/media39.mp4" width="384" controls preload="metadata"></video>

### 示例 9
<video src="assets/media40.mp4" width="384" controls preload="metadata"></video>

### 示例 10
<video src="assets/media41.mp4" width="384" controls preload="metadata"></video>

### 示例 11
<video src="assets/media42.mp4" width="384" controls preload="metadata"></video>

### 示例 12
<video src="assets/media43.mp4" width="384" controls preload="metadata"></video>

### 示例 13
<video src="assets/media44.mp4" width="384" controls preload="metadata"></video>

### 示例 14
<video src="assets/media45.mp4" width="384" controls preload="metadata"></video>

### 示例 15
<video src="assets/media46.mp4" width="384" controls preload="metadata"></video>

### 示例 16
<video src="assets/media47.mp4" width="384" controls preload="metadata"></video>

### 示例 17
<video src="assets/media48.mp4" width="384" controls preload="metadata"></video>

### 示例 18
<video src="assets/media49.mp4" width="384" controls preload="metadata"></video>

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
