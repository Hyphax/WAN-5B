# WAN 2.2 TI2V-5B API

This repository contains a FastAPI service for the WAN 2.2 TI2V-5B model, a 5 billion parameter text-image-to-video generation model.

## Features

- **⚡ Ultra-Fast Generation**: Optimized for A100 80GB with 1-3 minute generation times
- **🎬 Text-to-Video Generation**: Generate videos from text prompts
- **🖼️➡️🎥 Text-Image-to-Video Generation**: Generate videos from text prompts and input images  
- **🚀 Speed Optimizations**: Torch compilation, Flash attention, Fast schedulers
- **📡 RESTful API**: Easy-to-use HTTP endpoints with speed modes
- **🐳 Docker Support**: Ready for containerized deployment
- **🔧 A100 Optimized**: Specifically tuned for NVIDIA A100 80GB hardware

## Model Information

- **Model**: Wan-AI/Wan2.2-TI2V-5B
- **Parameters**: 5 billion
- **Capabilities**: Text-to-video and text-image-to-video generation
- **Output Format**: MP4 videos

## API Endpoints

### POST /generate

Generate a video from text prompt and optional image.

### POST /generate/ultrafast ⚡

**NEW**: Ultra-fast generation optimized for A100 80GB with maximum speed settings.

**Request Body:**
```json
{
  "prompt": "A cat playing in the garden",
  "image": "base64_encoded_image_data", // Optional
  "width": 960,
  "height": 544,
  "num_frames": 25,
  "steps": 28,
  "guidance_scale": 4.0,
  "negative_prompt": "blurry, low quality", // Optional
  "seed": 42 // Optional
}
```

**Response:**
```json
{
  "job_id": "abc123...",
  "status": "running"
}
```

### GET /jobs/{job_id}

Check the status of a generation job.

### GET /result/{job_id}

Download the generated video (returns MP4 file).

### GET /model_info

Get information about the loaded model and its capabilities.

### GET /healthz

Health check endpoint.

## Usage Examples

### ⚡ Ultra-Fast Generation (RECOMMENDED for A100)
```bash
curl -X POST "http://localhost:8000/generate/ultrafast" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing in the garden",
    "steps": 12
  }'
```
**⏱️ Estimated time: 1-3 minutes**

### 🎬 Standard Text-to-Video Generation
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "width": 960,
    "height": 544,
    "num_frames": 25,
    "steps": 20,
    "enable_fast_mode": true
  }'
```
**⏱️ Estimated time: 2-5 minutes**

### 🖼️➡️🎥 Text-Image-to-Video Generation
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Make this image come alive with gentle movement",
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "width": 960,
    "height": 544,
    "num_frames": 25,
    "steps": 15,
    "enable_fast_mode": true
  }'
```
**⏱️ Estimated time: 2-4 minutes**

## Docker Deployment

### 🚀 Quick Deploy (A100 Optimized)
```bash
# Build
docker build -t wan-ti2v-5b .

# Run with A100 optimizations
docker run --gpus all -p 8000:8000 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
  wan-ti2v-5b
```

### 🔍 Pre-deployment Safety Check
```bash
# Test before deploying
docker run --gpus all wan-ti2v-5b python startup_test.py
```

## Environment Variables

- `WAN_MODEL_ID`: Model identifier (default: "Wan-AI/Wan2.2-TI2V-5B")
- `MODELS_DIR`: Directory for model cache (default: "/models")
- `OUT_DIR`: Directory for output videos (default: "/data/outputs")
- `WARMUP`: Enable model warmup on startup (default: "1")

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.8+ (for local development)

## 🚀 Speed Optimizations for A100 80GB

This repository is **heavily optimized for lightning-fast generation** on A100 hardware:

### ⚡ Core Speed Features
- **Model Compilation**: Torch compile with `max-autotune` mode
- **Flash Attention**: GPU-optimized attention mechanisms  
- **TF32 Precision**: Fastest precision for A100
- **CUDNN Benchmark**: Optimized for consistent workloads
- **Fast Schedulers**: DDIM scheduler for reduced steps
- **Ultra-Fast Mode**: `/generate/ultrafast` endpoint with 10-15 steps

### 📊 Performance Targets
| Mode | Steps | Time | Quality |
|------|-------|------|---------|
| **Ultra-Fast** | 10-15 | **1-3 min** | Good |
| **Standard Fast** | 20 | **2-5 min** | High |
| **Quality** | 30+ | **5-8 min** | Excellent |

### 🎛️ Speed vs Quality Control
```json
{
  "steps": 12,           // Lower = faster
  "enable_fast_mode": true,
  "guidance_scale": 7.0   // Lower = faster
}
```

## Changes from Original

This repository has been updated from the original WAN 2.2 T2V-A14B model to use the smaller, more efficient WAN 2.2 TI2V-5B model with the following improvements:

- **🏃‍♂️ Maximum Speed**: Optimized for 1-3 minute generation on A100 80GB
- **🧠 Reduced Model Size**: From 14B to 5B parameters 
- **🖼️ Image Input Support**: Added text-image-to-video generation capability
- **⚡ Speed Endpoints**: New `/generate/ultrafast` for maximum speed
- **🔧 A100 Tuning**: Hardware-specific optimizations for NVIDIA A100
- **📡 Enhanced API**: Multiple generation modes and performance info

## License

Please refer to the original model's license on Hugging Face: [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

