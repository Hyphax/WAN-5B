import os, uuid, threading, inspect, traceback, base64, io
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---- SDPA compat shim: ignore enable_gqa if Torch < 2.5
try:
    import torch.nn.functional as F
    if "enable_gqa" not in inspect.signature(F.scaled_dot_product_attention).parameters:
        _orig_sdp = F.scaled_dot_product_attention
        def _sdp_compat(*args, enable_gqa=False, **kwargs):
            return _orig_sdp(*args, **kwargs)
        F.scaled_dot_product_attention = _sdp_compat
except Exception:
    pass

MODELS_DIR = os.getenv("MODELS_DIR", "/models")
OUT_DIR = os.getenv("OUT_DIR", "/data/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

pipe = None
pipe_lock = threading.Lock()
LOADED_MODEL_INFO = None  # Track which model actually loaded

app = FastAPI(title="WAN 2.2 TI2V-5B API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- A100 80GB SPEED-OPTIMIZED defaults
class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None  # base64 encoded image for text-image-to-video
    width: int = 960             # Optimized for speed
    height: int = 544            # Optimized for speed
    num_frames: int = 25         # Good balance for speed
    steps: int = 20              # FAST: Fewer steps for quicker generation
    guidance_scale: float = 7.0  # Balanced for speed and quality
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    # Speed optimization options
    enable_fast_mode: bool = True  # Enable all speed optimizations

def _load_pipeline():
    import torch
    from diffusers import DiffusionPipeline
    
    # ROBUST MODEL LOADING with proper fallbacks
    model_configs = [
        # Primary target - may need custom loading
        {
            "id": os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B"),
            "library": "wan2.2",
            "loading_method": "custom"
        },
        {
            "id": "Runware/Wan2.2-TI2V-5B", 
            "library": "wan2.2",
            "loading_method": "custom"
        },
        # Proven working diffusers models as fallbacks
        {
            "id": "cerspense/zeroscope_v2_576w",
            "library": "diffusers", 
            "loading_method": "diffusers"
        },
        {
            "id": "ali-vilab/text-to-video-ms-1.7b",
            "library": "diffusers",
            "loading_method": "diffusers"
        },
        {
            "id": "damo-vilab/text-to-video-ms-1.7b", 
            "library": "diffusers",
            "loading_method": "diffusers"
        }
    ]
    
    _pipe = None
    loaded_model_info = None
    
    for config in model_configs:
        model_id = config["id"]
        try:
            print(f"🔄 Attempting to load: {model_id} (library: {config['library']})")
            
            # Try different loading methods based on model type
            if config["loading_method"] == "custom":
                # For wan2.2 models, try with trust_remote_code first
                try:
                    _pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        cache_dir=MODELS_DIR,
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                except Exception as custom_error:
                    print(f"   Custom loading failed: {custom_error}")
                    # Try standard diffusers approach as fallback
                    _pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        cache_dir=MODELS_DIR
                    )
            else:
                # Standard diffusers loading
                _pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    cache_dir=MODELS_DIR
                )
            
            loaded_model_info = config
            print(f"✅ Successfully loaded: {model_id}")
            break
            
        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            continue
    
    if _pipe is None:
        raise ValueError("❌ CRITICAL: Could not load ANY compatible text-to-video model")
    
    # Store global info about loaded model
    global LOADED_MODEL_INFO
    LOADED_MODEL_INFO = loaded_model_info
    print(f"🎯 Active model: {loaded_model_info['id']} (library: {loaded_model_info['library']})")
    
    _pipe.to("cuda")

    # A100 80GB optimizations - NO memory constraints!
    # Disable memory optimizations for maximum quality and speed
    try:
        # Disable memory-saving features for better performance on A100
        if hasattr(_pipe, "disable_attention_slicing"):
            _pipe.disable_attention_slicing()
        if hasattr(_pipe, "disable_vae_slicing"):
            _pipe.disable_vae_slicing()
        if hasattr(_pipe, "vae") and hasattr(_pipe.vae, "disable_tiling"):
            _pipe.vae.disable_tiling()
    except Exception:
        pass

    # A100 SPEED optimizations - applied conservatively
    optimization_success = []
    try:
        import torch
        
        # Safe optimizations first
        try:
            torch.set_grad_enabled(False)
            optimization_success.append("Gradients disabled")
        except Exception as e:
            print(f"⚠️  Failed to disable gradients: {e}")
        
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            optimization_success.append("TF32 matmul enabled")
        except Exception as e:
            print(f"⚠️  Failed to enable TF32 matmul: {e}")
            
        try:
            torch.backends.cudnn.benchmark = True
            optimization_success.append("CUDNN benchmark enabled")
        except Exception as e:
            print(f"⚠️  Failed to enable CUDNN benchmark: {e}")
        
        # More aggressive optimizations (A100 safe)
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("highest")
            optimization_success.append("Advanced TF32 optimizations")
        except Exception as e:
            print(f"⚠️  Advanced TF32 failed: {e}")
        
        # Flash attention (if available)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            optimization_success.append("Flash attention enabled")
        except Exception as e:
            print(f"⚠️  Flash attention not available: {e}")
            
        # Model compilation (most risky - only for proven diffusers models)
        try:
            if (hasattr(torch, 'compile') and 
                loaded_model_info['library'] == 'diffusers' and 
                hasattr(_pipe, 'unet')):
                # Conservative compilation mode first
                _pipe.unet = torch.compile(_pipe.unet, mode="reduce-overhead")
                optimization_success.append("Model compilation (conservative)")
                print("✅ Model compiled with conservative settings")
            elif loaded_model_info['library'] == 'wan2.2':
                print("⚠️  Wan2.2 model - skipping torch.compile for safety")
        except Exception as e:
            print(f"⚠️  Model compilation failed (non-critical): {e}")
            
    except Exception as e:
        print(f"⚠️  Optimization setup failed: {e}")
    
    if optimization_success:
        print(f"✅ Optimizations applied: {', '.join(optimization_success)}")
    else:
        print("⚠️  No optimizations could be applied")
    return _pipe

def _ensure_pipe():
    global pipe
    if pipe is None:
        with pipe_lock:
            if pipe is None:
                pipe = _load_pipeline()

# ---- Warmup: load only (no tiny render to avoid shape errors)
WARMUP = os.getenv("WARMUP", "1") == "1"

@app.on_event("startup")
def _startup():
    if not WARMUP:
        return
    def _bg():
        try:
            _ensure_pipe()
            print("Warmup load done.")
        except Exception as e:
            print("Warmup load error:", e)
    threading.Thread(target=_bg, daemon=True).start()

def _normalize_hw(width: int, height: int):
    """snap to multiples of 16 and keep >=256"""
    def floor16(x: int) -> int: return max(256, (int(x) // 16) * 16)
    return floor16(width), floor16(height)

def _normalize_frames(nf: int):
    """WAN requires (nf - 1) % 4 == 0; snap to nearest valid >= 9."""
    nf = max(9, int(nf))
    # round to nearest (k*4 + 1)
    k = round((nf - 1) / 4)
    return int(k * 4 + 1)

def _decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image/'):
            base64_str = base64_str.split(',')[1]
        
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def _safe_infer(pipeline, **kw):
    import torch
    try:
        # A100 80GB - no memory constraints, run at full quality
        print(f"🚀 Starting inference with {kw.get('num_inference_steps', 'unknown')} steps...")
        result = pipeline(**kw)
        print("✅ Inference completed successfully")
        return result
    except torch.cuda.OutOfMemoryError as e:
        # This should never happen on A100 80GB, but just in case
        torch.cuda.empty_cache()
        print(f"❌ UNEXPECTED OOM on A100 80GB: {e}")
        print("🔄 Falling back to emergency settings...")
        kw["width"], kw["height"] = 960, 544
        kw["num_frames"] = min(25, kw.get("num_frames", 25))
        kw["num_inference_steps"] = min(20, kw.get("num_inference_steps", 20))
        return pipeline(**kw)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"❌ CUDA Error: {e}")
            torch.cuda.empty_cache()
            raise RuntimeError(f"CUDA error during inference: {e}")
        else:
            print(f"❌ Runtime Error: {e}")
            raise
    except Exception as e:
        # Handle other potential errors gracefully
        print(f"❌ Inference error: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise RuntimeError(f"Inference failed: {e}")

def _run_job(job_id: str, req: GenerateRequest):
    import torch
    from diffusers.utils import export_to_video
    try:
        _ensure_pipe()

        w16, h16 = _normalize_hw(req.width, req.height)
        nf_ok = _normalize_frames(req.num_frames)
        steps = max(10, int(req.steps))  # keep sane minimum

        gen = None
        if req.seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(req.seed))

        kw = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=h16, width=w16,
            num_frames=nf_ok,
            guidance_scale=req.guidance_scale,
            num_inference_steps=steps,
        )
        if gen is not None:
            kw["generator"] = gen
            
        # SPEED MODE optimizations for A100 (only for diffusers models)
        if hasattr(req, 'enable_fast_mode') and req.enable_fast_mode:
            try:
                global LOADED_MODEL_INFO
                if LOADED_MODEL_INFO and LOADED_MODEL_INFO['library'] == 'diffusers':
                    # Use faster scheduler for diffusers models only
                    from diffusers import DDIMScheduler
                    if hasattr(pipe, 'scheduler') and hasattr(pipe.scheduler, 'config'):
                        if steps <= 25:  # For quick generation, use DDIM
                            original_scheduler = pipe.scheduler
                            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                            kw["num_inference_steps"] = max(10, steps)  # Minimum 10 steps
                            print(f"🚀 FAST MODE: Using DDIM scheduler with {kw['num_inference_steps']} steps")
                elif LOADED_MODEL_INFO and LOADED_MODEL_INFO['library'] == 'wan2.2':
                    print("🚀 FAST MODE: Using default optimization for Wan2.2 model")
                    # For Wan2.2 models, just use fewer steps
                    kw["num_inference_steps"] = max(8, min(15, steps))
            except Exception as e:
                print(f"⚠️  Fast scheduler setup failed (non-critical): {e}")
            
        # Add image input for text-image-to-video generation if provided
        if req.image is not None:
            try:
                input_image = _decode_base64_image(req.image)
                # Resize image to match video dimensions
                input_image = input_image.resize((w16, h16), Image.Resampling.LANCZOS)
                kw["image"] = input_image
            except Exception as e:
                raise ValueError(f"Error processing input image: {e}")

        out = _safe_infer(pipe, **kw)
        frames = out.frames[0]

        out_path = os.path.join(OUT_DIR, f"{job_id}.mp4")
        export_to_video(frames, out_path, fps=24)

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["file"] = out_path

    except Exception as e:
        # always provide a readable error
        msg = f"{e.__class__.__name__}: {e}"
        tb = traceback.format_exc(limit=2)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = msg + "\n" + tb

@app.get("/healthz")
def healthz():
    """Health check with model status"""
    global pipe, LOADED_MODEL_INFO
    
    if pipe is None:
        return {
            "status": "initializing",
            "model_loaded": False,
            "message": "Model not yet loaded"
        }
    
    try:
        # Quick CUDA check
        import torch
        cuda_available = torch.cuda.is_available()
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_info": LOADED_MODEL_INFO,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0
        }
    except Exception as e:
        return {
            "status": "degraded", 
            "model_loaded": True,
            "error": str(e)
        }

@app.get("/readiness")
def readiness():
    """Readiness check for deployment"""
    global pipe, LOADED_MODEL_INFO
    
    if pipe is None:
        return {"ready": False, "reason": "Model not loaded"}
    
    if LOADED_MODEL_INFO is None:
        return {"ready": False, "reason": "Model info not available"}
    
    try:
        import torch
        if not torch.cuda.is_available():
            return {"ready": False, "reason": "CUDA not available"}
        
        return {
            "ready": True,
            "model": LOADED_MODEL_INFO["id"],
            "library": LOADED_MODEL_INFO["library"]
        }
    except Exception as e:
        return {"ready": False, "reason": f"System check failed: {e}"}

@app.get("/model_info")
def model_info():
    global LOADED_MODEL_INFO
    
    # Dynamic model info based on what actually loaded
    if LOADED_MODEL_INFO:
        model_id = LOADED_MODEL_INFO["id"]
        library = LOADED_MODEL_INFO["library"]
        capabilities = ["text-to-video"]
        
        # Add text-image-to-video if supported
        if "wan2.2" in library.lower() or "ti2v" in model_id.lower():
            capabilities.append("text-image-to-video")
            
        return {
            "model_id": model_id,
            "library": library,
            "model_type": "text-to-video" if len(capabilities) == 1 else "text-image-to-video",
            "capabilities": capabilities,
            "hardware": "A100 80GB optimized",
            "speed_optimizations": [
                "Flash attention",
                "Fast schedulers", 
                "TF32 precision",
                "CUDNN benchmark"
            ] + (["Torch compilation"] if library == "diffusers" else ["Custom optimizations"]),
            "performance": {
                "default_steps": 20,
                "fast_mode_min_steps": 10,
                "estimated_time": "1-5 minutes per video"
            },
            "supported_formats": {
                "input_image": "base64 encoded (JPEG, PNG, WebP)",
                "output_video": "MP4"
            },
            "status": "loaded"
        }
    else:
        return {
            "status": "not_loaded", 
            "error": "Model not yet initialized"
        }

@app.post("/generate")
def generate(req: GenerateRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "file": None, "error": None}
    bg.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "running"}

@app.post("/generate/ultrafast")
def generate_ultrafast(req: GenerateRequest, bg: BackgroundTasks):
    """Ultra-fast generation with maximum speed optimizations for A100"""
    # Override settings for maximum speed
    req.steps = min(req.steps, 15)  # Cap at 15 steps
    req.width = min(req.width, 960)  # Cap resolution for speed
    req.height = min(req.height, 544)
    req.num_frames = min(req.num_frames, 25)
    req.enable_fast_mode = True
    
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "file": None, "error": None, "mode": "ultrafast"}
    bg.add_task(_run_job, job_id, req)
    return {
        "job_id": job_id, 
        "status": "running",
        "mode": "ultrafast",
        "estimated_time": "1-3 minutes",
        "settings": {
            "steps": req.steps,
            "resolution": f"{req.width}x{req.height}",
            "frames": req.num_frames
        }
    }

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = {"job_id": job_id, **job}
    if job.get("file"):
        resp["result_url"] = f"/result/{job_id}"
    return JSONResponse(resp)

@app.get("/result/{job_id}")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("file"):
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(job["file"], media_type="video/mp4", filename=f"{job_id}.mp4")
