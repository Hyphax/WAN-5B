#!/usr/bin/env python3
"""
Deployment safety test - run this before starting the server
"""
import sys
import os

def test_imports():
    """Test critical imports"""
    print("🔍 Testing critical imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        from diffusers import DiffusionPipeline
        print("✅ DiffusionPipeline import successful")
        
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA...")
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return False
            
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {device_count} device(s)")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   Device {i}: {props.name} ({memory_gb:.1f}GB)")
            
        return True
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test if we can load at least one model"""
    print("\n🔍 Testing model loading...")
    try:
        from diffusers import DiffusionPipeline
        
        # Test with a small, reliable model first
        test_model = "cerspense/zeroscope_v2_576w"
        print(f"   Attempting to load: {test_model}")
        
        pipe = DiffusionPipeline.from_pretrained(
            test_model,
            torch_dtype=torch.bfloat16
        )
        print("✅ Model loading successful")
        
        # Quick inference test
        pipe.to("cuda")
        print("✅ Model moved to CUDA")
        
        del pipe
        torch.cuda.empty_cache()
        print("✅ Model cleanup successful")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 DEPLOYMENT SAFETY TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Model Loading", test_model_loading)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {name} test FAILED")
        except Exception as e:
            print(f"\n❌ {name} test ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ ALL TESTS PASSED - SAFE TO DEPLOY")
        return True
    else:
        print("❌ SOME TESTS FAILED - DO NOT DEPLOY")
        return False

if __name__ == "__main__":
    import torch  # Import here so we can use it in functions
    success = main()
    sys.exit(0 if success else 1)
