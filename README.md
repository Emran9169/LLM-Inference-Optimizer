# 🚀 LLM Inference Optimizer  
**Optimizing GPT-2 Inference with PyTorch, ONNX, and INT8 Quantization**  

---

## 📖 Project Overview  
This project explores how to **optimize Large Language Model (LLM) inference** by:  
1. Running a **baseline GPT-2 model in PyTorch**.  
2. Exporting GPT-2 to the **ONNX format** for accelerated inference.  
3. Applying **INT8 quantization** to improve performance and reduce memory usage.  
4. Benchmarking all approaches and comparing results.  

The goal is to demonstrate practical **model optimization and deployment techniques** that improve inference speed without sacrificing too much accuracy.  

---

## ⚙️ Tech Stack  
- **Python 3.10**  
- **PyTorch** – Baseline model inference  
- **ONNX Runtime** – Optimized execution engine  
- **ONNX Quantization Toolkit** – INT8 dynamic quantization  
- **Transformers (Hugging Face)** – Tokenizer & GPT-2 model  
- **Pandas / CSV** – Benchmark result storage  

---

## 📊 Benchmark Results  

| Engine       | Mean Latency (ms) | P95 Latency (ms) | Tokens/sec |
|--------------|------------------|------------------|------------|
| **PyTorch**  | ~2000+           | ~2100+           | ~30–40     |
| **ONNX**     | 1112.89          | 1172.04          | 57.51      |
| **ONNX INT8**| 484.36           | 500.73           | 132.13     |

✅ **2.3× faster** ONNX over PyTorch  
✅ **~4.5× faster** ONNX INT8 over PyTorch  

---

## 📂 Project Structure

'''
llm-inference-optimizer/

│── src/
│ ├── baseline_pytorch.py # Run baseline GPT-2 with PyTorch
│ ├── export_onnx.py # Export model to ONNX
│ ├── bench_onnx.py # Run ONNX inference benchmark
│ ├── bench_onnx_int8.py # Quantize & benchmark INT8 model
│
│── results/
│ ├── baseline_pytorch.csv # PyTorch results
│ ├── onnx_baseline.csv # ONNX results
│ ├── onnx_int8.csv # ONNX INT8 results
│
│── models/ # (ignored) contains exported ONNX models
│── requirements.txt
│── README.md
'''
