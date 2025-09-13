# src/bench_onnx_int8.py
"""
What this does:
- Takes your existing ONNX model (FP32) and makes an INT8 copy (smaller/faster).
- Benchmarks the INT8 model with the same workload as bench_onnx.py.
- Prints mean latency, P95, tokens/sec.
- Saves CSV to results/onnx_int8.csv
"""

import os, time, csv
import numpy as np
import onnxruntime as ort

# quantization API
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception as e:
    raise SystemExit(
        "Quantization module missing. Inside your venv, run:\n"
        "  pip install onnxruntime onnxruntime-tools\n"
        f"(Original error: {e})"
    )

from transformers import GPT2Tokenizer

# ----- SETTINGS (adjust if you need) -----
FP32_PATH = "gpt2.onnx"            # your existing model
INT8_PATH = "gpt2-int8.onnx"       # output path for INT8 model
PROMPT = "In 2025, embedded AI systems"
MAX_NEW_TOKENS = 64                # repeat count = proxy for work
RUNS = 10
CSV_PATH = "results/onnx_int8.csv"
# -----------------------------------------

def bench(model_path: str, label: str):
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    def one_forward(text: str):
        enc = tok(text, return_tensors="np")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        _ = sess.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})

    # Warm-up
    for _ in range(3):
        one_forward(PROMPT)

    times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        for _ in range(MAX_NEW_TOKENS):
            one_forward(PROMPT)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[{label}] Run {i+1:02d}/{RUNS}: {dt*1000:.1f} ms")

    times = np.array(times)
    mean_ms = times.mean() * 1000
    p95_ms  = np.percentile(times * 1000, 95)
    tokens_per_sec = MAX_NEW_TOKENS / times.mean()
    return mean_ms, p95_ms, tokens_per_sec

def main():
    os.makedirs("results", exist_ok=True)

    # 1) Quantize FP32 -> INT8 (weights become int8; model often runs faster on CPU)
    print("Quantizing to INT8…")
    quantize_dynamic(
    model_input=FP32_PATH,
    model_output=INT8_PATH,
    weight_type=QuantType.QInt8   # quantize weights to int8
)

    print(f"Saved INT8 model -> {INT8_PATH}")

    # 2) Benchmark INT8
    mean_ms, p95_ms, tps = bench(INT8_PATH, "INT8")

    # 3) Save CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine","mean_ms","p95_ms","tokens_per_sec"])
        w.writerow(["onnx_int8", f"{mean_ms:.2f}", f"{p95_ms:.2f}", f"{tps:.2f}"])

    print("\nONNX INT8 results:")
    print(f"Mean latency: {mean_ms:.1f} ms")
    print(f"P95  latency: {p95_ms:.1f} ms")
    print(f"Throughput:   {tps:.2f} tokens/sec")
    print(f"Saved results to: {CSV_PATH}")

if __name__ == "__main__":
    main()
