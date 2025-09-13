# src/bench_onnx.py
import os, time, csv
import numpy as np
import onnxruntime as ort
from transformers import GPT2Tokenizer

ONNX_PATH = "gpt2.onnx"
PROMPT = "In 2025, embedded AI systems"
MAX_NEW_TOKENS = 64
RUNS = 10
CSV_PATH = "results/onnx_baseline.csv"

def main():
    os.makedirs("results", exist_ok=True)

    tok = GPT2Tokenizer.from_pretrained("gpt2")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])

    def one_forward(prompt_text: str):
        # Get numpy inputs from the tokenizer
        enc = tok(prompt_text, return_tensors="np")
        input_ids = enc["input_ids"]

        # Some GPT-2 exports *require* attention_mask.
        # If tokenizer didn't return it (rare), make a mask of ones.
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        _ = sess.run(["logits"], inputs)

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
        print(f"Run {i+1:02d}/{RUNS}: {dt*1000:.1f} ms")

    times = np.array(times)
    mean_ms = times.mean() * 1000
    p95_ms  = np.percentile(times * 1000, 95)
    tokens_per_sec = MAX_NEW_TOKENS / times.mean()

    print("\nONNX Runtime (baseline):")
    print(f"Mean latency: {mean_ms:.1f} ms")
    print(f"P95  latency: {p95_ms:.1f} ms")
    print(f"Throughput:   {tokens_per_sec:.2f} tokens/sec")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine","mean_ms","p95_ms","tokens_per_sec"])
        w.writerow(["onnx", f"{mean_ms:.2f}", f"{p95_ms:.2f}", f"{tokens_per_sec:.2f}"])

    print(f"\nSaved results to: {CSV_PATH}")

if __name__ == "__main__":
    main()
