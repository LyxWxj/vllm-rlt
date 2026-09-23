"""Measure shared-layout prefill and verify generated tokens for a local Ouro model.

Run with a reserved GPU, for example:
    canhazgpu run --gpus 1 -- python -m benchmarks.shared_prefill \
        --model ~/models/Ouro-2.6B --output /tmp/shared-prefill.json
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_rlt.config import CacheConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.ouro import OuroForCausalLM
from vllm_rlt.request import Stage
from vllm_rlt.sampling_params import SamplingParams


def run_once(model, prompt_lengths, layout):
    prompts = {
        f"request-{index}": [
            3 + (index * 97 + token * 31) % model.config.vocab_size for token in range(length)
        ]
        for index, length in enumerate(prompt_lengths)
    }
    max_tokens = 4
    block_size = 16
    pages_per_request = (max(prompt_lengths) + max_tokens - 1 + block_size - 1) // block_size
    storage_depths = model.config.total_ut_steps if layout == "last_exited" else 1
    num_blocks = pages_per_request * len(prompts) * storage_depths
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=num_blocks, block_size=block_size, layout=layout),
        scheduler_config=SchedulerConfig(
            max_num_seqs=len(prompts),
            max_num_batched_tokens=sum(prompt_lengths),
            prefill_chunk_size=sum(prompt_lengths),
        ),
        attention_backend="triton",
    )
    params = SamplingParams(
        max_tokens=max_tokens,
        min_loops=model.config.total_ut_steps,
        max_loops=model.config.total_ut_steps,
        exit_threshold=1.0,
        temperature=0.0,
        ignore_eos=True,
    )
    for request_id, tokens in prompts.items():
        engine.add_request(request_id, tokens, params)

    torch.cuda.synchronize()
    start = time.perf_counter()
    prefill_seconds = 0.0
    outputs = {}
    while engine.has_unfinished_requests():
        step_start = time.perf_counter()
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
        batch = engine.last_schedule
        if batch is not None and batch.stage == Stage.PREFILL:
            torch.cuda.synchronize()
            prefill_seconds += time.perf_counter() - step_start
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        request_id: {
            "token_ids": output.token_ids,
            "exit_depths": output.exit_depths,
        }
        for request_id, output in outputs.items()
    }
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result, prefill_seconds, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("~/models/Ouro-2.6B"))
    parser.add_argument("--layout", choices=["shared", "last_exited"], default="shared")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[16, 24, 32, 40])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.warmup < 0 or args.repeats < 1 or any(length < 1 for length in args.prompt_lengths):
        raise ValueError("warmup must be nonnegative, repeats and prompt lengths must be positive")

    model = OuroForCausalLM.from_pretrained(
        args.model.expanduser(), device="cuda", dtype=torch.bfloat16
    )
    warm_outputs = None
    for _ in range(args.warmup):
        warm_outputs, _, _ = run_once(model, args.prompt_lengths, args.layout)

    samples = []
    for _ in range(args.repeats):
        outputs, prefill_seconds, elapsed = run_once(model, args.prompt_lengths, args.layout)
        if warm_outputs is not None and outputs != warm_outputs:
            raise AssertionError("generated tokens or exit depths differ from warmup")
        warm_outputs = outputs
        samples.append(
            {
                "prefill_seconds": prefill_seconds,
                "decode_seconds": elapsed - prefill_seconds,
                "elapsed_seconds": elapsed,
            }
        )

    token_count = sum(len(row["token_ids"]) for row in warm_outputs.values())
    prompt_token_count = sum(args.prompt_lengths)
    median_prefill = statistics.median(row["prefill_seconds"] for row in samples)
    median_decode = statistics.median(row["decode_seconds"] for row in samples)
    median_elapsed = statistics.median(row["elapsed_seconds"] for row in samples)
    result = {
        "model": str(args.model.expanduser()),
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "layout": args.layout,
        "prompt_lengths": args.prompt_lengths,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": token_count,
        "outputs": warm_outputs,
        "samples": samples,
        "median_prefill_seconds": median_prefill,
        "median_decode_seconds": median_decode,
        "median_elapsed_seconds": median_elapsed,
        "median_prompt_tokens_per_second": prompt_token_count / median_prefill,
        "median_decode_output_tokens_per_second": token_count / median_decode,
        "median_output_tokens_per_second": token_count / median_elapsed,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
