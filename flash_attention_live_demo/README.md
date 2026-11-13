# Live CUDA demonstration

The presentation notebook is [FlashAttention_Full_Run_Demo.ipynb](../FlashAttention_Full_Run_Demo.ipynb) in [Flash-Attention-Forward](https://github.com/ktk-07/Flash-Attention-Forward). The team reports a video duration of **14:58**. The notebook demonstrates fresh execution; its saved displays are previous observations until the cells are rerun.

## Setup and execution

Use an NVIDIA GPU with SM80 or newer, CUDA Toolkit/nvcc, CMake 3.20+, a C++17 host compiler and Python. On Windows, install Visual Studio 2022 C++ build tools and the Windows SDK. The helper can initialise the x64 compiler environment. The notebook's live CUDA steps do not require Nsight Compute or Compute Sanitizer; those tools are required by the separate full experiment.

Activate your Python environment and launch **from the repository root**:

```bash
python -m pip install -r part_c/requirements-analysis.txt
python -m pip install jupyterlab
python -m jupyterlab FlashAttention_Full_Run_Demo.ipynb
```

Select the same environment as the notebook kernel and execute cells in order. These commands also work from a Windows terminal with that environment active. Start JupyterLab in the checkout itself, including when its directory is named `Flash-Attention-Forward`. If starting elsewhere, set `FLASH_ATTENTION_REPO` to the absolute checkout path before launching JupyterLab.

The first CUDA build may take several minutes. It detects GPU 0's architecture and caches the executable using source/toolchain hashes. Complete the initial build before recording a presentation. Compilation or numerical failures stop the demonstration; inspect the recorded build log or case JSON before continuing.

## What the notebook runs

1. Detect the GPU, compile repository kernels and record provenance.
2. Check a small causal example against a full CPU FP64 reference.
3. Compare materialised A-PTX with fused C-PTX FA2 on identical FP16-rounded inputs.
4. Increase sequence length from N512 to N1024 at B4/H8/d64 and show latency and allocated tensor capacity separately.
5. Compare dense FP32 A with B, reporting memory savings and the observed latency outcome.
6. Validate the accepted historical bundle in `results/final_submission` without rerunning its GPU experiment.

Each measured implementation uses five warmups and 20 unprofiled CUDA-event samples. The display reports medians, quartiles, numerical checks and explicit allocation bytes. Small cases use full-output reference checks; larger cases use sampled query rows with finiteness checked over every output element. The workload cell exposes `SEED`, `N` and `N_LARGER`; keep `16 <= N < N_LARGER <= 2048` when changing the timing shapes.

The live recording used an RTX 4070 Ti. The accepted full-grid run used an RTX 3070 Ti Laptop GPU. The notebook detects the executing device; another supported GPU produces another experiment. Demo configurations are fixed: A tile16/softmax256 or A-PTX/softmax256, B vector/W4/BK32 and C-PTX configuration 0. They differ from the full run's independently frozen choices. Demo results must not replace the accepted full-grid results or be merged into their speedup denominators.

## Generated files

When launched from the repository root:

```text
flash_attention_live_demo/
  source/                          # CMakeLists.txt, demo_runner.cu, live_demo.py
  build/<source-hash>/             # executable, build log and build manifest
  runs/<UTC_timestamp>_<suffix>/
    provenance.json                # device/toolchain/source identity
    <mode>_N<length>_<suffix>.json  # checks, configuration and raw timing samples
```

The notebook embeds its three driver files and writes them into `source/`. It links the attention kernels and shared reference from the repository; retain the complete A/B/C checkout. The cached build and new measurement files are generated outputs. To distribute the demo as source, include the root notebook, this README and `source/`. The core submission ZIP helper does not automatically select the additional notebook/demo directory; the Git repository includes these files.

For the complete experiment, standalone parts and CPU-only figure regeneration, use [README.txt](../README.txt). For numerical contracts and report findings, use [README.md](../README.md) and the [accepted evidence verification](../results/final_submission/VERIFICATION.md).
