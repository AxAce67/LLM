#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <llama_cpp_dir> <hf_model_local_dir> <output_gguf>"
  echo "Example: $0 ~/llama.cpp ./models/hf_base/qwen2.5-1.5b-instruct ./models/qwen2.5-1.5b.gguf"
  exit 1
fi

LLAMA_CPP_DIR="$1"
HF_MODEL="$2"
OUT_GGUF="$3"
QUANT="${QUANT:-Q4_K_M}"

PY="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QBIN="${LLAMA_CPP_DIR}/llama-quantize"

if [[ ! -f "${PY}" ]]; then
  echo "convert_hf_to_gguf.py not found: ${PY}"
  exit 1
fi

if [[ ! -d "${HF_MODEL}" ]]; then
  echo "HF model path must be a local directory: ${HF_MODEL}"
  exit 1
fi

TMP_F16="$(mktemp /tmp/model-f16-XXXXXX.gguf)"
trap 'rm -f "${TMP_F16}"' EXIT

python3 "${PY}" "${HF_MODEL}" --outfile "${TMP_F16}" --outtype f16

if [[ -x "${QBIN}" ]]; then
  "${QBIN}" "${TMP_F16}" "${OUT_GGUF}" "${QUANT}"
else
  echo "llama-quantize not found. writing f16 GGUF only."
  cp -f "${TMP_F16}" "${OUT_GGUF}"
fi

echo "GGUF exported: ${OUT_GGUF}"
