#!/usr/bin/env bash
# Export SFT checkpoint → HF format → GGUF → Ollama
#
# Usage:
#   ./scripts/export_to_gguf.sh [checkpoint.pt] [tokenizer.model]
#
# Prerequisites:
#   - core_llm package installed (pip install -e core_llm/)
#   - llama.cpp cloned and built: https://github.com/ggerganov/llama.cpp
#     export LLAMA_CPP_DIR=/path/to/llama.cpp
#   - Ollama installed: https://ollama.com

set -euo pipefail

CHECKPOINT="${1:-sft_best.pt}"
TOKENIZER="${2:-tokenizer.model}"
HF_DIR="./exports/hf_model"
GGUF_F16="./exports/model-f16.gguf"
GGUF_Q4="./exports/model-q4_k_m.gguf"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_NAME="llm-ja-104m"

echo "=== Step 1: Export to HF format ==="
python core_llm/src/core_llm/scripts/export_hf_llama.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$HF_DIR"

echo ""
echo "=== Step 2: Convert to GGUF (F16) ==="
if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
    echo "ERROR: llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Clone it: git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR"
    echo "Then: pip install -r $LLAMA_CPP_DIR/requirements.txt"
    exit 1
fi

mkdir -p ./exports
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$HF_DIR" \
    --outfile "$GGUF_F16" \
    --outtype f16

echo ""
echo "=== Step 3: Quantize to Q4_K_M ==="
if [ ! -f "$LLAMA_CPP_DIR/llama-quantize" ] && [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    echo "ERROR: llama-quantize not found. Build llama.cpp first:"
    echo "  cd $LLAMA_CPP_DIR && cmake -B build && cmake --build build -j"
    exit 1
fi

QUANTIZE_BIN="$LLAMA_CPP_DIR/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
fi

"$QUANTIZE_BIN" "$GGUF_F16" "$GGUF_Q4" Q4_K_M

echo ""
echo "=== Step 4: Create Ollama model ==="
cat > ./exports/Modelfile << 'MODELFILE_EOF'
FROM ./model-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}{{ .Response }}<|end|>"""

SYSTEM "あなたは日本語で会話するアシスタントです。"

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
MODELFILE_EOF

cd ./exports
ollama create "$MODEL_NAME" -f Modelfile
cd -

echo ""
echo "=== Done! ==="
echo "Run with: ollama run $MODEL_NAME"
echo "Test:     ollama run $MODEL_NAME '東京の名所を教えてください'"
