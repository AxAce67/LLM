import os
import threading
import subprocess
from typing import Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
import traceback
import torch

import main_controller
from runtime.auto_tuner import detect_runtime_profile

app = FastAPI(title="LLM Builder Dashboard")
templates = Jinja2Templates(directory="templates")
_auto_profile = detect_runtime_profile() if os.environ.get("AUTO_TUNE", "1") == "1" else None
MAX_GENERATE_TOKENS = int(os.environ.get("MAX_GENERATE_TOKENS", str(_auto_profile["max_generate_tokens"] if _auto_profile else 256)))
DB_UI_URL = os.environ.get("DB_UI_URL", f"http://localhost:{os.environ.get('ADMINER_PORT', '8080')}")
SYSTEM_ROLE = os.environ.get("SYSTEM_ROLE", "worker")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "llm")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "llm_user")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
BOOTSTRAP_TOKEN = os.environ.get("BOOTSTRAP_TOKEN", "")
ALLOW_BOOTSTRAP_PASSWORD = os.environ.get("ALLOW_BOOTSTRAP_PASSWORD", "0") == "1"
ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN", "").strip()
ADMIN_API_TOKEN_REQUIRED = os.environ.get(
    "ADMIN_API_TOKEN_REQUIRED",
    "1" if SYSTEM_ROLE == "master" else "0",
) == "1"

# メインコントローラーの状態管理オブジェクト（ダッシュボード閲覧・操作専用モード）
state_manager = main_controller.SystemState(is_dashboard=True)

class ControlRequest(BaseModel):
    action: str

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=50, ge=1, le=MAX_GENERATE_TOKENS)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=2.0)
    rag: bool = True

class SourcePolicyRequest(BaseModel):
    domain_pattern: str
    source_type: str = "web"
    license_tag: str = "unknown"
    allow_training: bool = True
    base_weight: float = 1.0
    notes: str = ""


_inference_lock = threading.Lock()
_cached_model = None
_cached_tokenizer = None
_cached_device: Optional[str] = None
_eval_lock = threading.Lock()
_eval_running = False
_eval_last_output = ""
_eval_last_error = ""
_promote_lock = threading.Lock()
_hf_train_lock = threading.Lock()
_hf_train_running = False
_hf_train_last_output = ""
_hf_train_last_error = ""
_gguf_export_lock = threading.Lock()
_gguf_export_running = False
_gguf_export_last_output = ""
_gguf_export_last_error = ""


def _is_admin_request(request: Request) -> bool:
    if not ADMIN_API_TOKEN:
        if ADMIN_API_TOKEN_REQUIRED:
            return False
        return True
    token = request.headers.get("x-admin-token", "").strip()
    if not token:
        token = request.query_params.get("token", "").strip()
    return token == ADMIN_API_TOKEN


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_inference_runtime() -> Tuple[object, object, str]:
    global _cached_model, _cached_tokenizer, _cached_device

    with _inference_lock:
        if _cached_model is not None and _cached_tokenizer is not None and _cached_device is not None:
            return _cached_model, _cached_tokenizer, _cached_device

        import inference

        device = _resolve_device()
        tokenizer = inference.load_tokenizer()
        model, _ = inference.load_model(device=device)

        _cached_model = model
        _cached_tokenizer = tokenizer
        _cached_device = device
        return _cached_model, _cached_tokenizer, _cached_device

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    ダッシュボード画面を返す
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate")
async def generate_response(req: GenerateRequest):
    """
    学習済みモデルを使ってユーザーの入力プロンプトに対する続きのテキストを生成するAPI
    """
    try:
        import inference
        model, sp, device = _get_inference_runtime()
        prompt = req.prompt

        if req.rag and os.environ.get("ENABLE_RAG", "1") == "1":
            snippets = state_manager.db_manager.search_relevant_contents(req.prompt, limit=3)
            if snippets:
                context_text = "\n\n".join([f"[Doc {i+1}] {s[:700]}" for i, s in enumerate(snippets)])
                prompt = (
                    "You are a helpful assistant. Use retrieved context when relevant.\n"
                    f"Retrieved Context:\n{context_text}\n\n"
                    f"User Prompt:\n{req.prompt}\n\nAnswer:"
                )

        # テキスト生成
        requested_tokens = min(max(1, req.max_tokens), MAX_GENERATE_TOKENS)
        output_text = inference.generate_text(
            model,
            sp,
            prompt,
            max_new_tokens=requested_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            device=device
        )
        return JSONResponse(content={"status": "success", "text": output_text})
        
    except FileNotFoundError as e:
         return JSONResponse(status_code=400, content={"status": "error", "message": f"モデルが見つかりません: {str(e)}"})
    except Exception as e:
         error_msg = traceback.format_exc()
         return JSONResponse(status_code=500, content={"status": "error", "message": f"生成エラー: {e}\n{error_msg}"})

@app.get("/api/runtime-config")
async def runtime_config():
    profile = detect_runtime_profile() if os.environ.get("AUTO_TUNE_REALTIME", "1") == "1" else _auto_profile
    base_dir = os.path.dirname(os.path.abspath(__file__))
    production_ckpt = os.path.join(base_dir, "checkpoints", "ckpt_production.pt")
    return {
        "max_generate_tokens": MAX_GENERATE_TOKENS,
        "db_ui_url": DB_UI_URL,
        "system_role": SYSTEM_ROLE,
        "postgres_host": POSTGRES_HOST,
        "admin_token_required": bool(ADMIN_API_TOKEN_REQUIRED),
        "use_production_model": os.environ.get("USE_PRODUCTION_MODEL", "1") == "1",
        "production_checkpoint_exists": os.path.exists(production_ckpt),
        "auto_profile": profile,
    }


@app.get("/api/bootstrap")
async def bootstrap_config(token: str = ""):
    if not BOOTSTRAP_TOKEN or token != BOOTSTRAP_TOKEN:
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    payload = {
        "status": "success",
        "system_role": SYSTEM_ROLE,
        "postgres_db": POSTGRES_DB,
        "postgres_user": POSTGRES_USER,
        "postgres_port": POSTGRES_PORT,
    }
    if ALLOW_BOOTSTRAP_PASSWORD:
        payload["postgres_password"] = POSTGRES_PASSWORD
    return JSONResponse(content=payload)


@app.post("/api/nodes/control-all")
async def control_all_nodes(request: Request):
    if not _is_admin_request(request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    try:
        data = await request.json()
        action = data.get("action")
        role = data.get("role", "worker")
        if action not in ["start", "stop"]:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid action"})
        if role not in ["worker", "master", "all"]:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid role"})

        state_manager.db_manager.set_all_nodes_target_status(action, role=role)
        state_manager.log(f"User requested: {action.upper()} for role={role}")
        return JSONResponse(content={"status": "success", "message": f"Command {action} sent to role={role}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/evals")
async def list_evaluations():
    try:
        runs = state_manager.db_manager.get_latest_evaluation_runs(limit=20)
        return JSONResponse(content={"status": "success", "runs": runs})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


def _run_eval_task():
    global _eval_running, _eval_last_output, _eval_last_error
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = ["python3", os.path.join(base_dir, "eval", "evaluate_model.py")]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir, timeout=1800)
        _eval_last_output = (completed.stdout or "").strip()
        if completed.returncode != 0:
            _eval_last_error = (completed.stderr or completed.stdout or f"eval failed (rc={completed.returncode})").strip()
        else:
            _eval_last_error = (completed.stderr or "").strip()
    except Exception as e:
        _eval_last_output = ""
        _eval_last_error = str(e)
    finally:
        _eval_running = False


def _run_hf_train_task(base_model: str, train_text: str, output_dir: str):
    global _hf_train_running, _hf_train_last_output, _hf_train_last_error
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "bash",
        os.path.join(base_dir, "migration_hf", "run_train_auto.sh"),
    ]
    env = os.environ.copy()
    env["BASE_MODEL"] = base_model
    env["TRAIN_TEXT"] = train_text
    env["OUTPUT_DIR"] = output_dir
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir, timeout=7200, env=env)
        _hf_train_last_output = (completed.stdout or "").strip()
        if completed.returncode != 0:
            _hf_train_last_error = (completed.stderr or completed.stdout or f"hf train failed (rc={completed.returncode})").strip()
        else:
            _hf_train_last_error = (completed.stderr or "").strip()
    except Exception as e:
        _hf_train_last_output = ""
        _hf_train_last_error = str(e)
    finally:
        _hf_train_running = False


def _run_gguf_export_task(llama_cpp_dir: str, hf_model: str, out_gguf: str, quant: str):
    global _gguf_export_running, _gguf_export_last_output, _gguf_export_last_error
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "bash",
        os.path.join(base_dir, "migration_hf", "export_gguf.sh"),
        llama_cpp_dir,
        hf_model,
        out_gguf,
    ]
    env = os.environ.copy()
    if quant:
        env["QUANT"] = quant
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir, timeout=1800, env=env)
        _gguf_export_last_output = (completed.stdout or "").strip()
        if completed.returncode != 0:
            _gguf_export_last_error = (completed.stderr or completed.stdout or f"gguf export failed (rc={completed.returncode})").strip()
        else:
            _gguf_export_last_error = (completed.stderr or "").strip()
    except Exception as e:
        _gguf_export_last_output = ""
        _gguf_export_last_error = str(e)
    finally:
        _gguf_export_running = False


@app.post("/api/evals/run")
async def run_evaluation(request: Request):
    if not _is_admin_request(request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    global _eval_running
    with _eval_lock:
        if _eval_running:
            return JSONResponse(content={"status": "busy", "message": "Evaluation is already running."})
        _eval_running = True
        t = threading.Thread(target=_run_eval_task, daemon=True)
        t.start()
    return JSONResponse(content={"status": "started", "message": "Evaluation started."})


@app.get("/api/evals/status")
async def eval_status():
    return JSONResponse(
        content={
            "status": "running" if _eval_running else "idle",
            "last_output": _eval_last_output,
            "last_error": _eval_last_error,
        }
    )


@app.post("/api/migration/hf-train/run")
async def run_hf_train(request: Request):
    if not _is_admin_request(request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    global _hf_train_running
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    base_model = body.get("base_model") or os.environ.get("HF_BASE_MODEL_DEFAULT", "Qwen/Qwen2.5-1.5B-Instruct")
    train_text = body.get("train_text") or os.environ.get("HF_TRAIN_TEXT_PATH", "dataset/hf/train.txt")
    output_dir = body.get("output_dir") or os.environ.get("HF_OUTPUT_DIR", "models/hf_lora")
    with _hf_train_lock:
        if _hf_train_running:
            return JSONResponse(content={"status": "busy", "message": "HF training is already running."})
        _hf_train_running = True
        t = threading.Thread(
            target=_run_hf_train_task,
            args=(base_model, train_text, output_dir),
            daemon=True,
        )
        t.start()
    return JSONResponse(content={"status": "started", "message": "HF LoRA training started."})


@app.get("/api/migration/hf-train/status")
async def hf_train_status():
    return JSONResponse(
        content={
            "status": "running" if _hf_train_running else "idle",
            "last_output": _hf_train_last_output,
            "last_error": _hf_train_last_error,
        }
    )


@app.post("/api/migration/gguf-export/run")
async def run_gguf_export(request: Request):
    if not _is_admin_request(request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    global _gguf_export_running
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    llama_cpp_dir = body.get("llama_cpp_dir") or os.environ.get("LLAMA_CPP_DIR", "")
    hf_model = body.get("hf_model") or os.environ.get("HF_EXPORT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    out_gguf = body.get("out_gguf") or os.environ.get("GGUF_OUTPUT_PATH", "models/export/model.gguf")
    quant = body.get("quant") or os.environ.get("GGUF_QUANT", "Q4_K_M")
    if not llama_cpp_dir:
        return JSONResponse(status_code=400, content={"status": "error", "message": "llama_cpp_dir is required"})
    with _gguf_export_lock:
        if _gguf_export_running:
            return JSONResponse(content={"status": "busy", "message": "GGUF export is already running."})
        _gguf_export_running = True
        t = threading.Thread(
            target=_run_gguf_export_task,
            args=(llama_cpp_dir, hf_model, out_gguf, quant),
            daemon=True,
        )
        t.start()
    return JSONResponse(content={"status": "started", "message": "GGUF export started."})


@app.get("/api/migration/gguf-export/status")
async def gguf_export_status():
    return JSONResponse(
        content={
            "status": "running" if _gguf_export_running else "idle",
            "last_output": _gguf_export_last_output,
            "last_error": _gguf_export_last_error,
        }
    )


@app.get("/api/models")
async def list_models():
    try:
        models = state_manager.db_manager.list_model_versions(limit=20)
        return JSONResponse(content={"status": "success", "models": models})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/policies")
async def list_policies():
    try:
        rows = state_manager.db_manager.list_source_policies(limit=300)
        return JSONResponse(content={"status": "success", "policies": rows})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/datasets")
async def list_datasets():
    try:
        rows = state_manager.db_manager.get_latest_dataset_versions(limit=50)
        return JSONResponse(content={"status": "success", "datasets": rows})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/policies")
async def upsert_policy(req: SourcePolicyRequest, request: Request):
    if not _is_admin_request(request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    try:
        state_manager.db_manager.upsert_source_policy(
            domain_pattern=req.domain_pattern,
            source_type=req.source_type,
            license_tag=req.license_tag,
            allow_training=req.allow_training,
            base_weight=req.base_weight,
            notes=req.notes,
        )
        return JSONResponse(content={"status": "success", "message": f"Policy updated: {req.domain_pattern}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/models/promote")
async def promote_model(request: Request, force: bool = False):
    if not _is_admin_request(request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    with _promote_lock:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cmd = ["python3", os.path.join(base_dir, "ops", "promote_best_checkpoint.py")]
            env = os.environ.copy()
            if force:
                env["FORCE_PROMOTE"] = "1"
            completed = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir, timeout=120, env=env)
            if completed.returncode != 0:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": completed.stderr or completed.stdout or "promote failed"},
                )
            # promote後は推論モデルキャッシュをリセット
            global _cached_model, _cached_tokenizer, _cached_device
            _cached_model = None
            _cached_tokenizer = None
            _cached_device = None
            return JSONResponse(content={"status": "success", "message": (completed.stdout or "").strip()})
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/nodes")
async def get_nodes():
    """
    ダッシュボード用: 全ノードの情報を取得する
    """
    try:
        nodes = state_manager.db_manager.get_all_nodes()
        return JSONResponse(content={"status": "success", "nodes": nodes})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/nodes/{node_id}/control")
async def control_node(node_id: str, request: Request):
    """
    システム内の特定のノードにStart/Stopを指示するAPI
    """
    try:
        if not _is_admin_request(request):
            return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
        data = await request.json()
        action = data.get("action")
        if action in ["start", "stop"]:
            state_manager.db_manager.set_node_target_status(node_id, action)
            # 全体向けの命令としてもログを残す
            state_manager.log(f"User requested: {action.upper()} for node {node_id[-4:]}")
            return JSONResponse(content={"status": "success", "message": f"Command {action} sent to node {node_id}"})
        else:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid action"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/status")
async def get_status():
    """
    現在のシステム状態をJSONで返す
    """
    state_manager.load()
    return state_manager.state

@app.post("/api/control")
async def control_pipeline(request: ControlRequest, raw_request: Request):
    """
    システムの開始/停止を切り替えるAPI
    """
    if not _is_admin_request(raw_request):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden"})
    state_manager.load()
    if request.action == "start":
        state_manager.set_running(True)
        state_manager.log("User requested: START")
        return {"status": "success", "message": "Pipeline started"}
    elif request.action == "stop":
        state_manager.set_running(False)
        state_manager.log("User requested: STOP")
        return {"status": "success", "message": "Pipeline stopped"}
    else:
        return {"status": "error", "message": "Unknown action"}

if __name__ == "__main__":
    print("Starting Web Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
