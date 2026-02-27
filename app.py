import json
import os
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import traceback

import main_controller

app = FastAPI(title="LLM Builder Dashboard")
templates = Jinja2Templates(directory="templates")

# メインコントローラーの状態管理オブジェクト（ダッシュボード閲覧・操作専用モード）
state_manager = main_controller.SystemState(is_dashboard=True)

class ControlRequest(BaseModel):
    action: str

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

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
        # モデルとトークナイザをロード
        sp = inference.load_tokenizer()
        
        import torch
        device = 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'
        elif torch.cuda.is_available(): device = 'cuda'
        
        model, _ = inference.load_model(device=device)
        
        # テキスト生成
        output_text = inference.generate_text(model, sp, req.prompt, max_new_tokens=req.max_tokens, device=device)
        return JSONResponse(content={"status": "success", "text": output_text})
        
    except FileNotFoundError as e:
         return JSONResponse(status_code=400, content={"status": "error", "message": f"モデルが見つかりません: {str(e)}"})
    except Exception as e:
         error_msg = traceback.format_exc()
         return JSONResponse(status_code=500, content={"status": "error", "message": f"生成エラー: {e}\n{error_msg}"})

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
async def control_pipeline(request: ControlRequest):
    """
    システムの開始/停止を切り替えるAPI
    """
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
