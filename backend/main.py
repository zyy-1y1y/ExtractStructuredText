from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import json
import csv
import logging
from dotenv import load_dotenv
from pipeline import load_rules, process_documents, read_annotations, apply_new_rules, call_deepseek_generate_rules

# 加载.env文件中的环境变量
load_dotenv()

BASE_DIR = os.path.dirname(__file__)           # 当前脚本所处路径
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 输出路径（日志等）
os.makedirs(OUTPUT_DIR, exist_ok=True)         # 创建输出路径

# 配置日志输出路径、级别为INFO，定义日志格式：时间戳 + 日志级别 + 消息内容
# 使用更可靠的日志配置方法避免乱码
log_file = os.path.join(OUTPUT_DIR, 'pipeline.log')

# 创建文件处理器并设置UTF-8编码
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)

# 配置根日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# 同时配置根日志记录器，确保所有模块都能正确记录
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)


app = FastAPI(title="文本结构化提取工具")

# 挂载前端静态资源
frontend_dir = os.path.join(BASE_DIR, '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount('/static', StaticFiles(directory=frontend_dir), name='static')

# 定义前端发送的文档数据结构
class Document(BaseModel):
    doc_id: str
    raw_text: str

# 定义批量处理请求的数据结构
class ProcessRequest(BaseModel):
    documents: List[Document]

@app.get('/', response_class=HTMLResponse)
async def index():
    # 获取前端页面文件 index.html
    index_path = os.path.join(frontend_dir, 'index.html')
    # 如果前端文件存在，读取并返回该 HTML 文件内容
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse('<html><body><h3>请将前端页面文件 index.html 放在 /frontend 目录下并重启服务。</h3></body></html>')

@app.post('/api/process')
async def api_process(req: ProcessRequest):
    rules = load_rules()                          # 加载结构化提取规则
    docs = [{'doc_id': d.doc_id, 'raw_text': d.raw_text} for d in req.documents] # 将请求中的文档列表转换为处理所需的格式
    results = process_documents(docs, rules)      # 调用处理函数进行结构化提取
    return JSONResponse({'results': results})     # 将处理结果包装成 JSON 格式返回

@app.get('/api/structured')
async def get_structured():
    path = os.path.join(OUTPUT_DIR, 'structured.json')  # 结构化提取结果文件路径
    # 如果文件存在，返回文件内容，将文件作为 JSON 文件下载；否则返回 404 错误
    if os.path.exists(path):
        return FileResponse(path, media_type='application/json', filename='structured.json')
    return JSONResponse({'error': 'not found'}, status_code=404)

@app.get('/api/failures')
async def get_failures():
    path = os.path.join(OUTPUT_DIR, 'failures.jsonl')
    # 获取文本结构化处理过程中失败的记录（方便前端展示处理失败的情况）
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f if l.strip()]
        return JSONResponse({'failures': lines})
    return JSONResponse({'failures': []})

@app.post('/api/annotations/upload')
async def upload_annotations(file: UploadFile = File(...)):
    # 接受 CSV 文件并保存到 output/annotations.csv
    data = await file.read()
    path = os.path.join(OUTPUT_DIR, 'annotations.csv')
    with open(path, 'wb') as f:
        f.write(data)
    return JSONResponse({'status': 'ok', 'path': path})

@app.post('/api/annotations/add')
async def add_annotation(doc_id: str = Form(...), raw_text: str = Form(...), param_name: str = Form(...), param_value: str = Form(...)):
    path = os.path.join(OUTPUT_DIR, 'annotations.csv')
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['doc_id', 'raw_text', 'param_name', 'param_value'])
        writer.writerow([doc_id, raw_text, param_name, param_value])
    return JSONResponse({'status': 'ok'})

@app.post('/api/retrain')
async def retrain_and_apply():
    """
    重训练API：基于标注数据生成新规则并应用到系统
    
    Returns:
        JSONResponse: 重训练结果状态和详细信息
    """
    # 读取标注数据
    annotations = read_annotations()
    if not annotations:
        return JSONResponse({'status': 'no_annotations', 'message': '没有找到标注数据'})
    
    # 记录重训练开始
    logger.info(f"开始重训练，标注数据数量: {len(annotations)}")
    
    # 调用 DeepSeek 生成新规则
    new_rules = call_deepseek_generate_rules(annotations)
    
    if new_rules:
        # 应用新规则
        apply_new_rules(new_rules)
        
        # 记录重训练成功
        logger.info(f"重训练成功，应用了 {len(new_rules)} 条新规则")
        
        # 返回成功结果
        return JSONResponse({
            'status': 'applied', 
            'rules_count': len(new_rules),
            'message': f'成功应用了 {len(new_rules)} 条新规则',
            'annotations_count': len(annotations)
        })
    else:
        # 检查 DeepSeek 是否启用
        from pipeline import DEEPSEEK_ENABLE
        if not DEEPSEEK_ENABLE:
            logger.info("重训练跳过：DeepSeek 功能未启用")
            return JSONResponse({
                'status': 'deepseek_disabled', 
                'message': 'DeepSeek 功能未启用，无法进行规则重训练'
            })
        else:
            logger.info("重训练完成：DeepSeek 未生成新规则")
            return JSONResponse({
                'status': 'no_rules_generated', 
                'message': 'DeepSeek 未生成新规则，规则库保持不变',
                'annotations_count': len(annotations)
            })

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)

'''
> 说明：`main.py` 通过导入 `pipeline.py` 使用相同的解析逻辑。`/api/process` 接收 JSON 文档批量处理并返回结果。前端通过 fetch 调用该 API。
> 如果要启用 deepseek 的 HTTP 调用，额外安装 `requests`。
> 说明：该前端尽量简洁，只用 CDN 的 Vue3，方便把 `frontend/index.html` 直接放到桌面应用 WebView 或本地静态服务器。它支持：粘文本 → 调用 `/api/process` → 展示解析结果 → 将失败条目标注并上传到后端。


## 如何运行（开发环境）

1. 克隆/复制上述文件结构。
2. 进入 `backend`：

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv/Scripts/activate
   pip install -r requirements.txt
   ```
3. 启动服务：

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
4. 在浏览器打开 `http://localhost:8000/`（或把 `frontend/index.html` 通过 WebView 嵌入桌面程序）。

---

## 嵌入桌面应用（WebView）建议

* **Electron**：将 `frontend/index.html` 嵌入 BrowserWindow。后端可以一起作为子进程启动（或把后端部署到本机端口）。
* **Tauri**：更轻量，适合 Rust 的桌面打包。通过 `tauri.conf.json` 指定 dev URL 指向 `http://127.0.0.1:8000/` 或把静态文件打包。推荐用于生产桌面应用。
* **.NET WebView2 / JavaFX WebView / CEF**：都可以嵌入页面，后端仍然运行在本地并监听端口。

简要流程：

1. 将后端作为本地服务（`uvicorn`）运行（可打包为可执行子进程）。
2. 桌面应用启动时，创建 WebView 指向 `http://127.0.0.1:8000/`，或直接加载 `frontend/index.html` 并把 API URL 指向 `http://127.0.0.1:8000/api/...`。

---

## deepseek 集成要点（提示）

1. 在 `pipeline.py` 中实现 `call_deepseek_extract` 与 `call_deepseek_generate_rules`：发送 HTTP 请求到 deepseek，**要求模型输出 JSON**，例如：

```json
[{"name":"LVEF","regex":"...","keywords":["LVEF"]}, ...]
```

2. 将 `DEEPSEEK_ENABLE=True` 并把 `DEEPSEEK_API_KEY` / `DEEPSEEK_ENDPOINT` 设置为环境变量或写入 `backend/.env`（注意安全）。
3. 接收 deepseek 返回的规则 JSON，调用 `apply_new_rules(new_rules)` 持久化，并使用新的规则重跑数据。

---

## 我可以为你继续完成的事情（可选，选其一）

1. 把 deepseek 的真实 API 调用写入 `pipeline.py`（需要 deepseek 的 API 文档或示例请求/响应）。
2. 增加前端按钮实现“在界面中触发 retrain（/api/retrain）并展示返回的规则变更结果”。
3. 把整个后端封装为可执行的本地二进制（使用 PyInstaller）并给出 Electron/Tauri 的最小打包示例。

---

如果你确认，我可以现在：

* (A) 把 `call_deepseek_extract` 和 `call_deepseek_generate_rules` 按你提供的 deepseek API 文档实现为真实可运行调用；
* (B) 或者直接把本项目整理成一个 Git 仓库模板并把可运行的压缩包（代码）给你（说明如何部署）。

你选择哪一个我就继续完成（我会直接把对应代码写好）。
'''