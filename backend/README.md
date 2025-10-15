> 说明：`main.py` 通过导入 `pipeline.py` 使用相同的解析逻辑。`/api/process` 接收 JSON 文档批量处理并返回结果。前端通过 fetch 调用该 API。
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

2. 接收 deepseek 返回的规则 JSON，调用 `apply_new_rules(new_rules)` 持久化，并使用新的规则重跑数据。

---