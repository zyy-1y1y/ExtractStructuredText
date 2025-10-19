## 环境搭建（开发）

1.安装 python3.14+
2.安装 pycharm 或 vscode 或 trae(推荐)
3.安装 Rust
4.安装 visual studio 2022 并安装 desktop development with C++ (C++桌面应用开发)


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