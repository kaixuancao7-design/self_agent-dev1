import time
import json
import requests

from agent.react_agent import ReactAgent
from agent.langgraph_workflow import LangGraphAgent
from utils.config_handler import agent_cfg
import streamlit as st

# API 配置
API_BASE_URL = "http://localhost:8000/api"

def detect_task_complexity(prompt: str) -> str:
    """
    检测任务复杂度，决定使用哪种模式
    
    :param prompt: 用户输入
    :return: 推荐的模式（"react" 或 "langgraph"）
    """
    auto_switch = agent_cfg.get("langgraph", {}).get("auto_switch", False)
    
    if not auto_switch:
        return "react"
    
    complexity_score = 0
    
    if len(prompt) > 30:
        complexity_score += 1
    if len(prompt) > 60:
        complexity_score += 1
    
    complex_keywords = [
        "帮我", "请", "需要", "制定", "规划", "创建", "分析",
        "报告", "总结", "研究", "方案", "计划", "步骤", "流程",
        "设计", "撰写", "整理", "分析", "调研", "评估", "优化"
    ]
    keyword_matches = 0
    for keyword in complex_keywords:
        if keyword in prompt:
            keyword_matches += 1
    complexity_score += min(keyword_matches, 2)
    
    question_count = prompt.count("？") + prompt.count("?") + prompt.count("吗")
    if question_count >= 2:
        complexity_score += 1
    
    task_decompose_keywords = ["拆解", "分解", "步骤", "子任务", "先", "再", "然后", "依次"]
    for keyword in task_decompose_keywords:
        if keyword in prompt:
            complexity_score += 2
            break
    
    multi_step_indicators = ["首先", "其次", "接着", "最后", "第一步", "第二步"]
    for indicator in multi_step_indicators:
        if indicator in prompt:
            complexity_score += 2
            break
    
    if complexity_score >= 2:
        return "langgraph"
    return "react"

# API 调用函数
def api_call(endpoint: str, method: str = "GET", data: dict = None, files: dict = None) -> dict:
    """通用 API 调用函数"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url, params=data)
        else:
            return {"success": False, "error": f"不支持的方法: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"请求失败: {str(e)}"}

def api_chat_stream(session_id: str, message: str, mode: str):
    """流式 API 聊天"""
    try:
        url = f"{API_BASE_URL}/chat/stream"
        payload = {
            "session_id": session_id,
            "message": message,
            "mode": mode
        }
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        chunk = data.get('chunk', '')
                        finished = data.get('finished', False)
                        yield chunk
                        if finished:
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException as e:
        yield f"API 请求失败: {str(e)}"

def show_chat_page():
    st.title("🤖 智能知识库助手")
    st.divider()

    st.markdown("""
        <style>
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        .typing-cursor {
            animation: blink 1s infinite;
            color: #0ea5e9;
            font-weight: bold;
        }
        
        .user-message {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            color: white;
            border-radius: 12px 12px 4px 12px;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
            color: #1e293b;
            border-radius: 12px 12px 12px 4px;
        }
        
        .status-info {
            border-left: 4px solid #0ea5e9;
            padding-left: 12px;
            margin-bottom: 12px;
        }
        
        .status-success {
            border-left: 4px solid #22c55e;
            padding-left: 12px;
            margin-bottom: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 添加运行模式选择（本地模式 / API 模式）
    run_mode = st.sidebar.selectbox(
        "运行模式",
        ["🖥️ 本地模式", "🌐 API 模式"],
        index=0,
        help="本地模式：直接调用本地 Agent；API 模式：通过 FastAPI 后端调用"
    )

    # 初始化会话（API模式下）
    if run_mode == "🌐 API 模式":
        if "api_session_id" not in st.session_state:
            result = api_call("/sessions", "POST", {"user_id": "streamlit_user"})
            if result.get("success", True):
                st.session_state["api_session_id"] = result.get("session_id", "")
            else:
                st.session_state["api_session_id"] = "test_session"
        
        st.sidebar.info(f"会话 ID: {st.session_state['api_session_id']}")

    auto_switch_enabled = agent_cfg.get("langgraph", {}).get("auto_switch", False)
    
    if auto_switch_enabled:
        mode_options = ["🔄 自动切换模式", "⚡ ReactAgent 模式", "🧠 LangGraph 工作流模式"]
    else:
        mode_options = ["⚡ ReactAgent 模式", "🧠 LangGraph 工作流模式"]
    
    selected_mode = st.sidebar.selectbox(
        "选择工作模式",
        mode_options,
        index=0
    )

    if run_mode == "🖥️ 本地模式":
        if "agent" not in st.session_state:
            st.session_state["agent"] = ReactAgent()
            st.session_state["agent_mode"] = "⚡ ReactAgent 模式"
    
    if "message" not in st.session_state:
        st.session_state["message"] = []

    for idx, message in enumerate(st.session_state["message"]):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if st.button("📋 复制", key=f"copy_{idx}"):
                    import pyperclip
                    pyperclip.copy(message["content"])
                    st.success("已复制到剪贴板！")

    prompt = st.chat_input()

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["message"].append({"role":"user","content":prompt})
        response_messages = []
        
        clean_selected_mode = selected_mode.replace("🔄 ", "").replace("⚡ ", "").replace("🧠 ", "")
        if clean_selected_mode == "自动切换模式":
            detected_mode = detect_task_complexity(prompt)
            actual_mode = "🧠 LangGraph 工作流模式" if detected_mode == "langgraph" else "⚡ ReactAgent 模式"
            api_mode = "langgraph" if detected_mode == "langgraph" else "react"
        else:
            actual_mode = selected_mode
            api_mode = "langgraph" if "LangGraph" in actual_mode else "react"
        
        if run_mode == "🖥️ 本地模式":
            if st.session_state.get("agent_mode") != actual_mode:
                if "ReactAgent" in actual_mode:
                    st.session_state["agent"] = ReactAgent()
                else:
                    st.session_state["agent"] = LangGraphAgent()
                st.session_state["agent_mode"] = actual_mode
        
        st.sidebar.info(f"当前模式: {actual_mode}")
        
        status_placeholder = st.empty()
        status_placeholder.info("🤔 正在分析您的问题...")

        if run_mode == "🌐 API 模式":
            # 使用 API 模式
            def capture_api(generator, cache_list):
                status_shown = False
                for chunk in generator:
                    if chunk is None:
                        continue
                    cache_list.append(chunk)
                    
                    if not status_shown:
                        status_placeholder.success("✍️ 正在为您生成回答...")
                        status_shown = True
                    
                    if isinstance(chunk, str):
                        for char in chunk:
                            time.sleep(0.008)
                            yield char
                    else:
                        content = str(chunk)
                        for char in content:
                            time.sleep(0.008)
                            yield char
                
                status_placeholder.empty()
            
            with st.chat_message("assistant"):
                st.write_stream(capture_api(api_chat_stream(
                    st.session_state["api_session_id"], prompt, api_mode
                ), response_messages))
            full_response = "".join(response_messages)
        else:
            # 使用本地模式
            if "ReactAgent" in actual_mode:
                res_stream = st.session_state["agent"].execute_stream(prompt)
                
                def capture_react(generator, cache_list):
                    status_shown = False
                    for chunk in generator:
                        if chunk is None:
                            continue
                        cache_list.append(chunk)
                        
                        if not status_shown:
                            status_placeholder.success("✍️ 正在为您生成回答...")
                            status_shown = True
                        
                        if isinstance(chunk, str):
                            for char in chunk:
                                time.sleep(0.008)
                                yield char
                        elif isinstance(chunk, dict):
                            content = chunk.get('messages', [{}])[-1].get('content', '')
                            if content:
                                for char in content:
                                    time.sleep(0.008)
                                    yield char
                        else:
                            content = getattr(chunk, 'content', str(chunk))
                            for char in content:
                                time.sleep(0.008)
                                yield char
                    
                    status_placeholder.empty()
                
                with st.chat_message("assistant"):
                    st.write_stream(capture_react(res_stream, response_messages))
                full_response = "".join(response_messages)
            else:
                res_stream = st.session_state["agent"].run_stream(prompt)
                
                def capture_langgraph(generator, cache_list):
                    status_shown = False
                    for chunk in generator:
                        if chunk is None:
                            continue
                        cache_list.append(chunk)
                        
                        if not status_shown:
                            status_placeholder.success("✍️ 正在为您生成回答...")
                            status_shown = True
                        
                        if isinstance(chunk, str):
                            for char in chunk:
                                time.sleep(0.008)
                                yield char
                        else:
                            content = str(chunk)
                            for char in content:
                                time.sleep(0.008)
                                yield char
                    
                    status_placeholder.empty()
                
                with st.chat_message("assistant"):
                    st.write_stream(capture_langgraph(res_stream, response_messages))
                full_response = "".join(response_messages)
        
        if full_response:
            status_placeholder.success("✅ 回答完成")
            time.sleep(3)
            status_placeholder.empty()
        
        st.session_state["message"].append({"role":"assistant","content":full_response})
        st.rerun()


def show_upload_page():
    st.title("知识库管理")
    st.divider()

    # 添加运行模式选择
    run_mode = st.sidebar.selectbox(
        "运行模式",
        ["🖥️ 本地模式", "🌐 API 模式"],
        index=0,
        help="本地模式：直接操作本地数据库；API 模式：通过 FastAPI 后端操作"
    )

    from rag.vector_store import VectorStoreService
    from utils.config_handler import chroma_cfg
    from utils.file_handler import SUPPORTED_FILE_TYPES, get_file_md5_hex
    import tempfile
    import os

    if run_mode == "🖥️ 本地模式":
        if "vector_store" not in st.session_state:
            st.session_state["vector_store"] = VectorStoreService()
    
    if "upload_progress" not in st.session_state:
        st.session_state["upload_progress"] = 0
    if "delete_confirm" not in st.session_state:
        st.session_state["delete_confirm"] = False
    if "db_delete_confirm" not in st.session_state:
        st.session_state["db_delete_confirm"] = False
    if "create_db_name" not in st.session_state:
        st.session_state["create_db_name"] = ""
    if "delete_target" not in st.session_state:
        st.session_state["delete_target"] = None
    if "reparse_target" not in st.session_state:
        st.session_state["reparse_target"] = None
    if "reparse_file_name" not in st.session_state:
        st.session_state["reparse_file_name"] = None
    if "reparse_message" not in st.session_state:
        st.session_state["reparse_message"] = ""
    if "upload_key" not in st.session_state:
        st.session_state["upload_key"] = 0

    # 获取数据库列表
    if run_mode == "🌐 API 模式":
        db_list_result = api_call("/databases", "GET")
        db_list = db_list_result if isinstance(db_list_result, list) else []
        current_db_info = api_call("/databases/current", "GET")
        current_db = current_db_info.get("current_db", "default")
        stats = current_db_info
    else:
        db_list = VectorStoreService.list_databases()
        current_db = st.session_state["vector_store"].current_db
        stats = st.session_state["vector_store"].get_collection_stats()

    st.sidebar.subheader("数据库选择")
    
    if db_list:
        selected_db = st.sidebar.selectbox("选择数据库", db_list, 
            index=db_list.index(current_db) if current_db in db_list else 0)
        
        if selected_db != current_db:
            if run_mode == "🌐 API 模式":
                result = api_call(f"/databases/{selected_db}/switch", "POST")
            else:
                result = st.session_state["vector_store"].switch_database(selected_db)
            
            if result.get("success", False):
                st.sidebar.success(result["message"])
                st.rerun()
            else:
                st.sidebar.error(result.get("message", "切换失败"))

    new_db_name = st.sidebar.text_input("创建新数据库", st.session_state["create_db_name"])
    if st.sidebar.button("创建数据库"):
        st.session_state["create_db_name"] = new_db_name
        if run_mode == "🌐 API 模式":
            result = api_call("/databases", "POST", {"db_name": new_db_name})
        else:
            result = st.session_state["vector_store"].create_database(new_db_name)
        
        if result.get("success", False):
            st.sidebar.success(result["message"])
            st.session_state["create_db_name"] = ""
            st.rerun()
        else:
            st.sidebar.error(result.get("message", "创建失败"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已上传文件数", stats.get("total_files", 0))
    with col2:
        st.metric("总Chunks数", stats.get("total_chunks", 0))
    with col3:
        st.metric("当前数据库", stats.get("db_name", stats.get("current_db", "未知")))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📤 上传文件", "📋 文件列表", "🗑️ 数据库管理"])

    with tab1:
        st.subheader("上传文件到知识库")
        st.write(f"支持的文件类型：{', '.join(SUPPORTED_FILE_TYPES)}")
        
        uploaded_files = st.file_uploader(
            "拖拽文件到此处或点击选择文件",
            type=[t[1:] for t in SUPPORTED_FILE_TYPES],
            accept_multiple_files=True,
            help="支持拖拽上传多个文件",
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state['upload_key']}"
        )

        if uploaded_files:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            has_changes = False

            for i, uploaded_file in enumerate(uploaded_files, 1):
                status_text.text(f"正在处理文件 {i}/{total_files}: {uploaded_file.name}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    if run_mode == "🌐 API 模式":
                        with open(temp_file_path, "rb") as f:
                            result = api_call("/files/upload", "POST", files={"file": (uploaded_file.name, f)})
                    else:
                        result = st.session_state["vector_store"].upload_file(temp_file_path, uploaded_file.name)
                    
                    if result.get("success", False):
                        has_changes = True
                        st.success(f"✅ {result.get('message', '')}")
                        with st.expander(f"查看详情"):
                            st.write(f"📄 切分的Chunks数量: {result.get('chunks', '未知')}")
                            st.write(f"🔍 文件MD5: {result.get('md5', '未知')}")
                    else:
                        st.warning(f"⚠️ {result.get('message', '上传失败')}")
                finally:
                    os.unlink(temp_file_path)
                
                progress_bar.progress(i / total_files)
            
            status_text.text("上传完成！")
            if has_changes:
                st.session_state["upload_key"] += 1
                st.rerun()

    with tab2:
        st.subheader("已上传文件列表")
        
        if run_mode == "🌐 API 模式":
            uploaded_files = api_call("/files", "GET")
            if not isinstance(uploaded_files, list):
                uploaded_files = []
        else:
            uploaded_files = st.session_state["vector_store"].get_uploaded_files()

        data_categories = []
        if run_mode == "🖥️ 本地模式":
            data_categories = st.session_state["vector_store"].get_data_categories()
        
        if data_categories:
            st.markdown("### 本地数据目录")
            for category in data_categories:
                category_path = os.path.join(chroma_cfg['data_path'], category)
                if run_mode == "🖥️ 本地模式":
                    file_count = len(st.session_state["vector_store"].get_data_files(category))
                else:
                    file_count = 0
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"- **{category}** ({file_count} 个文件)")
                with col2:
                    if run_mode == "🖥️ 本地模式":
                        button_text = "一键自动同步导入根目录文件" if category == st.session_state["vector_store"].default_db_name else f"导入 {category}"
                        if st.button(button_text, key=f"import_cat_{category}"):
                            with st.spinner(f"正在导入数据目录 {category} ..."):
                                result = st.session_state["vector_store"].import_data_category(category)
                                if result["success"]:
                                    st.success(result["message"])
                                else:
                                    st.error(result["message"])
                                st.rerun()
            st.divider()

        if uploaded_files:
            for idx, file_info in enumerate(uploaded_files, 1):
                md5 = file_info.get("md5") if isinstance(file_info, dict) else str(file_info)
                file_name = file_info.get("file_name", md5) if isinstance(file_info, dict) else md5
                chunks = file_info.get("chunks", "未知") if isinstance(file_info, dict) else "未知"
                uploaded_at = file_info.get("uploaded_at", "") if isinstance(file_info, dict) else ""
                short_md5 = f"{md5[:8]}..."
                is_deleting = st.session_state["delete_confirm"] and st.session_state["delete_target"] == md5

                with st.expander(f"{idx}. {file_name}", expanded=is_deleting):
                    st.markdown(
                        f"- **文件名**：{file_name}\n- **MD5**：`{short_md5}`\n- **Chunks**：{chunks}"
                        + (f"\n- **上传时间**：{uploaded_at}" if uploaded_at else "")
                    )
                    
                    if is_deleting:
                        st.warning("⚠️ 确认删除此文件？此操作不可撤销！")
                        del_col1, del_col2 = st.columns(2)
                        with del_col1:
                            if st.button("确认删除", key=f"confirm_delete_{md5}"):
                                if run_mode == "🌐 API 模式":
                                    result = api_call(f"/files/{md5}", "DELETE")
                                else:
                                    result = st.session_state["vector_store"].remove_file(md5)
                                
                                if result.get("success", False):
                                    st.success(result.get("message", "删除成功"))
                                else:
                                    st.error(result.get("message", "删除失败"))
                                st.session_state["delete_confirm"] = False
                                st.session_state["delete_target"] = None
                                st.rerun()
                        with del_col2:
                            if st.button("取消", key=f"cancel_delete_{md5}"):
                                st.session_state["delete_confirm"] = False
                                st.session_state["delete_target"] = None
                    else:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if run_mode == "🖥️ 本地模式":
                                if st.button("🪄 重新解析", key=f"reparse_{md5}"):
                                    st.session_state["reparse_target"] = md5
                                    st.session_state["reparse_file_name"] = file_name
                                    st.session_state["delete_confirm"] = False
                                    st.session_state["reparse_message"] = ""
                                    st.rerun()
                        with col2:
                            if st.button("🗑️ 删除", key=f"delete_{md5}"):
                                st.session_state["delete_target"] = md5
                                st.session_state["delete_confirm"] = True
                                st.session_state["reparse_target"] = None
        elif stats.get("total_chunks", 0) > 0:
            st.warning("⚠️ 数据库中存在向量数据，但文件元数据信息丢失")
        else:
            st.info("暂无已上传的文件")

        if st.session_state["reparse_target"] and run_mode == "🖥️ 本地模式":
            st.divider()
            st.subheader("重新解析文件")
            st.info(
                f"当前目标文件：{st.session_state['reparse_file_name']} (MD5: {st.session_state['reparse_target']})"
            )
            uploaded_reparse_file = st.file_uploader(
                "请上传要重新解析的文件",
                type=[t[1:] for t in SUPPORTED_FILE_TYPES],
                accept_multiple_files=False,
                key="reparse_upload",
                help="上传与目标文件对应的原始文件以重新解析。"
            )

            if uploaded_reparse_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_reparse_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_reparse_file.getvalue())
                    temp_file_path = temp_file.name

                try:
                    file_md5 = get_file_md5_hex(temp_file_path)
                    if file_md5 != st.session_state["reparse_target"]:
                        st.error("上传文件的 MD5 与目标文件不匹配，请确认上传的是同一个文件。")
                    else:
                        result = st.session_state["vector_store"].reparse_file(temp_file_path, uploaded_reparse_file.name)
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                        st.session_state["reparse_target"] = None
                        st.session_state["reparse_file_name"] = None
                        st.session_state["reparse_message"] = result.get("message", "")
                        st.rerun()
                finally:
                    os.unlink(temp_file_path)

            if st.button("取消重新解析", key="cancel_reparse"):
                st.session_state["reparse_target"] = None
                st.session_state["reparse_file_name"] = None
                st.session_state["reparse_message"] = ""
                st.rerun()

    with tab3:
        st.subheader("数据库管理")
        
        st.write("### 当前数据库信息")
        st.write(f"- 数据库名称: {current_db}")
        st.write(f"- 存储路径: {chroma_cfg.get('persist_directory', '')}/{current_db}")
        st.write(f"- 已上传文件: {stats.get('total_files', 0)} 个")
        st.write(f"- 总Chunks: {stats.get('total_chunks', 0)} 个")

        st.divider()
        
        st.write("### 危险操作")
        if current_db != chroma_cfg.get('default_database_name', 'default'):
            if len(db_list) > 1:
                if st.button("🗑️ 删除当前数据库", key="delete_db"):
                    st.session_state["db_delete_confirm"] = True
            else:
                st.warning("⚠️ 至少需要保留一个数据库，无法删除")
        else:
            st.info("默认数据库不可删除")
            st.session_state["db_delete_confirm"] = False

        if st.session_state["db_delete_confirm"]:
            st.error(f"""
            ⚠️ **警告！此操作将删除数据库 `{current_db}`！**
            
            以下内容将被永久删除：
            - 所有已上传的文件记录
            - 所有向量数据
            - 所有索引信息
            
            此操作**不可撤销**！
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                confirm_text = st.text_input(f"请输入 '{current_db}' 确认删除", key="confirm_text")
                if confirm_text == current_db:
                    if st.button("确认删除数据库", key="final_delete_db"):
                        if run_mode == "🌐 API 模式":
                            result = api_call(f"/databases/{current_db}", "DELETE")
                        else:
                            result = st.session_state["vector_store"].delete_database()
                        
                        if result.get("success", False):
                            st.success(result.get("message", "删除成功"))
                            new_db_list = [db for db in db_list if db != current_db]
                            if new_db_list:
                                if run_mode == "🌐 API 模式":
                                    api_call(f"/databases/{new_db_list[0]}/switch", "POST")
                                else:
                                    st.session_state["vector_store"].switch_database(new_db_list[0])
                        else:
                            st.error(result.get("message", "删除失败"))
                        st.session_state["db_delete_confirm"] = False
                        st.rerun()
            with col2:
                if st.button("取消", key="cancel_db_delete"):
                    st.session_state["db_delete_confirm"] = False
                    st.rerun()

    st.divider()
    st.subheader("操作提示")
    st.info("""
    - 📤 上传的文件会被自动切分为多个Chunk并存储到向量数据库
    - 🔍 相同的文件（通过MD5校验）不会被重复上传
    - 📋 在文件列表中可以查看、删除已上传的文件
    - 🗑️ 删除数据库需要二次确认，操作前请谨慎
    - 📁 支持多数据库管理，可在左侧创建和切换数据库
    - 📄 支持文件类型：txt, pdf, docx, md, xlsx, xls, csv, html, htm
    """)


def main():
    st.sidebar.title("导航菜单")
    page = st.sidebar.radio("选择页面", ["智能问答助手", "知识库管理"])

    if page == "智能问答助手":
        show_chat_page()
    elif page == "知识库管理":
        show_upload_page()


if __name__ == "__main__":
    main()