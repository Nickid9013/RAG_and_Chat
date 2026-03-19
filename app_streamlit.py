import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
import json
import html

# 深色主题 + 聊天布局（支持多轮对话）
st.set_page_config(
    page_title="RAG 年报问答",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 深色主题样式：背景与对话气泡明显区分
st.markdown("""
<style>
    /* 整体深色背景 */
    .stApp { background-color: #1a1a1f; }
    .stApp header { background-color: transparent !important; }
    
    /* 左侧边栏 */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #252530 0%, #1a1a1f 100%); }
    [data-testid="stSidebar"] .stMarkdown { color: #e4e4e7; }
    
    /* 主区域容器 */
    .main-content { 
        max-width: 820px; 
        margin: 0 auto; 
        padding: 24px 0 32px 0;
    }
    
    /* 当前主题标题（主区顶部） */
    .topic-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e4e4e7;
        margin-bottom: 20px;
        padding-bottom: 8px;
    }
    
    /* 占位提示 */
    .placeholder-hint {
        text-align: center;
        color: #71717a;
        padding: 60px 20px;
        font-size: 0.95rem;
    }
    
    /* 对话气泡与背景明显区分 */
    [data-testid="stChatMessage"] {
        background: #252530 !important;
        border: 1px solid #3d3d45;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }
    
    /* 纯文本展示区域（不解析 Markdown） */
    .chat-plain-text {
        color: #e4e4e7;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* 用户消息气泡靠右：右侧列内的气泡不拉伸 */
    [data-testid="column"]:last-child [data-testid="stChatMessage"] {
        margin-left: auto;
        max-width: 85%;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 pipeline（放 session_state 避免重复创建）
if "pipeline" not in st.session_state:
    root_path = Path("data/stock_data")
    st.session_state.pipeline = Pipeline(root_path, run_config=max_config)
if "messages" not in st.session_state:
    # messages: [{"role": "user"|"assistant", "content": str, ...}]
    st.session_state.messages = []
if "qa_history" not in st.session_state:
    # qa_history: 传给后端的多轮对话（user/assistant），不包含检索上下文
    st.session_state.qa_history = []


def _plain_text_html(text: str) -> str:
    """将内容转为纯文本展示的 HTML，不解析 Markdown（如 # 章节一 按原文显示）。"""
    if not text:
        return ""
    escaped = html.escape(text)
    return f'<div class="chat-plain-text">{escaped.replace(chr(10), "<br>")}</div>'


def _strip_md_json(s):
    s = (s or "").strip()
    if s.startswith("```") and "```" in s[3:]:
        a, b = s.find("```") + 3, s.rfind("```")
        n = s.find("\n", s.find("```") + 3)
        if n > 0:
            a = n + 1
        if b > a:
            s = s[a:b].strip()
    return s


def _parse_answer(answer):
    """解析 pipeline 返回为 (step_by_step, reasoning_summary, relevant_pages, final_answer)，失败返回 (None, None, None, None)。"""
    answer_dict = {}
    if isinstance(answer, str):
        try:
            answer_dict = json.loads(_strip_md_json(answer))
        except Exception:
            return None, None, None, None
    else:
        answer_dict = answer if isinstance(answer, dict) else {}

    if "step_by_step_analysis" in answer_dict or "reasoning_summary" in answer_dict:
        struct = answer_dict
    elif isinstance(answer_dict.get("content"), dict):
        struct = answer_dict["content"]
    else:
        raw = answer_dict.get("final_answer") or answer_dict.get("content") or ""
        if isinstance(raw, str) and raw.strip():
            try:
                struct = json.loads(_strip_md_json(raw))
            except Exception:
                struct = {}
        else:
            struct = answer_dict
    return (
        struct.get("step_by_step_analysis", "-"),
        struct.get("reasoning_summary", "-"),
        struct.get("relevant_pages", []),
        struct.get("final_answer", "-"),
    )


# ---------- 左侧边栏 ----------
with st.sidebar:
    st.markdown("### 📄 RAG 年报问答")
    st.markdown("---")
    if st.button("🔄 开启新对话", use_container_width=True, type="primary"):
        # 仅清理会话数据，保留 pipeline
        st.session_state.messages = []
        st.session_state.qa_history = []
        st.rerun()
    st.markdown("---")
    st.caption("基于公司年报的 RAG 问答 · 向量检索 + LLM 推理")


# ---------- 主内容区 ----------
main = st.container()
with main:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.markdown('<p class="topic-title">对话</p>', unsafe_allow_html=True)

    # 渲染历史消息（用户气泡在右侧，助手在左侧；纯文本显示）
    if st.session_state.messages:
        for m in st.session_state.messages:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            if role == "user":
                col_left, col_right = st.columns([1, 2])
                with col_left:
                    st.write("")
                with col_right:
                    with st.chat_message(role):
                        st.markdown(_plain_text_html(content or ""), unsafe_allow_html=True)
            else:
                with st.chat_message(role):
                    st.markdown(_plain_text_html(content or ""), unsafe_allow_html=True)
                    step_by_step = m.get("step_by_step_analysis")
                    reasoning_summary = m.get("reasoning_summary")
                    relevant_pages = m.get("relevant_pages")
                    if step_by_step is not None or reasoning_summary is not None or relevant_pages is not None:
                        with st.expander("查看推理与引用", expanded=False):
                            if step_by_step is not None:
                                st.markdown("**分步推理**")
                                st.markdown(_plain_text_html(str(step_by_step)), unsafe_allow_html=True)
                            if reasoning_summary is not None:
                                st.markdown("**推理摘要**")
                                st.markdown(_plain_text_html(str(reasoning_summary)), unsafe_allow_html=True)
                            if relevant_pages is not None:
                                st.markdown("**相关页面**")
                                st.write(relevant_pages if relevant_pages else "-")
    else:
        st.markdown('<p class="placeholder-hint">在下方输入问题并回车发送，开始对话。</p>', unsafe_allow_html=True)

    # 输入区：回车发送（不换行）
    prompt = st.chat_input("输入问题并回车发送（Shift+Enter 可换行）")
    if prompt and prompt.strip():
        user_question = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("正在检索与生成答案..."):
            try:
                history = st.session_state.get("qa_history", [])
                answer, new_history = st.session_state.pipeline.answer_single_question(
                    user_question,
                    kind="string",
                    history=history,
                )
                st.session_state.qa_history = new_history
                step_by_step, reasoning_summary, relevant_pages, final_answer = _parse_answer(answer)
                if step_by_step is None:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "返回内容无法解析为结构化答案。"}
                    )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final_answer,
                            "step_by_step_analysis": step_by_step,
                            "reasoning_summary": reasoning_summary,
                            "relevant_pages": relevant_pages,
                        }
                    )
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"生成答案时出错: {e}"})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
