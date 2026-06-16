import streamlit as st
import yt_dlp
import whisper
from openai import OpenAI
import json
import os
import warnings
import cv2
import base64
import subprocess

# 忽略不必要的底层警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. 后端核心逻辑模块
# ==========================================

@st.cache_resource
def load_whisper_model():
    """加载语音识别模型（全局缓存，只加载一次）"""
    # 如果觉得 small 慢，可以改为 "base"
    return whisper.load_model("small")


def run_video_download(video_url):
    """【通道A】下载网络视频，保留原视频用于抽帧，并剥离音频"""
    output_name = "current_task_audio"

    # 清理历史残留文件，防止干扰
    for f in os.listdir("."):
        if f.startswith(output_name):
            try:
                os.remove(f)
            except:
                pass

    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{output_name}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'keepvideo': True,  # 保留原视频，给后面的 OpenCV 截图用
        'quiet': True,
        'socket_timeout': 60,
        'retries': 10,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # 寻找保留下来的视频文件路径
    video_file = None
    for f in os.listdir("."):
        if f.startswith(output_name) and not f.endswith(".mp3"):
            video_file = f
            break

    return f"{output_name}.mp3", video_file


def extract_audio_from_local(uploaded_file):
    """【通道B】从本地视频中剥离音频，并返回临时视频路径"""
    temp_video_path = f"temp_uploaded_video{os.path.splitext(uploaded_file.name)[1]}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    output_audio_path = "current_task_audio.mp3"
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    # 调用底层 FFmpeg 分离音频
    cmd = f'ffmpeg -i "{temp_video_path}" -vn -acodec libmp3lame -ar 16000 -ab 128k -y "{output_audio_path}"'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_audio_path, temp_video_path


def extract_keyframes(video_path, interval_sec=5, max_frames=6):
    """📸 视觉引擎：提取视频关键帧"""
    if not video_path or not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    frame_interval = int(fps * interval_sec)
    base64_frames = []
    count = 0

    while cap.isOpened() and len(base64_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            # 压缩尺寸，避免大模型拒收过大的图库
            frame_resized = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame_resized)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_str)

        count += 1

    cap.release()
    return base64_frames


def run_whisper_transcribe(audio_path, model_instance):
    """🎙️ 模块二：Whisper 全球语音自适应识别"""
    # 去除语言限制，自动侦测语种
    result = model_instance.transcribe(audio_path)
    detected_lang = result.get("language", "未知语言")
    return result["text"], detected_lang


def run_multimodal_extract(transcript_text, base64_frames, api_key):
    """🧠 模块三：通义千问多模态大模型 (Qwen-VL) 图文交叉提取并自动翻译"""
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    system_prompt = """
    你是一个资深的跨境电商选品专家。请结合我提供的【视频背景语音字幕】以及【视频关键帧画面】，提取出画面和语音中真实存在的、具有商业潜力的实体物品（例如：运动装备、服饰、手工艺品、工具等）。

    ⚠️ 铁律：
    1. 必须完全基于画面内容或字幕内容提取，绝对不能凭空捏造！
    2. 如果画面和字幕中完全没有明显商品（例如只有风景、纯文字PPT），请直接输出空列表：[]
    3. 无论提供的字幕是什么语言，请务必将其翻译理解后，统一使用【简体中文】输出商品名称和卖点特征！

    严格按照以下 JSON 格式输出，不要带任何 markdown 标记：
    [
        {
            "product_name": "中文物品名称",
            "features": "中文卖点（提取视频或画面中体现的材质、颜色、风格或受众）",
            "timestamp": "在视频中出现的位置或状态"
        }
    ]
    """

    # 构建图文混排的数据包
    user_content = [
        {"type": "text", "text": f"【视频语音字幕如下】\n{transcript_text}\n\n【视频关键帧截图已附带，请结合分析】"}
    ]
    for b64 in base64_frames:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content


# ==========================================
# 2. 前端可视化 UI 界面
# ==========================================

st.set_page_config(page_title="AI 多模态选品台", page_icon="🌍", layout="wide")

# 注入 CSS 美化卡片
st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #f8f9fa; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #ff4b4b; transition: transform 0.2s;
    }
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"]:hover {
        transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- 侧边栏控制台 -----------------
with st.sidebar:
    st.header("⚙️ 引擎控制台")
    st.markdown("**阿里云 API 配置**")
    user_api_key = st.text_input("Qwen-VL API Key", type="password", help="输入百炼平台申请的API Key")

    st.divider()
    st.markdown("🛠️ **调试开关**")
    show_raw_json = st.toggle("显示底层 JSON 源码", value=False)

    if user_api_key:
        st.success("API Key 已加载")
    else:
        st.warning("请先填入 API Key")

# ----------------- 主界面 -----------------
st.title("🌍 AI 多模态溯源选品智能体")
st.markdown(
    "<span style='color:#666; font-size: 1.1em;'>搭载多模态视觉引擎与全球语种自适应，真正实现“所见即所得”的商业级提取。</span>",
    unsafe_allow_html=True)

whisper_model = load_whisper_model()

tab1, tab2 = st.tabs(["🔗 载入网络链接", "📤 解析本地文件"])
source_type, video_input_data = None, None

with tab1:
    url_input = st.text_input("输入流媒体链接 (YouTube / Bilibili 等)", placeholder="https://...")
    if url_input:
        source_type, video_input_data = "url", url_input

with tab2:
    file_input = st.file_uploader("拖拽视频文件到此处 (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"])
    if file_input:
        source_type, video_input_data = "file", file_input

# ----------------- 触发流水线 -----------------
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    start_btn = st.button("🚀 启动全自动化选品流水线", type="primary", use_container_width=True)

if start_btn:
    if not source_type:
        st.error("⚠️ 请先提供视频源（输入链接或上传文件）！")
    elif not user_api_key:
        st.error("⚠️ 请先在左侧边栏填入通义千问的 API Key！")
    else:
        st.toast("多模态流水线已启动...", icon="⏳")
        target_video_path = None

        with st.status("⚙️ 正在执行多模态解析流水线...", expanded=True) as status:

            # --- 第一步：视频处理 ---
            try:
                if source_type == "url":
                    st.write("📥 1/3 正在下载网络视频并剥离音轨...")
                    audio_file, target_video_path = run_video_download(video_input_data)
                else:
                    st.write("📥 1/3 正在解析本地视频并提取音频流...")
                    audio_file, target_video_path = extract_audio_from_local(video_input_data)
            except Exception as e:
                status.update(label="视频处理失败", state="error")
                st.error(f"处理失败: {e}")
                st.stop()

            # --- 第二步：语音听写 ---
            st.write("🎙️ 2/3 Whisper 模型正在侦测语种并提取字幕...")
            transcript, detected_lang = run_whisper_transcribe(audio_file, whisper_model)
            st.write(f"🌐 自动侦测到视频语言: **{str(detected_lang).upper()}**")

            # --- 第二点五步：视觉引擎抽帧 ---
            st.write("📸 2.5/3 OpenCV 视觉引擎正在截取视频关键帧...")
            frames = extract_keyframes(target_video_path, interval_sec=5, max_frames=6)
            st.write(f"✅ 成功截取 {len(frames)} 张关键画面提供给大模型！")

            # --- 第三步：视觉大模型 ---
            st.write("🧠 3/3 视觉大模型 (Qwen-VL) 正在进行【图文交叉比对翻译】...")
            try:
                llm_output = run_multimodal_extract(transcript, frames, user_api_key)
                status.update(label="✨ 全流水线解析完成！", state="complete", expanded=False)
            except Exception as e:
                status.update(label="大模型请求失败", state="error")
                st.error(f"调用视觉大模型失败，请检查 API Key 或网络: {e}")
                st.stop()
            finally:
                # 阅后即焚：清理占用空间的临时视频文件
                if target_video_path and os.path.exists(target_video_path):
                    os.remove(target_video_path)

        st.balloons()

        # ----------------- 结果展示 -----------------
        st.divider()
        with st.expander("📝 点击查看视频原始听写字幕"):
            st.write(transcript if transcript.strip() else "未检测到清晰的人声语音。")

        st.subheader("📦 挖掘到的高潜选品卡片")

        try:
            clean_json = llm_output.strip().replace("```json", "").replace("```", "")
            product_list = json.loads(clean_json)

            if not product_list:
                st.info("💡 经视觉大模型严格交叉验证：该视频中未检测到明显的实体商品。")
            else:
                cols = st.columns(3)
                for idx, item in enumerate(product_list):
                    with cols[idx % 3]:
                        st.markdown(f"#### 🏷️ {item.get('product_name', '未知商品')}")
                        st.markdown(f"**核心卖点:** {item.get('features', '暂无')}")
                        st.markdown(f"⏱️ **位置:** `{item.get('timestamp', '未知')}`")

                if show_raw_json:
                    st.divider()
                    st.subheader("⚙️ 底层 JSON 数据")
                    st.json(product_list)

        except Exception as json_err:
            st.error("解析数据时出错，大模型未按规范输出 JSON。")
            with st.expander("查看原始输出与报错"):
                st.write(json_err)
                st.text_area("大模型原始输出", llm_output, height=200)