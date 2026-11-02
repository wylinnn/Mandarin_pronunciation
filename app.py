Python 3.14.3 (v3.14.3:323c59a5e34, Feb  3 2026, 11:41:37) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
import streamlit as st
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import edge_tts
import asyncio
import io
import tempfile
import os

from pypinyin import lazy_pinyin, Style


# =====================================================
# 1. 網站基本設定
# =====================================================

st.set_page_config(
    page_title="華語聲調練習與比對系統",
    page_icon="🎧",
    layout="wide"
)

st.title("華語聲調練習與比對系統")

st.markdown(
    "聆聽台灣華語標準發音，錄製自己的聲音，"
    "比較兩者的音高變化。"
)


# =====================================================
# 2. 語音設定
# =====================================================

VOICES = {
    "女性｜曉辰": "zh-TW-HsiaoChenNeural",
    "男性｜雲哲": "zh-TW-YunJheNeural"
}


# =====================================================
# 3. Edge TTS 語音生成
# =====================================================

async def generate_edge_tts(text, voice):

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    audio_data = bytearray()

    async for chunk in communicate.stream():

        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])

    return bytes(audio_data)


@st.cache_data(show_spinner=False, ttl=3600)
def get_reference_audio(text, voice):

    return asyncio.run(
        generate_edge_tts(text, voice)
    )


# =====================================================
# 4. 語音播放速度調整
# =====================================================

@st.cache_data(show_spinner=False)
def change_playback_speed(audio_bytes, rate):

    rate = float(rate)

    if rate == 1.0:
        return audio_bytes, "audio/mpeg"

    # 將音檔暫存
    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ) as temp:

        temp.write(audio_bytes)
        temp_path = temp.name

    try:

        # 讀取音訊
        y, sr = librosa.load(
            temp_path,
            sr=None,
            mono=True
        )

        # 保留音高的時間伸縮
        slowed_audio = librosa.effects.time_stretch(
            y,
            rate=rate
        )

        # 輸出 WAV
        output = io.BytesIO()

        sf.write(
            output,
            slowed_audio,
            sr,
            format="WAV",
            subtype="PCM_16"
        )

        return output.getvalue(), "audio/wav"

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# 5. 讀取音訊
# =====================================================

def load_audio(audio_bytes, suffix):

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp:

        temp.write(audio_bytes)
        temp_path = temp.name

    try:

        y, sr = librosa.load(
            temp_path,
            sr=16000,
            mono=True
        )

        return y, sr

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# 6. 擷取音高曲線
# =====================================================

def extract_pitch_contour(audio_bytes, suffix=".wav"):

    try:

        y, sr = load_audio(audio_bytes, suffix)

        if len(y) == 0:
            return None

        duration = len(y) / sr

        if duration < 0.15:
            return None

        sound = parselmouth.Sound(
            y,
            sampling_frequency=sr
        )

        pitch = sound.to_pitch_ac(
            time_step=0.01,
            pitch_floor=65,
            pitch_ceiling=500,
            voicing_threshold=0.45
        )

        f0 = pitch.selected_array["frequency"]
        times = pitch.xs()

        # 有效發聲區間
        voiced_mask = (
            (f0 > 0) &
            np.isfinite(f0) &
            (f0 >= 65) &
            (f0 <= 500)
        )

        voiced_ratio = np.mean(voiced_mask)

        if voiced_ratio < 0.45:
            return None

        voiced_f0 = f0[voiced_mask]
        voiced_times = times[voiced_mask]

        if len(voiced_f0) < 10:
            return None

        # 去除極端音高值
        lower = np.percentile(voiced_f0, 5)
        upper = np.percentile(voiced_f0, 95)

        valid = (
            (voiced_f0 >= lower) &
            (voiced_f0 <= upper)
        )

        voiced_f0 = voiced_f0[valid]
        voiced_times = voiced_times[valid]

        if len(voiced_f0) < 10:
            return None

        # 使用說話者自身的中位數作為音高基準
        median_f0 = np.median(voiced_f0)

        # 轉換成相對音高（半音）
        semitones = 12 * np.log2(
            voiced_f0 / median_f0
        )

        # 標準化時間軸
        normalized_time = np.linspace(
            0,
            100,
            100
        )

        original_time = (
            (voiced_times - voiced_times[0]) /
            (voiced_times[-1] - voiced_times[0])
        ) * 100

        contour = np.interp(
            normalized_time,
            original_time,
            semitones
        )

        return {
            "time": normalized_time,
            "contour": contour,
            "voiced_ratio": voiced_ratio,
            "median_f0": median_f0,
            "duration": duration
        }

    except Exception as e:

        st.error(f"音高分析錯誤：{e}")

        return None


# =====================================================
# 7. 漢語拼音及聲調顯示
# =====================================================

def get_pinyin_info(text):

    syllables = lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True
    )

    results = []

    for syllable in syllables:

        if not syllable:
            continue

        last = syllable[-1]

        if last in "12345":

            tone_number = int(last)

            tone_names = {
                1: "第一聲",
                2: "第二聲",
                3: "第三聲",
                4: "第四聲",
                5: "輕聲"
            }

            results.append({
                "pinyin": syllable,
                "tone": tone_names[tone_number]
            })

        else:

            results.append({
                "pinyin": syllable,
                "tone": "無法判定"
            })

    return results


# =====================================================
# 8. DTW 時間對齊及曲線比較
# =====================================================

def compare_contours(reference, student):

    ref = reference["contour"]
    stu = student["contour"]

    # DTW 動態時間校準
    D, wp = librosa.sequence.dtw(
        X=ref.reshape(1, -1),
        Y=stu.reshape(1, -1),
        metric="euclidean"
    )

    # 調整路徑方向
    wp = wp[::-1]

    aligned_ref = ref[wp[:, 0]]
    aligned_stu = stu[wp[:, 1]]

    difference = aligned_ref - aligned_stu

    mae = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(difference ** 2))

    return {
        "aligned_ref": aligned_ref,
        "aligned_student": aligned_stu,
        "mae": mae,
        "rmse": rmse,
        "path": wp
    }


# =====================================================
# 9. 側邊設定
# =====================================================

with st.sidebar:

    st.header("練習設定")

    selected_voice = st.selectbox(
        "標準發音",
        list(VOICES.keys())
    )

    voice = VOICES[selected_voice]

    speed = st.select_slider(
        "播放速度",
        options=[0.75, 0.85, 1.0, 1.25],
        value=1.0,
        format_func=lambda x: f"{x}×"
    )

    st.divider()

    st.caption(
        "建議先聆聽標準發音，"
        "再錄製自己的發音。"
    )


# =====================================================
# 10. 輸入練習內容
# =====================================================

st.subheader("1. 輸入練習字詞")

word = st.text_input(
    "請輸入中文字",
    placeholder="例如：媽、馬、老師、你好",
    max_chars=30
)

if word:

    st.divider()

    # =================================================
    # 11. 顯示拼音
    # =================================================

    st.subheader("2. 拼音及目標聲調")

    pinyin_info = get_pinyin_info(word)

    columns = st.columns(
        max(1, len(pinyin_info))
    )

    for i, item in enumerate(pinyin_info):

        with columns[i]:

            st.metric(
                label=item["pinyin"],
                value=item["tone"]
            )

    st.caption(
        "拼音為詞語的字典讀音；"
        "尚未自動處理連續語流中的聲調變化。"
    )

    # =================================================
    # 12. 產生標準語音
    # =================================================

    st.divider()

    st.subheader("3. 聆聽標準發音")

    try:

        with st.spinner("正在生成標準語音..."):

            reference_audio = get_reference_audio(
                word,
                voice
            )

        playback_audio, audio_format = (
            change_playback_speed(
                reference_audio,
                speed
            )
        )

        st.audio(
            playback_audio,
            format=audio_format
        )

        st.caption(
            f"聲音：{selected_voice}｜"
            f"播放速度：{speed}×"
        )

    except Exception as e:

        st.error(
            f"語音生成失敗：{e}"
        )

        st.stop()

    # =================================================
    # 13. 學生錄音
    # =================================================

    st.divider()

    st.subheader("4. 錄製自己的發音")

    student_audio = st.audio_input(
        "點擊麥克風開始錄音",
        key="student_recording"
    )

    if student_audio is not None:

        student_bytes = student_audio.getvalue()

        st.audio(
            student_bytes,
            format="audio/wav"
        )

        # =================================================
        # 14. 音高分析
        # =================================================

        st.divider()

        st.subheader("5. 聲調曲線分析")

        with st.spinner("正在分析音高曲線..."):

            ref_result = extract_pitch_contour(
                reference_audio,
                suffix=".mp3"
            )

            stu_result = extract_pitch_contour(
                student_bytes,
                suffix=".wav"
            )

        if ref_result is None:

            st.error(
                "無法分析標準語音，請重新載入頁面。"
            )

        elif stu_result is None:

            st.warning(
                "無法取得足夠的有效音高資料。"
                "請在安靜的環境重新錄音，"
                "並確保完整念出字詞。"
            )

        else:

            # =================================================
            # 15. 品質檢查
            # =================================================

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "錄音長度",
                f"{stu_result['duration']:.2f} 秒"
            )

            col2.metric(
                "有效發聲比例",
                f"{stu_result['voiced_ratio'] * 100:.1f}%"
            )

            col3.metric(
                "學生音高基準",
                f"{stu_result['median_f0']:.1f} Hz"
            )

            # =================================================
            # 16. 原始音高曲線
            # =================================================

            st.markdown("#### 音高曲線")

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.plot(
                ref_result["time"],
                ref_result["contour"],
                label="Standard",
                color="blue",
                linewidth=2.5,
                linestyle="--"
            )

            ax.plot(
                stu_result["time"],
                stu_result["contour"],
                label="Student",
                color="red",
                linewidth=2
            )

            ax.set_xlabel(
                "Normalized Time (%)"
            )

            ax.set_ylabel(
                "Relative Pitch (Semitones)"
            )

            ax.set_title(
                "Pitch Contour Comparison"
            )

            ax.legend()

            ax.grid(
                True,
                linestyle=":",
                alpha=0.5
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            # =================================================
            # 17. DTW 比較
            # =================================================

            st.markdown("#### 時間對齊後的曲線比較")

            comparison = compare_contours(
                ref_result,
                stu_result
            )

            aligned_ref = comparison["aligned_ref"]
            aligned_stu = comparison["aligned_student"]

            fig2, ax2 = plt.subplots(
                figsize=(10, 4)
            )

            x = np.linspace(
                0,
                100,
                len(aligned_ref)
            )

            ax2.plot(
                x,
                aligned_ref,
                label="Standard",
                color="blue",
                linewidth=2.5,
                linestyle="--"
            )

            ax2.plot(
                x,
                aligned_stu,
                label="Student",
                color="red",
                linewidth=2
            )

            ax2.set_xlabel(
                "Aligned Time (%)"
            )

            ax2.set_ylabel(
                "Relative Pitch (Semitones)"
            )

            ax2.set_title(
                "DTW Aligned Pitch Comparison"
            )

            ax2.legend()

            ax2.grid(
                True,
                linestyle=":",
                alpha=0.5
            )

            fig2.tight_layout()

            st.pyplot(
                fig2,
                use_container_width=True
            )

            plt.close(fig2)

            # =================================================
            # 18. 比較數據
            # =================================================

            st.markdown("#### 音高差異")

            col1, col2 = st.columns(2)

            col1.metric(
                "平均絕對差異（MAE）",
                f"{comparison['mae']:.2f} semitones"
            )

            col2.metric(
                "均方根差異（RMSE）",
                f"{comparison['rmse']:.2f} semitones"
            )

            st.info(
                "以上數值代表時間對齊後的音高曲線差異，"
                "不是聲調正確率，也不是發音評分。"
                "數值越小代表兩條相對音高曲線越接近，"
                "但不能單獨據此判定聲調是否正確。"
            )

            # =================================================
            # 19. 額外資訊
...             # =================================================
... 
...             with st.expander("查看音高分析資訊"):
... 
...                 st.write(
...                     "標準語音音高基準：",
...                     f"{ref_result['median_f0']:.2f} Hz"
...                 )
... 
...                 st.write(
...                     "學生語音音高基準：",
...                     f"{stu_result['median_f0']:.2f} Hz"
...                 )
... 
...                 st.write(
...                     "標準語音有效發聲比例：",
...                     f"{ref_result['voiced_ratio'] * 100:.2f}%"
...                 )
... 
...                 st.write(
...                     "學生語音有效發聲比例：",
...                     f"{stu_result['voiced_ratio'] * 100:.2f}%"
...                 )
... 
... else:
... 
...     st.info(
...         "請先輸入一個中文字或詞語，開始聲調練習。"
...     )
... 
... 
... # =====================================================
... # 20. 頁尾
... # =====================================================
... 
... st.divider()
... 
... st.caption(
...     "Mandarin Pronunciation Practice System"
