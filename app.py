import asyncio
import io
import os
import tempfile

import edge_tts
import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import soundfile as sf
import streamlit as st
from pypinyin import Style, lazy_pinyin


# =====================================================
# 1. 網站基本設定
# =====================================================

st.set_page_config(
    page_title="華語聲調練習與比對系統",
    page_icon="🎧",
    layout="wide",
)

st.title("華語聲調練習與比對系統")
st.markdown(
    "聆聽台灣華語標準發音，錄製自己的聲音，"
    "比較兩者的相對音高變化。"
)


# =====================================================
# 2. 語音設定
# =====================================================

VOICES = {
    "女性｜曉辰": "zh-TW-HsiaoChenNeural",
    "男性｜雲哲": "zh-TW-YunJheNeural",
}


# =====================================================
# 3. Edge TTS 語音生成
# =====================================================

async def _generate_edge_tts(text: str, voice: str) -> bytes:
    """透過 Edge TTS 產生 MP3 音訊。"""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio_data = bytearray()

    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio_data.extend(chunk["data"])

    if not audio_data:
        raise RuntimeError("語音服務沒有回傳音訊資料。")

    return bytes(audio_data)


@st.cache_data(show_spinner=False, ttl=3600)
def get_reference_audio(text: str, voice: str) -> bytes:
    """快取標準語音，避免每次互動都重新呼叫 TTS。"""
    return asyncio.run(_generate_edge_tts(text, voice))


# =====================================================
# 4. 音訊解碼
# =====================================================

def decode_audio(audio_bytes: bytes, suffix: str, target_sr=None):
    """將上傳或產生的音訊解碼成單聲道 NumPy 陣列。"""
    if not audio_bytes:
        raise ValueError("音訊資料是空的。")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        y, sr = librosa.load(
            temp_path,
            sr=target_sr,
            mono=True,
        )

        if y.size == 0:
            raise ValueError("無法從音檔讀取到有效音訊。")

        return y, sr

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# 5. 語音播放速度調整
# =====================================================

@st.cache_data(show_spinner=False)
def change_playback_speed(audio_bytes: bytes, rate: float):
    """
    調整播放速度。
    rate < 1 時使用 time-stretch，盡量保留原本音高。
    """
    rate = float(rate)

    if rate == 1.0:
        return audio_bytes, "audio/mpeg"

    y, sr = decode_audio(audio_bytes, suffix=".mp3", target_sr=None)

    slowed_audio = librosa.effects.time_stretch(
        y=y,
        rate=rate,
    )

    output = io.BytesIO()
    sf.write(
        output,
        slowed_audio,
        sr,
        format="WAV",
        subtype="PCM_16",
    )

    return output.getvalue(), "audio/wav"


# =====================================================
# 6. 擷取音高曲線
# =====================================================

def extract_pitch_contour(audio_bytes: bytes, suffix: str):
    """
    擷取 F0，轉換成相對於說話者自身中位數的半音曲線。
    回傳 dict；分析失敗時回傳 None。
    """
    try:
        y, sr = decode_audio(
            audio_bytes,
            suffix=suffix,
            target_sr=16000,
        )

        duration = len(y) / sr
        if duration < 0.15:
            return None

        sound = parselmouth.Sound(
            y,
            sampling_frequency=sr,
        )

        pitch = sound.to_pitch_ac(
            time_step=0.01,
            pitch_floor=65,
            pitch_ceiling=500,
            voicing_threshold=0.40,
        )

        f0 = pitch.selected_array["frequency"]
        times = pitch.xs()

        if f0.size == 0 or times.size != f0.size:
            return None

        voiced_mask = (
            (f0 > 0)
            & np.isfinite(f0)
            & (f0 >= 65)
            & (f0 <= 500)
        )

        voiced_ratio = float(np.mean(voiced_mask))
        voiced_f0 = f0[voiced_mask]
        voiced_times = times[voiced_mask]

        # 短詞可能只有少量有效音高框架，因此同時檢查比例與數量。
        if len(voiced_f0) < 8 or voiced_ratio < 0.20:
            return None

        # 排除極端值，降低偶發倍頻／半頻錯誤的影響。
        lower = np.percentile(voiced_f0, 2)
        upper = np.percentile(voiced_f0, 98)
        valid = (voiced_f0 >= lower) & (voiced_f0 <= upper)

        voiced_f0 = voiced_f0[valid]
        voiced_times = voiced_times[valid]

        if len(voiced_f0) < 8:
            return None

        median_f0 = float(np.median(voiced_f0))
        if not np.isfinite(median_f0) or median_f0 <= 0:
            return None

        semitones = 12.0 * np.log2(voiced_f0 / median_f0)

        time_span = float(voiced_times[-1] - voiced_times[0])
        if time_span <= 0:
            return None

        normalized_time = np.linspace(0.0, 100.0, 100)
        original_time = (
            (voiced_times - voiced_times[0]) / time_span
        ) * 100.0

        # np.interp 需要遞增的 x；去除重複時間點。
        unique_time, unique_indices = np.unique(
            original_time,
            return_index=True,
        )
        unique_semitones = semitones[unique_indices]

        if len(unique_time) < 2:
            return None

        contour = np.interp(
            normalized_time,
            unique_time,
            unique_semitones,
        )

        if not np.all(np.isfinite(contour)):
            return None

        return {
            "time": normalized_time,
            "contour": contour,
            "voiced_ratio": voiced_ratio,
            "median_f0": median_f0,
            "duration": float(duration),
        }

    except Exception as exc:
        # 保留錯誤供 Streamlit 顯示，避免將錯誤誤判為「沒有音高」。
        return {
            "error": str(exc),
        }


# =====================================================
# 7. 漢語拼音及聲調顯示
# =====================================================

def get_pinyin_info(text: str):
    syllables = lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
    )

    tone_names = {
        1: "第一聲",
        2: "第二聲",
        3: "第三聲",
        4: "第四聲",
        5: "輕聲",
    }

    results = []

    for syllable in syllables:
        if not syllable:
            continue

        last = syllable[-1]
        if last in "12345":
            tone_number = int(last)
            results.append({
                "pinyin": syllable,
                "tone": tone_names[tone_number],
            })
        else:
            results.append({
                "pinyin": syllable,
                "tone": "無法判定",
            })

    return results


# =====================================================
# 8. DTW 時間對齊及曲線比較
# =====================================================

def compare_contours(reference, student):
    """
    使用簡單動態時間規整（DTW）對齊兩條一維曲線。
    不依賴 librosa.sequence.dtw，減少版本差異造成的問題。
    """
    ref = np.asarray(reference["contour"], dtype=float)
    stu = np.asarray(student["contour"], dtype=float)

    if ref.size == 0 or stu.size == 0:
        raise ValueError("音高曲線是空的。")

    n = len(ref)
    m = len(stu)

    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = abs(ref[i - 1] - stu[j - 1])
            cost[i, j] = local_cost + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    # 回溯最佳路徑。
    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        candidates = (
            cost[i - 1, j - 1],
            cost[i - 1, j],
            cost[i, j - 1],
        )
        direction = int(np.argmin(candidates))

        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1

    while i > 0:
        path.append((i - 1, 0))
        i -= 1

    while j > 0:
        path.append((0, j - 1))
        j -= 1

    path = np.asarray(path[::-1], dtype=int)

    aligned_ref = ref[path[:, 0]]
    aligned_student = stu[path[:, 1]]
    difference = aligned_ref - aligned_student

    return {
        "aligned_ref": aligned_ref,
        "aligned_student": aligned_student,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "path": path,
    }


# =====================================================
# 9. 側邊設定
# =====================================================

with st.sidebar:
    st.header("練習設定")

    selected_voice = st.selectbox(
        "標準發音",
        list(VOICES.keys()),
    )
    voice = VOICES[selected_voice]

    speed = st.select_slider(
        "播放速度",
        options=[0.75, 0.85, 1.0, 1.25],
        value=1.0,
        format_func=lambda value: f"{value}×",
    )

    st.divider()
    st.caption(
        "建議先聆聽標準發音，再錄製自己的發音。"
    )


# =====================================================
# 10. 輸入練習內容
# =====================================================

st.subheader("1. 輸入練習字詞")

word = st.text_input(
    "請輸入中文字",
    placeholder="例如：媽、馬、老師、你好",
    max_chars=30,
).strip()


if word:
    st.divider()

    # =================================================
    # 11. 顯示拼音
    # =================================================

    st.subheader("2. 拼音及目標聲調")
    pinyin_info = get_pinyin_info(word)

    if pinyin_info:
        columns = st.columns(len(pinyin_info))
        for index, item in enumerate(pinyin_info):
            with columns[index]:
                st.metric(
                    label=item["pinyin"],
                    value=item["tone"],
                )
    else:
        st.info("目前無法取得這段文字的拼音資訊。")

    st.caption(
        "拼音為自動轉換結果；多音字可能需要人工確認，"
        "目前尚未自動處理連續語流中的聲調變化。"
    )

    # =================================================
    # 12. 產生標準語音
    # =================================================

    st.divider()
    st.subheader("3. 聆聽標準發音")

    try:
        with st.spinner("正在生成標準語音..."):
            reference_audio = get_reference_audio(word, voice)

        playback_audio, audio_format = change_playback_speed(
            reference_audio,
            speed,
        )

        st.audio(
            playback_audio,
            format=audio_format,
        )

        st.caption(
            f"聲音：{selected_voice}｜播放速度：{speed}×"
        )

    except Exception as exc:
        st.error(
            "標準語音生成或播放速度調整失敗。"
            f"錯誤訊息：{exc}"
        )
        st.stop()

    # =================================================
    # 13. 學生錄音
    # =================================================

    st.divider()
    st.subheader("4. 錄製自己的發音")

    student_audio = st.audio_input(
        "點擊麥克風開始錄音",
        key="student_recording",
    )

    if student_audio is not None:
        student_bytes = student_audio.getvalue()

        st.audio(
            student_bytes,
            format="audio/wav",
        )

        # =================================================
        # 14. 音高分析
        # =================================================

        st.divider()
        st.subheader("5. 聲調曲線分析")

        with st.spinner("正在分析音高曲線..."):
            ref_result = extract_pitch_contour(
                reference_audio,
                suffix=".mp3",
            )
            stu_result = extract_pitch_contour(
                student_bytes,
                suffix=".wav",
            )

        # 分析函式若回傳 error dict，明確顯示錯誤，不當成正常結果使用。
        ref_error = (
            ref_result.get("error")
            if isinstance(ref_result, dict)
            else None
        )
        stu_error = (
            stu_result.get("error")
            if isinstance(stu_result, dict)
            else None
        )

        if ref_error:
            st.error(f"標準語音分析失敗：{ref_error}")

        elif ref_result is None:
            st.warning(
                "無法取得標準語音的有效音高資料。"
                "請重新輸入字詞或稍後再試。"
            )

        elif stu_error:
            st.error(f"學生錄音分析失敗：{stu_error}")

        elif stu_result is None:
            st.warning(
                "無法取得足夠的有效音高資料。"
                "請在安靜的環境重新錄音，並確保完整念出字詞。"
            )

        else:
            # =================================================
            # 15. 品質檢查
            # =================================================

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "錄音長度",
                f"{stu_result['duration']:.2f} 秒",
            )
            col2.metric(
                "有效發聲比例",
                f"{stu_result['voiced_ratio'] * 100:.1f}%",
            )
            col3.metric(
                "學生音高基準",
                f"{stu_result['median_f0']:.1f} Hz",
            )

            # =================================================
            # 16. 原始音高曲線
            # =================================================

            st.markdown("#### 音高曲線")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                ref_result["time"],
                ref_result["contour"],
                label="Standard",
                linewidth=2.5,
                linestyle="--",
            )
            ax.plot(
                stu_result["time"],
                stu_result["contour"],
                label="Student",
                linewidth=2,
            )

            ax.set_xlabel("Normalized Time (%)")
            ax.set_ylabel("Relative Pitch (Semitones)")
            ax.set_title("Pitch Contour Comparison")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # =================================================
            # 17. DTW 比較
            # =================================================

            st.markdown("#### 時間對齊後的曲線比較")

            try:
                comparison = compare_contours(
                    ref_result,
                    stu_result,
                )

                aligned_ref = comparison["aligned_ref"]
                aligned_student = comparison["aligned_student"]

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                x = np.linspace(0, 100, len(aligned_ref))

                ax2.plot(
                    x,
                    aligned_ref,
                    label="Standard",
                    linewidth=2.5,
                    linestyle="--",
                )
                ax2.plot(
                    x,
                    aligned_student,
                    label="Student",
                    linewidth=2,
                )

                ax2.set_xlabel("Aligned Time (%)")
                ax2.set_ylabel("Relative Pitch (Semitones)")
                ax2.set_title("DTW Aligned Pitch Contour")
                ax2.legend()
                ax2.grid(True, linestyle=":", alpha=0.5)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)

                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

                # =================================================
                # 18. 比較數據
                # =================================================

                st.markdown("#### 音高差異")

                col1, col2 = st.columns(2)
                col1.metric(
                    "平均絕對差異（MAE）",
                    f"{comparison['mae']:.2f} semitones",
                )
                col2.metric(
                    "均方根差異（RMSE）",
                    f"{comparison['rmse']:.2f} semitones",
                )

                st.info(
                    "以上數值代表時間對齊後的相對音高曲線差異，"
                    "不是聲調正確率，也不是發音評分。"
                    "數值越小代表兩條曲線越接近，"
                    "但不能單獨據此判定聲調是否正確。"
                )

                # =================================================
                # 19. 額外資訊
                # =================================================

                with st.expander("查看音高分析資訊"):
                    st.write(
                        "標準語音音高基準：",
                        f"{ref_result['median_f0']:.2f} Hz",
                    )
                    st.write(
                        "學生語音音高基準：",
                        f"{stu_result['median_f0']:.2f} Hz",
                    )
                    st.write(
                        "標準語音有效發聲比例：",
                        f"{ref_result['voiced_ratio'] * 100:.2f}%",
                    )
                    st.write(
                        "學生語音有效發聲比例：",
                        f"{stu_result['voiced_ratio'] * 100:.2f}%",
                    )

            except Exception as exc:
                st.error(f"曲線比較時發生錯誤：{exc}")

else:
    st.info("請先輸入一個中文字或詞語，開始聲調練習。")


# =====================================================
# 20. 頁尾
# =====================================================

st.divider()
st.caption("Mandarin Pronunciation Practice System")
