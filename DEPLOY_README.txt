# 部署說明

1. 將 `app.py`、`requirements.txt`、`packages.txt` 放在 GitHub repository 的同一層（根目錄）。
2. 確認 `app.py` 第一行是 `import asyncio`，不要把終端機輸出、`>>>`、`...` 或 Markdown 程式碼框貼進檔案。
3. Streamlit Community Cloud 的 Main file path 設為 `app.py`。
4. 推送更新後，在 Streamlit Cloud 重新啟動／重新部署應用程式。

## 注意
- `packages.txt` 讓部署環境安裝 ffmpeg，供 MP3 解碼使用。
- 若部署平台可選 Python 版本，建議先使用 Python 3.11 或 3.12，並讓所有檔案位於同一個 repository 根目錄。
- Edge TTS 需要網路連線；語音產生服務若暫時無法連線，網站會顯示錯誤。
- 本程式比較的是相對音高曲線，不會輸出聲調正確率。每個字的音節切分、多音字讀音及第三聲變調仍需另外處理。
