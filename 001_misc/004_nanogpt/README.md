## 2026/04/23
> nanoGPT 
* 好，我來寫訓練迴圈，搭配 TinyShakespeare。
* 包含：
資料 — 自動下載 TinyShakespeare，字元級 tokenizer
訓練迴圈 — gradient clipping、定期評估
學習率 — warmup + cosine decay
Checkpoint — 自動儲存最佳 val loss
在 Colab T4 上大約跑 10～15 分鐘就能看到像樣的莎士比亞風格輸出。
* 把這個也推上 GitHub，然後開 Colab 跑看看。
* train.py 裡有 from nanogpt import GPT, GPTConfig，所以兩個檔案要在同一層。

2026/04/23