import os
import json
import pandas as pd
from google import genai
from google.genai.errors import ServerError

ITEMS = [
    "正面心情",
    "負面心情",
    "中性心情",
]

input_csv= "all_conversation_log.csv"

def analyze_notes(input_csv: str):
    """
    分析 CSV 文件中的 '筆記' 欄位，返回處理結果。
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("請設定環境變數 GEMINI_API_KEY")
    
    client = genai.Client(api_key=gemini_api_key)
    df = pd.read_csv(input_csv)
    
    if "筆記" not in df.columns:
        raise ValueError("CSV 文件中未找到 '筆記' 欄位")
    
    dialogues = df["筆記"].dropna().tolist()
    dialogues = [str(d).strip() for d in dialogues]
    
    batch_size = 10
    results = []
    for start_idx in range(0, len(dialogues), batch_size):
        batch = dialogues[start_idx:start_idx + batch_size]
        batch_results = process_batch_dialogue(client, batch)
        results.extend(batch_results)
    
    # 將結果合併回原始 DataFrame
    for item in ITEMS:
        df[item] = [res.get(item, "") for res in results]
    
    return df

def process_batch_dialogue(client, dialogues: list, delimiter="-----"):
    """
    將多筆逐字稿合併成一個批次請求，並返回處理結果。
    """
    prompt = (
        "你是一位語境分析專家，請根據以下編碼規則判斷以下日常用語，\n"
        + "\n".join(ITEMS) +
        "\n\n請依據評估結果，對每個項目：若觸及則標記為 1，否則留空。"
        " 根據筆記的語句判斷運動完的心情為正向還是負面，如果不是正面也不是負面，則判斷為中性心情\n"
        " 請對每筆逐字稿產生 JSON 格式回覆，並在各筆結果間用下列分隔線隔開：\n"
        f"{delimiter}\n"
        "例如：\n"
        "```json\n"
        "{\n  \"正面心情\": \"1\",\n  \"負面心情\": \"\"}\n"
        f"{delimiter}\n"
        "{{...}}\n```"
    )
    batch_text = f"\n{delimiter}\n".join(dialogues)
    content = prompt + "\n\n" + batch_text

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content
        )
    except ServerError as e:
        print(f"API 呼叫失敗：{e}")
        return [{item: "" for item in ITEMS} for _ in dialogues]
    
    parts = response.text.split(delimiter)
    results = []
    for part in parts:
        part = part.strip()
        if part:
            results.append(parse_response(part))
    return results

def parse_response(response_text):
    """
    嘗試解析 Gemini API 回傳的 JSON 格式結果。
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    try:
        result = json.loads(cleaned)
        for item in ITEMS:
            if item not in result:
                result[item] = ""
        return result
    except Exception as e:
        print(f"解析 JSON 失敗：{e}")
        print("原始回傳內容：", response_text)
        return {item: "" for item in ITEMS}
