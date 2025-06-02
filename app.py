from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from dataAgent import process_data
from DRai import analyze_notes
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import asyncio
import csv
import os
import re
import matplotlib    
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
matplotlib.use('Agg') 
from fpdf import FPDF
app = Flask(__name__)
exercise_log = []



@app.route('/')
def home():
    return render_template('home.html', exercise_log=exercise_log)



@app.route('/add', methods=['GET', 'POST'])
def add_record():
    if request.method == 'POST':
        date = request.form['date']
        exercise_type = request.form['type']
        duration = request.form['duration']
        note = request.form['note']
        
        # 新增到模擬資料庫
        new_record = {
            '日期': date,
            '運動類型': exercise_type,
            '持續時間(分)': duration,
            '筆記': note
        }
        exercise_log.append(new_record)
        
        # 寫入 CSV 文件
        with open('all_exercise_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['日期', '運動類型', '持續時間(分)', '筆記'])
            # 如果文件是空的，寫入標題
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(new_record)
        
        return redirect(url_for('view_records'))
    return render_template('add.html')



@app.route('/records')
def view_records():
    # 讀取 CSV 文件內容
    logs = []
    with open('all_exercise_log.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            logs.append(row)
    
    # 將資料順序反轉
    logs.reverse()
    
    # 將讀取的資料傳遞給模板
    return render_template('records.html', logs=logs)



@app.route('/process_data', methods=['POST'])
def process_data_route():
    try:
        # 執行資料處理
        result = asyncio.run(process_data())
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]}), 500
        
        # 確認 CSV 文件是否存在
        csv_file_path = result.get("output_file", "all_conversation_log.csv")
        if not os.path.exists(csv_file_path):
            return jsonify({"success": False, "error": "找不到對話紀錄文件"}), 500
        
        # 成功處理後返回成功訊息
        return jsonify({"success": True, "message": "資料處理完成。", "output_file": csv_file_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



def format_content(text):
    # 確保 text 是字符串類型，否則返回 None
    if not isinstance(text, str):
        return None

    # 跳過運動資料類的內容
    if text.startswith("目前正在處理第") or "以下為該批次資料" in text:
        return None

    # 將 **...** 改為 <strong>...<strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # 將段落開頭的數字 (e.g., 1.、2.) 切成段落
    text = re.sub(r'(\n|^)(\d+\.\s*)', r'\1<br><strong>\2</strong>', text)

    # 將條列項 (如 * xxx 或 - xxx) 換成 <li>
    text = re.sub(r'\n[\*\-]\s*(.*?)\n', r'<li>\1</li>', text)

    # 包成 ul
    text = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)

    # 換行轉 <br>
    text = text.replace('\n', '<br>')

    return text



@app.route('/table')
def index():
    df = pd.read_csv('all_conversation_log.csv')

    # 確保 content 列中沒有空值
    df['content'] = df['content'].fillna("")

    # 格式化 content 欄並過濾掉 None
    df['formatted_content'] = df['content'].apply(format_content)
    filtered_df = df[df['formatted_content'].notnull()]

    return render_template('table.html', data=filtered_df['formatted_content'].tolist())



@app.route("/process_notes", methods=["POST"])
def process_notes():
    try:
        input_csv = "all_exercise_log.csv"
        result_df = analyze_notes(input_csv)
        result_csv = "all_processed_notes.csv"
        result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
        return jsonify({"success": True, "message": "分析完成", "result_csv": result_csv})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})



@app.route("/result")
def result():
    result_csv = "all_processed_notes.csv"
    df = pd.read_csv(result_csv)
    df.fillna("", inplace=True)  # 確保 NaN 值被替換為空白

     # 統計每個欄位中值為 1 的總數
    stats = {
        "正面心情": (df["正面心情"] == 1).sum() if "正面心情" in df.columns else 0,
        "負面心情": (df["負面心情"] == 1).sum() if "負面心情" in df.columns else 0,
        "中性心情": (df["中性心情"] == 1).sum() if "中性心情" in df.columns else 0,
    }


    return render_template("result.html", data=df.to_dict(orient="records"), stats=stats)
    


@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        # 讀取 CSV 並格式化內容
        df = pd.read_csv('all_conversation_log.csv')
        df['formatted_content'] = df['content'].apply(format_content)
        filtered_df = df[df['formatted_content'].notnull()]

        # 初始化 PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # 添加支持 Unicode 的字體
        font_path = os.path.join("fonts", "NotoSansTC-Regular.ttf")
        pdf.add_font("NotoSans", "", font_path, uni=True)
        pdf.set_font("NotoSans", size=12)

        # 添加標題
        pdf.set_font("NotoSans", size=24)
        pdf.cell(200, 10, txt="運動分析報告", ln=True, align="C")
        pdf.ln(10)

        # 插入圖片
        pdf_width = 210  # A4 寬度（mm）
        image_width = 160  # 圖片寬度（mm）

        image_path = "static/activity_counts_chart.png"
        if os.path.exists(image_path):
            x_centered = (pdf_width - image_width) / 2  # 計算置中的 x 坐標
            pdf.image(image_path, x=x_centered, y=20, w=image_width)  # x, y 為圖片位置，w 為圖片寬度
            pdf.ln(95)  # 添加空白行以避免文字覆蓋圖片

        image_path = "static/activity_duration_chart.png"
        if os.path.exists(image_path):
            x_centered = (pdf_width - image_width) / 2  # 計算置中的 x 坐標
            pdf.image(image_path, x=x_centered, y=120, w=image_width)  # x, y 為圖片位置，w 為圖片寬度
            pdf.ln(95)  # 添加空白行以避免文字覆蓋圖片
    
        pdf.set_font("NotoSans", size=16)
        pdf.cell(200, 10, txt="總結", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("NotoSans", size=12)
        for index, row in filtered_df.iterrows():
            # 去除 HTML 標籤並分段
            content = re.sub(r'<.*?>', '', row['formatted_content'])  # 移除 HTML 標籤
            content = re.sub(r'\*', '', content)  # 移除所有 *
            content = re.sub(r'\s+', ' ', content).strip()  # 移除多餘空格並修剪首尾空格
            content = re.sub(r'\b(TERMINATE|exit|：)\b', '', content)  # 移除 "TERMINATE" 和 "exit"
            # 只保留 "總結" 之後的文本
            if "總結" in content:
                content = content.split("總結", 1)[1]  # 分割文本並保留 "總結" 之後的部分
            
            # 處理隱藏的換行符號（例如 \u2028 或 \u2029）
            content = re.sub(r'[\u2028\u2029]', ' ', content)  # 替換隱藏的換行符號為空格
            
            # 合併所有換行符號成一串文字
            content = re.sub(r'\n+', ' ', content)  # 將多個換行符號替換為單個空格
            
            # 添加文字到 PDF
            pdf.multi_cell(0, 10, txt=content, align="L")  # 左對齊段落


        output_pdf = os.path.join("Data", "exercise_analysis_report.pdf")
        pdf.output(output_pdf)

        return jsonify({"success": True, "message": "PDF 已生成", "pdf_path": f"/{output_pdf}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@app.route('/Data/exercise_analysis_report.pdf')
def download_pdf():
    pdf_file_path = os.path.join("Data", "exercise_analysis_report.pdf")  # 文件保存到 PDF 資料夾
    if os.path.exists(pdf_file_path):
        return Response(
            open(pdf_file_path, "rb"),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename=exercise_analysis_report.pdf"}
        )
    else:
        return "找不到檔案", 404



@app.route('/charts')
def charts():
    return render_template('charts.html')



@app.route('/generate_charts', methods=['GET'])
def generate_charts():
    try:
        matplotlib.rc('font', family='Microsoft JhengHei',size=16)

        csv_path = "all_processed_notes.csv"
        df = pd.read_csv(csv_path)

        
        activity_counts = df['運動類型'].value_counts()
        plt.figure(figsize=(10, 6))
        activity_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('每個運動類別的次數')
        plt.xlabel('運動類型')
        plt.ylabel('次數')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/activity_counts_chart.png')
        plt.close()

        
        activity_duration = df.groupby('運動類型')['持續時間(分)'].sum()
        plt.figure(figsize=(10, 6))
        activity_duration.plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title('每個運動類別的持續時間(分)')
        plt.xlabel('運動類型')
        plt.ylabel('總持續時間(分)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/activity_duration_chart.png')
        plt.close()

        
        positive_emotions = df.groupby('運動類型')['正面心情'].sum()
        plt.figure(figsize=(10, 6))
        positive_emotions.plot(kind='bar', color='lightblue', edgecolor='black')
        plt.title('每個運動類別的正面心情統計')
        plt.xlabel('運動類型')
        plt.ylabel('正面心情次數')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/positive_emotions_chart.png')
        plt.close()

        
        negative_emotions = df.groupby('運動類型')['負面心情'].sum()
        plt.figure(figsize=(10, 6))
        negative_emotions.plot(kind='bar', color='salmon', edgecolor='black')
        plt.title('每個運動類別的負面心情統計')
        plt.xlabel('運動類型')
        plt.ylabel('負面心情次數')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/negative_emotions_chart.png')
        plt.close()

        charts = [
            "static/activity_counts_chart.png",
            "static/activity_duration_chart.png",
            "static/positive_emotions_chart.png",
            "static/negative_emotions_chart.png"
        ]
        return render_template('charts.html', log="圖表已生成！", charts=charts)
    except Exception as e:
        return render_template('charts.html', log=f"生成圖表時出現錯誤: {str(e)}", charts=[])



if __name__ == '__main__':
    app.run(debug=True)