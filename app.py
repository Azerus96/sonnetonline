import gradio as gr
import os
import anthropic
from dotenv import load_dotenv
import io
import pypdf
import json

load_dotenv()

def analyze_review(review, history, file_obj=None):
    # 1. Подготовка истории (как в нашем чат-боте)
    history = history or []
    history.append({"role": "user", "content": review})

    # 2. Обработка файла (если есть) - как в нашем чат-боте
    if file_obj:
        try:
            if file_obj.name.lower().endswith(".pdf"):
                with open(file_obj.name, "rb") as f:
                    pdf_reader = pypdf.PdfReader(f)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                history[-1]["content"] += f"\n\nСодержимое PDF:\n{pdf_text}"
            else:
                gr.Warning(f"Файл {file_obj.name} не является PDF. Поддерживаются только PDF.")
                history[-1]["content"] += f"\n\nФайл {file_obj.name} прикреплен, но не обработан (поддерживаются только PDF)."
        except Exception as e:
            gr.Error(f"Ошибка при обработке файла: {e}")
            history[-1]["content"] += f"\n\nОшибка при обработке файла: {e}"

    # 3. Формируем запрос к Claude (используем system prompt для анализа)
    system_prompt = """
    You are a helpful assistant that specializes in analyzing customer reviews.
    Analyze the provided review and determine:
    1. Sentiment: Is the review positive, negative, or neutral?
    2. Key points: Summarize the key points of the review.
    3. (Optional) Provide a score (1-5 stars) based on the sentiment.

    Format your response as follows:

    Sentiment: [Positive/Negative/Neutral]
    Key points:
    - [Point 1]
    - [Point 2]
    - [Point 3]
    ...
    (Optional) Score: [1-5]
    """

    try:
        client = anthropic.Anthropic()
        response_stream = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,  # Уменьшил max_tokens, т.к. это анализ, а не длинный чат
            temperature=0,
            system=system_prompt,
            messages=history,
            stream=True
        )

        full_response = ""
        messages = []
        for chunk in response_stream:
            full_response += chunk.content[0].text
            messages = []
            for i in range(0, len(history)):
                if history[i]["role"] == "user":
                    bot_message = history[i+1]["content"] if i+1 < len(history) and history[i+1]["role"] == "assistant" else full_response
                    messages.append((history[i]["content"], bot_message))
            yield "", messages

        history.append({"role": "assistant", "content": full_response})
        messages = []
        for i in range(0, len(history) - 1, 2):
            messages.append((history[i]["content"], history[i+1]["content"]))
        yield "", messages

    except Exception as e:
        gr.Error(f"Ошибка API Claude: {e}")
        yield f"Error: {e}", history

def clear_history():
    return None, [], None

with gr.Blocks(title="Review Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Analyze Customer Reviews with Claude 3.5 Sonnet")
    chatbot = gr.Chatbot(label="Analysis Result", value=[], height=400) # Уменьшил высоту
    with gr.Row():
        review_input = gr.Textbox(
            label="Customer Review",
            placeholder="Enter a customer review here, or upload a PDF...",
            lines=5,
            scale=4,
        )
        file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1)

    with gr.Row():
        analyze_button = gr.Button("Analyze")
        clear = gr.ClearButton([review_input, chatbot, file_upload])
        clear_hist_button = gr.Button("Clear History")

    review_input.submit(analyze_review, [review_input, chatbot, file_upload], [review_input, chatbot])
    file_upload.upload(analyze_review, [review_input, chatbot, file_upload], [review_input, chatbot])
    clear_hist_button.click(clear_history, [], [review_input, chatbot, file_upload])
    analyze_button.click(analyze_review, [review_input, chatbot, file_upload], [review_input, chatbot]) # Кнопка Analyze
    demo.load(None, [], [chatbot])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
