import gradio as gr
import os
import anthropic
from dotenv import load_dotenv
import io
import pypdf
import json  # Для красивого вывода JSON

load_dotenv()

def chat(message, history, file_obj=None):
    print("=== НОВОЕ СООБЩЕНИЕ ===")  # Лог: начало нового сообщения
    print(f"Входное сообщение: {message}")

    history = history or []
    history.append({"role": "user", "content": message})
    print(f"История (после добавления сообщения пользователя):\n{json.dumps(history, indent=2)}")

    if file_obj:
        print(f"Загружен файл: {file_obj.name}, тип: {file_obj.type}, размер: {len(file_obj.data)} байт")
        try:
            if file_obj.name.lower().endswith(".pdf"):
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_obj.data))
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                history[-1]["content"] += f"\n\nСодержимое PDF:\n{pdf_text}"
                print("PDF успешно прочитан.")
            else:
                gr.Warning(f"Файл {file_obj.name} не является PDF.  Поддерживаются только PDF.") # Предупреждение в UI
                history[-1]["content"] += f"\n\nФайл {file_obj.name} прикреплен, но не обработан (поддерживаются только PDF)."
        except Exception as e:
            gr.Error(f"Ошибка при обработке файла: {e}")  # Ошибка в UI
            print(f"Ошибка при обработке файла: {e}")  # Лог ошибки
            history[-1]["content"] += f"\n\nОшибка при обработке файла: {e}"

    try:
        client = anthropic.Anthropic()

        # Формируем запрос к API (для логов)
        request_payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 8192,
            "temperature": 0,
            "system": "You are a helpful assistant. You can analyze PDF documents...",
            "messages": history,
            "stream": True
        }
        print(f"Запрос к API Claude:\n{json.dumps(request_payload, indent=2)}")

        response_stream = client.messages.create(**request_payload)

        full_response = ""
        messages = []
        for chunk in response_stream:
            print(f"Получен chunk от API: {chunk}")  # Лог: каждый chunk
            full_response += chunk.content[0].text

            messages = []
            for i in range(0, len(history)):
                if history[i]["role"] == "user":
                    bot_message = history[i+1]["content"] if i+1 < len(history) and history[i+1]["role"] == "assistant" else full_response
                    messages.append((history[i]["content"], bot_message))
            print(f"messages перед yield (внутри цикла): {messages}") # Лог: messages
            yield "", messages

        history.append({"role": "assistant", "content": full_response})
        print(f"Полный ответ от API: {full_response}") # Лог: полный ответ

        # Обновляем messages ПОСЛЕ добавления полного ответа:
        messages = []
        for i in range(0, len(history)):
            if history[i]["role"] == "user":
                bot_message = history[i + 1]["content"] if i + 1 < len(history) and history[i + 1]["role"] == "assistant" else ""
                messages.append((history[i]["content"], bot_message))
        print(f"messages перед yield (после цикла): {messages}") # Лог: messages
        yield "", messages

    except Exception as e:
        gr.Error(f"Ошибка API Claude: {e}")  # Ошибка в UI
        print(f"Ошибка API Claude: {e}")  # Лог ошибки
        yield f"Error: {e}", history

def clear_history():
    print("=== ОЧИСТКА ИСТОРИИ ===")
    return None, []

with gr.Blocks(title="Claude Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chat with Claude 3.5 Sonnet (with PDF support)")
    chatbot = gr.Chatbot(label="Claude 3.5 Sonnet", value=[], height=550)
    with gr.Row():
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here, or upload a PDF...",
            autofocus=True,
            lines=2,
            scale=4,
        )
        file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1)

    with gr.Row():
        send_button = gr.Button("Send")
        clear = gr.ClearButton([msg, chatbot, file_upload])
        clear_hist_button = gr.Button("Clear History")

    msg.submit(chat, [msg, chatbot, file_upload], [msg, chatbot], queue=False) # queue=False убрал
    file_upload.upload(chat, [msg, chatbot, file_upload], [msg, chatbot], queue=False) # queue=False убрал
    clear_hist_button.click(clear_history, [], [msg, chatbot, file_upload], queue=False) # queue=False убрал
    send_button.click(chat, [msg, chatbot, file_upload], [msg, chatbot], queue=False) # queue=False убрал
    demo.load(None, [], [chatbot], queue=False) # Добавил для очистки при запуске

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
