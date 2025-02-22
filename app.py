import gradio as gr
import os
import anthropic
from dotenv import load_dotenv
import io
import pypdf
import json

load_dotenv()

def chat(message, history, file_obj=None):
    print("=== НОВОЕ СООБЩЕНИЕ ===")
    print(f"Входное сообщение: {message}")

    history = history or []
    history.append({"role": "user", "content": message})
    print(f"История (после добавления сообщения пользователя):\n{json.dumps(history, indent=2)}")

    if file_obj:
        print(f"Загружен файл: {file_obj.name}")  # Лог: имя файла
        try:
            # Чтение PDF, если файл - PDF
            if file_obj.name.lower().endswith(".pdf"):
                with open(file_obj.name, "rb") as f: # Открываем как файл
                    pdf_reader = pypdf.PdfReader(f)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                history[-1]["content"] += f"\n\nСодержимое PDF:\n{pdf_text}"
                print("PDF успешно прочитан.")
            else:
                gr.Warning(f"Файл {file_obj.name} не является PDF. Поддерживаются только PDF.")
                history[-1]["content"] += f"\n\nФайл {file_obj.name} прикреплен, но не обработан (поддерживаются только PDF)."
        except Exception as e:
            gr.Error(f"Ошибка при обработке файла: {e}")
            print(f"Ошибка при обработке файла: {e}")
            history[-1]["content"] += f"\n\nОшибка при обработке файла: {e}"

    try:
        client = anthropic.Anthropic()
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
            print(f"Получен chunk от API: {chunk}")
            full_response += chunk.content[0].text

            messages = []
            for i in range(0, len(history)):
                if history[i]["role"] == "user":
                    bot_message = history[i+1]["content"] if i+1 < len(history) and history[i+1]["role"] == "assistant" else full_response
                    messages.append((history[i]["content"], bot_message))
            print(f"messages перед yield (внутри цикла): {messages}")
            yield "", messages

        history.append({"role": "assistant", "content": full_response})
        print(f"Полный ответ от API: {full_response}")

        messages = []
        for i in range(0, len(history)):
            if history[i]["role"] == "user":
                bot_message = history[i + 1]["content"] if i + 1 < len(history) and history[i + 1]["role"] == "assistant" else ""
                messages.append((history[i]["content"], bot_message))
        print(f"messages перед yield (после цикла): {messages}")
        yield "", messages

    except Exception as e:
        gr.Error(f"Ошибка API Claude: {e}")
        print(f"Ошибка API Claude: {e}")
        yield f"Error: {e}", history

def clear_history():
    print("=== ОЧИСТКА ИСТОРИИ ===")
    # Должны вернуть столько же значений сколько и принимают
    return None, [], None

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

    msg.submit(chat, [msg, chatbot, file_upload], [msg, chatbot])
    file_upload.upload(chat, [msg, chatbot, file_upload], [msg, chatbot])
    clear_hist_button.click(clear_history, [], [msg, chatbot, file_upload])
    send_button.click(chat, [msg, chatbot, file_upload], [msg, chatbot])
    demo.load(None, [], [chatbot])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
