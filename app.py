import gradio as gr
import os
import anthropic
from dotenv import load_dotenv
import io  # Для работы с байтовыми потоками
import pypdf  # Для чтения PDF

load_dotenv()

# Функция для обработки сообщений и файлов
def chat(message, history, file_obj=None):
    history = history or []
    history.append({"role": "user", "content": message})

    if file_obj:
        # Обработка файла, если он есть
        try:
            # Чтение PDF файла
            if file_obj.name.lower().endswith(".pdf"):
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_obj.data))
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

                # Добавляем содержимое PDF к сообщению пользователя
                history[-1]["content"] += f"\n\nСодержимое PDF:\n{pdf_text}"
            else:
                # Для других типов файлов можно добавить свою логику
                history[-1]["content"] += f"\n\nФайл {file_obj.name} прикреплен, но не обработан (поддерживаются только PDF)."

        except Exception as e:
            history[-1]["content"] += f"\n\nОшибка при обработке файла: {e}"

    try:
        client = anthropic.Anthropic()
        response_stream = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0,  # Устанавливаем температуру в 0
            system="You are a helpful assistant. You can analyze PDF documents and answer questions about them. You can also write code and format it appropriately using Markdown. If you write code, put it in a code block with the language specified.",
            messages=history,
            stream=True
        )

        full_response = ""
        for chunk in response_stream:
            full_response += chunk.content[0].text
            messages = [(history[i]["content"], history[i+1]["content"] if i+1 < len(history) else "") for i in range(0, len(history), 2)]
            if len(messages) > 0:
                messages[-1] = (messages[-1][0], full_response)
            else:
                messages = [(message, full_response)]
            yield "", messages

        history.append({"role": "assistant", "content": full_response})

    except Exception as e:
        yield f"Error: {e}", history

def clear_history():
    return None, []

# Создаем Gradio-интерфейс
with gr.Blocks(title="Claude Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chat with Claude 3.5 Sonnet (with PDF support)")
    chatbot = gr.Chatbot(label="Claude 3.5 Sonnet", value=[], height=550)
    with gr.Row():  # Размещаем элементы в строку
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here, or upload a PDF...",
            autofocus=True,
            lines=2,
            scale=4,  # Занимает 4/5 ширины
        )
        file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1)  # Занимает 1/5

    clear = gr.ClearButton([msg, chatbot, file_upload])
    clear_hist_button = gr.Button("Clear History")

    # Обработчики событий
    msg.submit(chat, [msg, chatbot, file_upload], [msg, chatbot])
    file_upload.upload(chat, [msg, chatbot, file_upload], [msg, chatbot])  # Добавили обработчик
    clear_hist_button.click(clear_history, [], [msg, chatbot, file_upload])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
