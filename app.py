import gradio as gr
import os
import anthropic
from dotenv import load_dotenv
import io
import pypdf

load_dotenv()

def chat(message, history, file_obj=None):
    history = history or []
    history.append({"role": "user", "content": message})

    if file_obj:
        try:
            if file_obj.name.lower().endswith(".pdf"):
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_obj.data))
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                history[-1]["content"] += f"\n\nСодержимое PDF:\n{pdf_text}"
            else:
                history[-1]["content"] += f"\n\nФайл {file_obj.name} прикреплен, но не обработан (поддерживаются только PDF)."
        except Exception as e:
            history[-1]["content"] += f"\n\nОшибка при обработке файла: {e}"

    try:
        client = anthropic.Anthropic()
        response_stream = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            temperature=0,
            system="You are a helpful assistant. You can analyze PDF documents...",
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

    with gr.Row(): # Добавляем еще одну строку для кнопок
        send_button = gr.Button("Send") # Добавляем кнопку Send
        clear = gr.ClearButton([msg, chatbot, file_upload])
        clear_hist_button = gr.Button("Clear History")

    # Обработчики событий
    msg.submit(chat, [msg, chatbot, file_upload], [msg, chatbot])
    file_upload.upload(chat, [msg, chatbot, file_upload], [msg, chatbot])
    clear_hist_button.click(clear_history, [], [msg, chatbot, file_upload])
    send_button.click(chat, [msg, chatbot, file_upload], [msg, chatbot]) # Добавляем обработчик для кнопки Send


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
