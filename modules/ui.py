import os

import gradio as gr

from modules import options
from modules.context import Context
from modules.model import infer

from model_vits import vits

# import librosa


css = "style.css"
script_path = "scripts"
_gradio_template_response_orig = gr.routes.templates.TemplateResponse

vits = vits.Vits()
def tts(text):
    if text:
        blocks = text.split("<br/>")
        text = ''.join(blocks)
        result = vits.generateSound(text)
        # audio path
        vocie_path = os.path.abspath(result[-1])
        print('voice_path:'+vocie_path)
        
        return '<audio controls="" autoplay="autoplay" preload="metadata" src="http://127.0.0.1:17860/file={}"></audio>'.format(vocie_path)

def predict(ctx, query, max_length, top_p, temperature, use_stream_chat,bspk_assist):
    ctx.limit_round()
    flag = True
    for _, output in infer(
            query=query,
            history=ctx.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            use_stream_chat=use_stream_chat
    ):
        if flag:
            ctx.append(query, output)
            flag = False
        else:
            ctx.update_last(query, output)
        yield ctx.rh, ""
        
    ctx.refresh_last()
    if bspk_assist:
        query,output = ctx.rh[-1]
        output = '<br/>'.join([output,tts(output)])
        ctx.rh[-1] = (query,output)
    yield ctx.rh, ""

def clear_history(ctx):
    ctx.clear()
    return gr.update(value=[])


def apply_max_round_click(ctx, max_round):
    ctx.max_rounds = max_round
    return f"Applied: max round {ctx.max_rounds}"

def set_spk(bspk_assist,spk_name):
    assert bspk_assist and spk_name, 'selected spks: {}, bspk assist: {}, must set'.format(spk_name,bspk_assist)
    # 使用语音助手，relaod model
    vits.set_spk(spk_name)
    return

def show_spks(bselect):
    if bselect:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def create_ui():
    reload_javascript()

    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
        _ctx = Context()
        state = gr.State(_ctx)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""<h2><center>ChatGLM WebUI</center></h2>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            max_length = gr.Slider(minimum=4, maximum=4096, step=4, label='Max Length', value=2048)
                            top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.7)
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature', value=0.95)

                        with gr.Row():
                            max_rounds = gr.Slider(minimum=1, maximum=100, step=1, label="最大对话轮数", value=20)
                            apply_max_rounds = gr.Button("✔", elem_id="del-btn")
                                            
                        with gr.Row():
                            # todo call from sovits module
                            bspk_assist = gr.Checkbox(value=False,label='是否启用语音助手',info='use vits convert text to speech',elem_id='spk_assister')  
                        with gr.Row():
                            spk_name = gr.Dropdown(choices=vits.get_spks(),value='',label="请选择你的播报员", visible=False)
                                
                        bspk_assist.change(fn=show_spks,
                                        inputs=bspk_assist,
                                        outputs=spk_name
                                        )
                        spk_name.change(fn=set_spk,
                                        inputs=[bspk_assist,spk_name],
                                        outputs=[]
                                        )
                        cmd_output = gr.Textbox(label="Command Output")
                        with gr.Row():
                            use_stream_chat = gr.Checkbox(label='使用流式输出', value=True)
                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear_history_btn = gr.Button("清空对话")

                        with gr.Row():
                            sync_his_btn = gr.Button("同步对话")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

                        with gr.Row():
                            save_md_btn = gr.Button("保存为 MarkDown")

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=800)
                with gr.Row():
                    input_message = gr.Textbox(placeholder="输入你的内容...(按 Ctrl+Enter 发送)", show_label=False, lines=4, elem_id="chat-input").style(container=False)
                    clear_input = gr.Button("🗑️", elem_id="del-btn")

                with gr.Row():
                    submit = gr.Button("发送", elem_id="c_generate")

                with gr.Row():
                    revoke_btn = gr.Button("撤回")

        submit.click(predict, inputs=[
            state,
            input_message,
            max_length,
            top_p,
            temperature,
            use_stream_chat,
            bspk_assist
        ], outputs=[
            chatbot,
            input_message
        ])
        
        revoke_btn.click(lambda ctx: ctx.revoke(), inputs=[state], outputs=[chatbot])
        clear_history_btn.click(clear_history, inputs=[state], outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[input_message], outputs=[input_message])
        save_his_btn.click(lambda ctx: ctx.save_history(), inputs=[state], outputs=[cmd_output])
        save_md_btn.click(lambda ctx: ctx.save_as_md(), inputs=[state], outputs=[cmd_output])
        load_his_btn.upload(lambda ctx, f: ctx.load_history(f), inputs=[state, load_his_btn], outputs=[chatbot])
        sync_his_btn.click(lambda ctx: ctx.rh, inputs=[state], outputs=[chatbot])
        apply_max_rounds.click(apply_max_round_click, inputs=[state, max_rounds], outputs=[cmd_output])

    with gr.Blocks(css=css, analytics_enabled=False) as settings_interface:
        with gr.Row():
            reload_ui = gr.Button("Reload UI")

        def restart_ui():
            options.need_restart = True

        reload_ui.click(restart_ui)

    interfaces = [
        (chat_interface, "Chat", "chat"),
        (settings_interface, "Settings", "settings")
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="ChatGLM") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

    return demo


def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    javascript = ""
    # with open("script.js", "r", encoding="utf8") as js_file:
    #     javascript = f'<script>{js_file.read()}</script>'

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as js_file:
            javascript += f"\n<script>{js_file.read()}</script>"

    # todo: theme
    # if cmd_opts.theme is not None:
    #     javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    def template_response(*args, **kwargs):
        res = _gradio_template_response_orig(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
