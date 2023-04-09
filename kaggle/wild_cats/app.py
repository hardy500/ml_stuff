import gradio as gr
from inference import predict

iface = gr.Interface(
        fn=predict, 
        inputs='image',
        outputs='label'
)
iface.launch()
