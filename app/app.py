import gradio as gr
from predict import predict

def run(img):
    if img is None:
        return "Upload a photo first.", None
    return predict(img)

demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Fruit photo"),
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Label(num_top_classes=3, label="Top guesses"),
    ],
    title="Peel-Appeal",
    description=(
        "Upload a close-up photo of a single fruit. Works best when the fruit fills most "
        "of the frame. Trained on 8 fruits (apple, banana, grape, guava, jujube, orange, "
        "pomegranate, strawberry); accuracy drops on distant or cluttered photos."
    ),
)

if __name__ == "__main__":
    demo.launch()