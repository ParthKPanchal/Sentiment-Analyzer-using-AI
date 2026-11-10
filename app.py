import spaces
import gradio as gr
from transformers import pipeline

# Load the AI Model
model = pipeline("sentiment-analysis", device=0)

# This function runs when the user clicks Submit
@spaces.GPU
def analyze_sentiment(text):
    if not text.strip():
        return "Please enter a sentence!"
    
    result = model(text)[0]
    label = result['label']
    score = round(result['score']*100,2)

    if label == 'POSITIVE':
        return f'POSTIVE - The AI is {score}% confident!'
    else:
        return f'NEGATIVE - The AI is {score}% confident!'

# Build the APP User Interface
app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder='Type a sentence here...'),
    outputs=gr.Textbox(label='Result'),
    title="Sentiment Analyzer",
    description= "Type any sentence and I will tell you if it is POSITIVE or NEGATIVE!"
)

#launch the APP
app.launch()