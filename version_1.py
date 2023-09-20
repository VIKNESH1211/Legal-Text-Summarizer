from fastapi import FastAPI
import pandas as pd
import re
import torch
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI(title="ML Models as API on Google Colab", description="with FastAPI and ColabCode", version="1.0")

# # Initialize logging
# my_logger = logging.getLogger()
# my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='logs.log')

# Load the legal dictionary from an Excel file
legal_dict = pd.read_excel('DIC.xlsx')

# Convert the legal dictionary into a dictionary for easy lookup
legal_to_civilian = dict(zip(legal_dict['WORD'], legal_dict['MEANING']))

# Function to convert legal text to civilian language
def legal_to_civilian_language(legal_text):
    # Replace legal terms with civilian language equivalents
    for term, civilian_term in legal_to_civilian.items():
        legal_text = re.sub(r'\b{}\b'.format(re.escape(term)), civilian_term, legal_text)

    return legal_text


hf_name = 'pszemraj/led-large-book-summary'

summarizer = pipeline(
    "summarization",
    hf_name,
    device=0 if torch.cuda.is_available() else -1,
)

class SummarizationRequest(BaseModel):
    text: str

@app.post("/summarize/")
async def summarize(request: SummarizationRequest):

    input_text = request.text

    civilian_text = legal_to_civilian_language(input_text)

    result = summarizer(
    civilian_text,
    min_length=16,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    early_stopping=True,
)
    return {"summary": result}

