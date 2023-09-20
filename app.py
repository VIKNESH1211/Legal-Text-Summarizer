from fastapi import FastAPI
import pandas as pd
import re
import torch
from pydantic import BaseModel
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ML Models as API on Google Colab", description="with FastAPI and ColabCode", version="1.0")

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins here instead of "*"
    allow_methods=["*"],  # You can specify specific HTTP methods (e.g., ["GET", "POST"])
    allow_headers=["*"],  # You can specify specific headers if needed
    allow_credentials=True,  # You can set this to True if you need to include credentials in the request
    expose_headers=["Content-Disposition"],  # You can specify headers to expose to the client
)

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

#model for SUMMARY
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")




#model for Translation
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model_1 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")



def get_response(input_text):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=300, return_tensors="pt").to(torch_device)
  gen_out = model_1.generate(**batch,max_length=60,num_beams=5, num_return_sequences=1 )
  output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
  return output_text


# Summarization request model
class SummarizationRequest(BaseModel):
    text: str

# Text generation request model
class TextGenerationRequest(BaseModel):
    text: str

@app.post("/summarize/")
async def summarize(request: SummarizationRequest):
    input_text = request.text

    result_d = summarizer(input_text, max_length=130, min_length=30, do_sample=False)

    # Check if data is a list with at least one element
    if isinstance(result_d, list) and len(result_d) > 0:
        # Access the first element (in this case, there's only one element)
        first_element = result_d[0]

        # Check if the 'summary_text' key exists in the dictionary and if it's a string
        if 'summary_text' in first_element and isinstance(first_element['summary_text'], str):
            summary_text = first_element['summary_text']
            result = summary_text

        else:
            print("Value for 'summary_text' not found or not a string")
    else:
        print("Data is empty or not a list")

    return {"summary": result}
    

@app.post("/translate")
async def translate(request: TextGenerationRequest):

    input_text = request.text

    model_inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # translate from English to Hindi
    generated_tokens = model_1.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return {"translation": translation}
