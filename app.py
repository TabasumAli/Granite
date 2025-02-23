# import streamlit as st
# import os
# import re
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from PyPDF2 import PdfReader
# from peft import get_peft_model, LoraConfig, TaskType

# # ✅ Fix CUDA Memory Fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # 🔹 Load IBM Granite Model with 4-bit Quantization
# MODEL_NAME = "ibm-granite/granite-3.1-2b-instruct"
# quant_config = BitsAndBytesConfig(load_in_4bit=True)  # Use 4-bit quantization

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Ensure model initialization correctly
# torch.cuda.empty_cache()  # Clear GPU memory before loading model

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=quant_config,
#     device_map="auto",  # Auto-assign layers to available GPUs/CPUs
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use FP16 if GPU is available
# ).to(device)  # Move model to correct device

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # 🔹 Apply LoRA Fine-Tuning Configuration
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )
# model = get_peft_model(model, lora_config)
# model.eval()

# # 🛠 Function to Read & Extract Text from PDFs
# def read_files(file):
#     file_context = ""
#     reader = PdfReader(file)
    
#     for page in reader.pages:
#         text = page.extract_text()
#         if text:
#             file_context += text + "\n"
    
#     return file_context.strip()

# # 🛠 Function to Format AI Prompts
# # 🛠 Function to Format AI Prompts
# def format_prompt(system_msg, user_msg, file_context=""):
#     if file_context:
#         system_msg += f" The user has provided a contract document. Use its context to generate insights, but do not repeat or summarize the document itself."
#     return [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg}
#     ]
# # 🛠 Function to Generate AI Responses
# def generate_response(input_text, max_tokens=1000, top_p=0.9, temperature=0.7):
#     torch.cuda.empty_cache()  # ✅ Clear GPU memory before inference
    
#     model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         output = model.generate(
#             **model_inputs,
#             max_new_tokens=max_tokens,
#             do_sample=True,
#             top_p=top_p,
#             temperature=temperature,
#             num_return_sequences=1,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # 🛠 Function to Clean AI Output
# def post_process(text):
#     cleaned = re.sub(r'戥+', '', text)  # Remove unwanted symbols
#     lines = cleaned.splitlines()
#     unique_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))
#     return "\n".join(unique_lines)

# # 🛠 Function to Handle RAG with IBM Granite & Streamlit
# def granite_simple(prompt, file):
#     file_context = read_files(file) if file else ""
    
#     system_message = "You are IBM Granite, a legal AI assistant specializing in contract analysis."
    
#     messages = format_prompt(system_message, prompt, file_context)
#     input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     response = generate_response(input_text)
#     return post_process(response)

# # 🔹 Streamlit UI
# def main():
#     st.set_page_config(page_title="Contract Analysis AI", page_icon="📜", layout="wide")

#     st.title("📜 AI-Powered Contract Analysis Tool")
#     st.write("Upload a contract document (PDF) for a detailed AI-driven legal and technical analysis.")

#     # 🔹 Sidebar Settings
#     with st.sidebar:
#         st.header("⚙️ Settings")
#         max_tokens = st.slider("Max Tokens", 50, 1000, 250, 50)
#         top_p = st.slider("Top P (sampling)", 0.1, 1.0, 0.9, 0.1)
#         temperature = st.slider("Temperature (creativity)", 0.1, 1.0, 0.7, 0.1)

#     # 🔹 File Upload Section
#     uploaded_file = st.file_uploader("📂 Upload a contract document (PDF)", type="pdf")

#     if uploaded_file is not None:
#         temp_file_path = "temp_uploaded_contract.pdf"
#         with open(temp_file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         st.success("✅ File uploaded successfully!")

#         # 🔹 User Input for Analysis
#         user_prompt = "Perform a detailed technical analysis of the attached contract document, highlighting potential risks, legal pitfalls, compliance issues, and areas where contractual terms may lead to future disputes or operational challenges."

#         # user_prompt = st.text_area(
#         #     "📝 Describe what you want to analyze:",
#         #     "Perform a detailed technical analysis of the attached contract document, highlighting potential risks, legal pitfalls, compliance issues, and areas where contractual terms may lead to future disputes or operational challenges."
#         # )
#         # with st.empty():  # This hides the text area
#         #   user_prompt = st.text_area(
#         #       "📝 Describe what you want to analyze:",
#         #       "Perform a detailed technical analysis of the attached contract document, highlighting potential risks, legal pitfalls, compliance issues, and areas where contractual terms may lead to future disputes or operational challenges."
#         #   )


#         if st.button("🔍 Analyze Document"):
#             with st.spinner("Analyzing contract document... ⏳"):
#                 final_answer = granite_simple(user_prompt, temp_file_path)

#             # 🔹 Display Analysis Result
#             st.subheader("📑 Analysis Result")
#             st.write(final_answer)

#             # 🔹 Remove Temporary File
#             os.remove(temp_file_path)

# # 🔥 Run Streamlit App
# if __name__ == '__main__':
#     main()


# import streamlit as st
# import os
# import re
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from PyPDF2 import PdfReader
# from peft import get_peft_model, LoraConfig, TaskType

# # ✅ Force CPU execution for Streamlit Cloud
# device = torch.device("cpu")

# # 🔹 Load IBM Granite Model (CPU-Compatible)
# MODEL_NAME = "ibm-granite/granite-3.1-2b-instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="cpu",  # Force CPU execution
#     torch_dtype=torch.float32  # Use float32 since Streamlit runs on CPU
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# # 🔹 Apply LoRA Fine-Tuning Configuration
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )
# model = get_peft_model(model, lora_config)
# model.eval()

# # 🛠 Function to Read & Extract Text from PDFs
# def read_files(file):
#     file_context = ""
#     reader = PdfReader(file)
    
#     for page in reader.pages:
#         text = page.extract_text()
#         if text:
#             file_context += text + "\n"
    
#     return file_context.strip()

# # 🛠 Function to Format AI Prompts
# def format_prompt(system_msg, user_msg, file_context=""):
#     if file_context:
#         system_msg += f" The user has provided a contract document. Use its context to generate insights, but do not repeat or summarize the document itself."
#     return [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_msg}
#     ]

# # 🛠 Function to Generate AI Responses
# def generate_response(input_text, max_tokens=1000, top_p=0.9, temperature=0.7):
#     model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         output = model.generate(
#             **model_inputs,
#             max_new_tokens=max_tokens,
#             do_sample=True,
#             top_p=top_p,
#             temperature=temperature,
#             num_return_sequences=1,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # 🛠 Function to Clean AI Output
# def post_process(text):
#     cleaned = re.sub(r'戥+', '', text)  # Remove unwanted symbols
#     lines = cleaned.splitlines()
#     unique_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))
#     return "\n".join(unique_lines)

# # 🛠 Function to Handle RAG with IBM Granite & Streamlit
# def granite_simple(prompt, file):
#     file_context = read_files(file) if file else ""
    
#     system_message = "You are IBM Granite, a legal AI assistant specializing in contract analysis."
    
#     messages = format_prompt(system_message, prompt, file_context)
#     input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     response = generate_response(input_text)
#     return post_process(response)

# # 🔹 Streamlit UI
# def main():
#     st.set_page_config(page_title="Contract Analysis AI", page_icon="📜", layout="wide")

#     st.title("📜 AI-Powered Contract Analysis Tool")
#     st.write("Upload a contract document (PDF) for a detailed AI-driven legal and technical analysis.")

#     # 🔹 Sidebar Settings
#     with st.sidebar:
#         st.header("⚙️ Settings")
#         max_tokens = st.slider("Max Tokens", 50, 1000, 250, 50)
#         top_p = st.slider("Top P (sampling)", 0.1, 1.0, 0.9, 0.1)
#         temperature = st.slider("Temperature (creativity)", 0.1, 1.0, 0.7, 0.1)

#     # 🔹 File Upload Section
#     uploaded_file = st.file_uploader("📂 Upload a contract document (PDF)", type="pdf")

#     if uploaded_file is not None:
#         temp_file_path = "temp_uploaded_contract.pdf"
#         with open(temp_file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         st.success("✅ File uploaded successfully!")

#         # 🔹 User Input for Analysis
#         user_prompt = "Perform a detailed technical analysis of the attached contract document, highlighting potential risks, legal pitfalls, compliance issues, and areas where contractual terms may lead to future disputes or operational challenges."

#         if st.button("🔍 Analyze Document"):
#             with st.spinner("Analyzing contract document... ⏳"):
#                 final_answer = granite_simple(user_prompt, temp_file_path)

#             # 🔹 Display Analysis Result
#             st.subheader("📑 Analysis Result")
#             st.write(final_answer)

#             # 🔹 Remove Temporary File
#             os.remove(temp_file_path)

# # 🔥 Run Streamlit App
# if __name__ == '__main__':
#     main()


import streamlit as st
import os
import re
import torch
import pdfplumber
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch


# ✅ Force CPU execution for Streamlit Cloud
device = torch.device("cpu")

# 🔹 Load IBM Granite Model (No Shard Checkpoints)
MODEL_NAME = "ibm-granite/granite-3.1-2b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32,
    offload_folder="offload",
    ignore_mismatched_sizes=True  # 🚀 Fixes sharded checkpoint issues
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 🔹 Apply LoRA Fine-Tuning Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.eval()

# 🛠 Function to Read & Extract Text from PDFs (Using pdfplumber)
def read_files(file):
    file_context = ""
    with pdfplumber.open(file) as reader:
        for page in reader.pages:
            text = page.extract_text()
            if text:
                file_context += text + "\n"
    return file_context.strip()

# 🛠 Function to Format AI Prompts
def format_prompt(system_msg, user_msg, file_context=""):
    if file_context:
        system_msg += " The user has provided a contract document. Use its context to generate insights, but do not repeat or summarize the document itself."
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

# 🛠 Function to Generate AI Responses
def generate_response(input_text, max_tokens=1000, top_p=0.9, temperature=0.7):
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 🛠 Function to Clean AI Output
def post_process(text):
    cleaned = re.sub(r'戥+', '', text)  # Remove unwanted symbols
    lines = cleaned.splitlines()
    unique_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))

    return "\n".join(unique_lines)

# 🛠 Function to Handle RAG with IBM Granite & Streamlit
def granite_simple(prompt, file):
    file_context = read_files(file) if file else ""
    
    system_message = "You are IBM Granite, a legal AI assistant specializing in contract analysis."
    
    messages = format_prompt(system_message, prompt, file_context)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = generate_response(input_text)
    return post_process(response)

# 🔹 Streamlit UI
def main():
    st.set_page_config(page_title="Contract Analysis AI", page_icon="📜", layout="wide")

    st.title("📜 AI-Powered Contract Analysis Tool")
    st.write("Upload a contract document (PDF) for a detailed AI-driven legal and technical analysis.")

    # 🔹 Sidebar Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        max_tokens = st.slider("Max Tokens", 50, 1000, 250, 50)
        top_p = st.slider("Top P (sampling)", 0.1, 1.0, 0.9, 0.1)
        temperature = st.slider("Temperature (creativity)", 0.1, 1.0, 0.7, 0.1)

    # 🔹 File Upload Section
    uploaded_file = st.file_uploader("📂 Upload a contract document (PDF)", type="pdf")

    if uploaded_file is not None:
        temp_file_path = "temp_uploaded_contract.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ File uploaded successfully!")

        # 🔹 User Input for Analysis
        user_prompt = "Perform a detailed technical analysis of the attached contract document, highlighting potential risks, legal pitfalls, compliance issues, and areas where contractual terms may lead to future disputes or operational challenges."

        if st.button("🔍 Analyze Document"):
            with st.spinner("Analyzing contract document... ⏳"):
                final_answer = granite_simple(user_prompt, temp_file_path)

            # 🔹 Display Analysis Result
            st.subheader("📑 Analysis Result")
            st.write(final_answer)

            # 🔹 Remove Temporary File
            os.remove(temp_file_path)

# 🔥 Run Streamlit App
if __name__ == '__main__':
    main()
