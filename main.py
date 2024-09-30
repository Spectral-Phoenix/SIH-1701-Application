import os
import openai
import PyPDF2
import json
import streamlit as st
import google.generativeai as genai
from duckduckgo_search import DDGS
import requests
import io  # Import the io module
import cohere

# Set your Cerebras API key
os.environ["CEREBRAS_API_KEY"] = "csk-4w6rw2kxyk2k3phf35d4pp2x552yvrnx85rtwmkh3c8th5kx"  # Replace with your actual key
client = openai.OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY")
)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY","AIzaSyCVamDnNAeezeywfPTet9t_Fvd_DTTIuu0"))
co = cohere.Client('QNug1dWb6KLsMJxuB3xZ40l6EdyCpGur1iz1mJON')

st.title("Insight AI")

uploaded_file = st.file_uploader("Upload a Plaint document", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Create the prompt for the LLM
    prompt = f"""
    ## Instructions for JSON Output Generation

    You are a legal document analyzer. Your task is to extract key information from the provided legal document and structure it into a JSON format. 

    **Input:**
    A legal document text (provided below).

    **Output:** 
    A JSON object containing the following key-value pairs:

    * `"case_number"`: The case number (string). 
    * `"plaintiff_name"`: The name of the plaintiff (string).
    * `"defendant_name"`: The name of the defendant (string).
    * `"claim_amount"`: The amount claimed (string or number).
    * `"brief_description"`: A brief description of the case (string).
    * `"type_of_case"`: The type of legal case (e.g., "contract dispute", "personal injury") (string).
    * `"under_the_laws"`: The relevant laws or statutes under which the case is filed (string).

    **Important:**
    * **Only return the JSON object.** Do not include any other text or explanations.
    * If a piece of information is not found in the document, leave the corresponding value as an empty string or `null`.

    ---
    **Document Text:** {text}
    """

    # Make the API call to Cerebras
    with st.spinner("Analyzing the document..."):
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

    # Process the response and display the results
    raw_output = response.choices[0].message.content.strip()

    # Clean up the response to extract the JSON
    json_start = raw_output.find("{")
    json_end = raw_output.rfind("}") + 1
    json_string = raw_output[json_start:json_end]

    try:
        json_output = json.loads(json_string)
        st.json(json_output)

        # Search for the relevant law in DuckDuckGo
        if "under_the_laws" in json_output and json_output["under_the_laws"]:
            search_query = f"{json_output['under_the_laws']} filetype:pdf"
            searcher = DDGS()  # Initialize the search object
            results = searcher.text(search_query)  # Perform a text search

            # Loop through search results until a successful PDF is processed
            for result in results:
                if result.get("href"):
                    pdf_url = result["href"]
                    try:
                        with st.spinner(f"Fetching and analyzing PDF: {pdf_url}"):
                            response = requests.get(pdf_url)
                            response.raise_for_status()

                            # Wrap the bytes content in BytesIO
                            pdf_file = io.BytesIO(response.content) 

                            # Extract text from the fetched PDF
                            pdf_text = ""
                            pdf_reader = PyPDF2.PdfReader(pdf_file)  # Use pdf_file here
                            for page_num in range(len(pdf_reader.pages)):
                                page = pdf_reader.pages[page_num]
                                pdf_text += page.extract_text()

                            # Create a new prompt with the JSON output and the relevant law PDF text 
                            new_prompt = f"""
                            You are a legal assistant. You have been provided with information extracted from a legal document (in JSON format) and the text of a relevant law.

                            **JSON Output:**
                            {json.dumps(json_output, indent=4)}

                            **Relevant Law Text:**
                            {pdf_text}

                            **Task:** 
                            Analyze the legal document information in the context of the provided law. Provide a detailed analysis of how the law applies to the case, highlighting key sections of the law that are relevant to the claims made in the document.
                            """
                            generation_config = {
                              "temperature": 1,
                              "top_p": 0.95,
                              "top_k": 64,
                              "max_output_tokens": 8192,
                              "response_mime_type": "text/plain",
                            }
                            
                            model = genai.GenerativeModel(
                              model_name="gemini-1.5-flash-exp-0827",
                              generation_config=generation_config,
                            )

                            chat_session = model.start_chat(
                              history=[
                              ]
                            )

                            response = chat_session.send_message(new_prompt)
                            analysis = response.text.strip()
                            # Display the analysis from the second API call
                            st.write("## Legal Analysis:")
                            st.write(analysis)

                            # If successful, break the loop
                            break 

                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 403:
                            st.warning(f"Skipping PDF: Access forbidden (403) for {pdf_url}")
                        else:
                            st.warning(f"Skipping PDF: Error accessing {pdf_url}: {e}")

                    except Exception as e:  # Catch other potential errors during PDF processing
                        st.warning(f"Skipping PDF: Error processing {pdf_url}: {e}")

            else:  # If the loop completes without finding a successful PDF
                st.error("No accessible PDFs found for the relevant law.")
            
            if st.button("Translate Summary"):
                with st.spinner("Translating..."):
                    try:
                        response = co.generate(
                            model='c4ai-aya-23-35b',
                            prompt=f'Translate the below text into hindi:\n\n{analysis}',
                            max_tokens=8192,
                            temperature=0.9,
                            k=0,
                            stop_sequences=[],
                            return_likelihoods='NONE')
                        
                        translated_text = response.generations[0].text
                        st.write("### Translated Summary (Hindi):")
                        st.write(translated_text)

                    except Exception as e:
                        st.error(f"Error during translation: {e}")
            

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        st.write(f"Raw output from the model: {json_string}")