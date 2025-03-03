import base64
import gradio as gr
import lancedb
import os
import re
from dotenv import load_dotenv
from io import BytesIO
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import LanceDB
from openai import OpenAI
from PIL import Image
from utils import *
from gradio.themes.utils import sizes
import pandas as pd

import shutil
import fitz
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
import torch
import pyarrow as pa


load_dotenv()


HOME_PATH = os.getenv('HOME_PROD')
PROJ_HOME_PATH = os.getenv('PROJ_HOME_PROD')
MM_VECTOR_STORE = os.getenv('MM_VS_PROD')


parameterFn=f'{HOME_PATH}/location_of_api_parameters.csv'
parameterDf=pd.read_csv(parameterFn)

api_url=parameterDf[parameterDf['parameter']=='url']['value'].tolist()[0]
vllmFn=f'{HOME_PATH}/location_of_vllm_list.csv'

vllmDf=pd.read_csv(vllmFn)

port=vllmDf[vllmDf['name']=="llava16_vicuna_7b"]['port'].tolist()[0]
MODEL_ID=vllmDf[vllmDf['name']=="llava16_vicuna_7b"]['llm'].tolist()[0]

VLLM_API_URL=f'http://{api_url}:{port}/v1/'


logger = CustomLogger(f"{HOME_PATH}/location_of_dev_log.log")
logger.log_info(f"Status || Start logging!")


def info_error_log(info_or_error, message):
    logger.log_info(message)
    if info_or_error == "info":
        gr.Info(message, duration=7)
    elif info_or_error == "error":
        raise gr.Error(message, duration=7)


###################################################################################
############################## Chatbot/RAG Function ###############################
###################################################################################


def prep_model():
    print(f"MODEL_ID: {MODEL_ID}")
    print(f"VLLM_API_URL: {VLLM_API_URL}")
    llm = ChatOpenAI(
        model_name=MODEL_ID,
        api_key="EMPTY",
        base_url=VLLM_API_URL,
        max_tokens=512,
        temperature=0.1,
    )

    logger.log_info(f"Status || LLM loaded: {MODEL_ID}")
    return llm


def preparing_vector_store():
    encode_kwargs = {'normalize_embeddings': False}
    model_kwargs = {'device': 'cuda'}

    logger.log_info(f"Status || Loading e5-large-v2...")
    e5_embedding = SentenceTransformerEmbeddings(
        model_name="intfloat/e5-large-v2",
        encode_kwargs=encode_kwargs,
        model_kwargs=model_kwargs
    )
    db = lancedb.connect(f"{HOME_PATH}/location_of_vector_store")
    lancedb_search = LanceDB(connection=db,
                         table_name="9_summarized",
                         embedding=e5_embedding,
                         text_key="original_content",
                         vector_key="embeddings",
                        )
    VS_in_use= "9_summarized"
    logger.log_info(f"Status || All embeddings loaded.")
    return e5_embedding, db, lancedb_search, VS_in_use


def obtain_vector_stores():
    folder = f"{HOME_PATH}/location_of_vector_store"
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    sub_folders = [vs.split('.lance')[0] for vs in sub_folders]
    return sub_folders


def like_response_fn(data: gr.LikeData):
    if data.liked:
        gr.Info("You upvoted a response")
    else:
        gr.Info("You downvoted a response")


def update_vector_store_list():
    updated_list_of_vector_stores = obtain_vector_stores()
    return gr.Dropdown(choices=updated_list_of_vector_stores, type="value", multiselect=False, allow_custom_value=False, label="Available Vector Stores")


def change_vector_store_for_llm(new_vector_store_name):
    global db, lancedb_search, e5_embedding, VS_in_use

    lancedb_search = LanceDB(connection=db,
                         table_name=new_vector_store_name,
                         embedding=e5_embedding,
                         text_key="original_content",
                         vector_key="embeddings",
                        )

    VS_in_use = new_vector_store_name


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    logger.log_info(f"Number of documents retrieved: || {len(docs)}")

    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc_page_content = doc.page_content
        if looks_like_base64(doc_page_content) and is_image_data(doc_page_content):
            doc = resize_base64_image(doc_page_content, size=(1300, 600))
            b64_images.append(doc)
            # image_base64_list.append(doc)
        else:
            texts.append(doc_page_content)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join all the contexts into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for b64_image in data_dict["context"]["images"]:

            image_message = {
                "type": "image_url",
                "image_url": b64_image,
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a smart question answering Assistant.\n"
            "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide a helpful answer related to the user question. \n"
            f"User question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}\n"
            "Assistant:\n"
        ),
    }
    messages.append(text_message)  # list of image_json(s) and text_json
    return messages


def MultiModal_LLM(msgs_input):
    temp_base64_image_list = []
    for each_input in msgs_input:
        if each_input["type"] == "text":
            txt_prompt = each_input["text"]
        elif each_input["type"] == "image_url":
            temp_base64_image_list.append(each_input["image_url"])
    num_images = len(temp_base64_image_list)

    client = OpenAI(
            api_key="EMPTY",
            base_url=VLLM_API_URL,
        )

    if num_images == 0:
        logger.log_info(f"Number of images found in top k documents: || {num_images}")
        chat_completion_from_url = client.chat.completions.create(
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": txt_prompt
                    },
                ],
            }],
            model=MODEL_ID,
            temperature=0.1,
            stream=True,
            max_tokens=512,
            extra_body={
                'repetition_penalty': 1,
            }
        )
    elif num_images == 1:
        logger.log_info(f"Number of images found in top k documents: || {num_images}")
        b64_img = temp_base64_image_list[0]
        chat_completion_from_url = client.chat.completions.create(
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": txt_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        },
                    },
                ],
            }],
            model=MODEL_ID,
            temperature=0.1,
            stream=True,
            max_tokens=512,
            extra_body={
                'repetition_penalty': 1,
            }
        )
    elif num_images > 1:
        logger.log_info(f"Number of images found in top k documents: || {num_images}")
        images = [load_image(img_file) for img_file in temp_base64_image_list]
        image = concatenate_images(images=images, strategy="vertical", dist_images=20, grid_resolution=None)
        image = image.resize((128, 128), Image.LANCZOS)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        b64_img = base64.b64encode(img_byte_arr).decode('utf-8')
        txt_prompt =  txt_prompt.replace("You will be given a mix of text, tables, and image(s) usually of charts or graphs.",
                                         f"You are provided with an image consisting of {num_images} images stitched together, separated by a 20px high horizontal bar.")
        chat_completion_from_url = client.chat.completions.create(
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": txt_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        },
                    },
                ],
            }],
            model=MODEL_ID,
            temperature=0.1, 
            stream=True,
            max_tokens=512,
            extra_body={
                'repetition_penalty': 1,
            }
        )

    logger.log_info(f"VectorStoreName || {VS_in_use}")
    return chat_completion_from_url


def multi_modal_rag_chain(retriever, model):
    """
    Multi-modal RAG chain
    """

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | RunnableLambda(MultiModal_LLM)
    )
    return chain
    

def obtaining_chatbot1_msg(user_message, history):
    return "", history + [[user_message, None]]


def generating_output_store(history, prompt_template):
    logger.log_info(f"LLMOutput || {history[-1][0]}")
    multi_modal_rag_pipeline = multi_modal_rag_chain(lancedb_search.as_retriever(search_kwargs={"k": 6,}), vllm_api)
    stream = multi_modal_rag_pipeline.invoke(history[-1][0])
    history[-1][1] = ""
    
    for chunk in stream:
        history[-1][1] += (chunk.choices[0].delta.content or "")
        yield history


def fetch_retrieved_docs(history):
    logger.log_info(f"Status || LLM loaded: {history[-1][1]}")
    tbl = db.open_table(VS_in_use)
    results = (tbl.search(query = e5_embedding.embed_documents([history[-1][0]])[0],
                          vector_column_name="embeddings").limit(6).to_pandas())
    text_results = results[results["type"] == "text"]
    img_results = results[results["type"] == "image"]
    text_results = text_results.drop(columns=["summarized_content", "embeddings", "type"])
    text_results = text_results[["id", "original_content", "doc_name", "pg_no"]]
    img_results = img_results["original_content"].to_list()
    img_results = [Image.open(BytesIO(base64.b64decode(each))) for each in img_results]
    return text_results, img_results


###################################################################################
####################### Vector Store Management Functions #########################
###################################################################################


def preview_vector_store_fn(vector_store_name):
    preview_table = db.open_table(vector_store_name).to_pandas()
    vector_store_length = len(preview_table)
    doc_list = set()
    doc_list_text = ""
    doc_types = {}

    for each_doc in preview_table["doc_name"].to_list():
        doc_list.add(each_doc)
        the_type = each_doc.split('.')[-1]
        if the_type in doc_types:
            doc_types[the_type] += 1
        else:
            doc_types[the_type] = 1
    total_doc_types = pd.DataFrame(doc_types, index=['i',])
    for each in doc_list:
        doc_list_text = doc_list_text + str(each) + " \n"
    return vector_store_length, len(doc_list), doc_list_text, total_doc_types


def find_similar_images(vector_store_name, check_vs_image):
    if len(check_vs_image["files"]) == 0:
        tbl = db.open_table(vector_store_name)

        results = (tbl.search(query = e5_embedding.embed_documents([check_vs_image["text"]])[0],
                            vector_column_name="embeddings")
                            .where("type = 'image'")
                            .limit(5)
                            .to_pandas()
                            )
        results = results["original_content"].to_list()
        if len(results) == 0:
            gr.Info("No similar images found")
        else:
            results = [Image.open(BytesIO(base64.b64decode(each))) for each in results]
            return results
    else:
        the_image =Image.open(check_vs_image['files'][0])
        img_byte_arr = BytesIO()
        the_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        b64_img = base64.b64encode(img_byte_arr).decode('utf-8')

        client = OpenAI(
            api_key="EMPTY",
            base_url=VLLM_API_URL,
        )

        chat_completion_from_url = client.chat.completions.create(
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the provided image as detailed as you can."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # "url": img
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        },
                    },
                ],
            }],
            model=MODEL_ID,
            temperature=0.1, 
            stream=False, 
            max_tokens=512,
            extra_body={
                'repetition_penalty': 1,
            }
        )
        result = chat_completion_from_url.choices[0].message.content
        tbl = db.open_table(vector_store_name)
        results = (tbl.search(query = e5_embedding.embed_documents([result])[0],
                            vector_column_name="embeddings")
                            .where("type = 'image'")
                            .limit(5)
                            .to_pandas()
                            )
        results = results["original_content"].to_list()
        if len(results) == 0:
            gr.Info("No similar images found")
        else:
            results = [Image.open(BytesIO(base64.b64decode(each))) for each in results]
            return results
        

###################################################################################
###################### Creating New Vector Store Functions ########################
###################################################################################


# Extract elements from PDF
def extract_pdf_elements(path, fname):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    return partition_pdf(
        filename=os.path.join(path, fname),
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        infer_table_structure=True,
        strategy='hi_res',
        chunking_strategy="by_title",
        max_characters=2000,
        new_after_n_chars=1300,
        combine_text_under_n_chars=1000,
        include_metadata=True,
    )


# Categorize elements by type
def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in tqdm(raw_pdf_elements):
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

def cat_counter(raw_pdf_elements):
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1


# https://stackoverflow.com/questions/77787505/how-to-get-table-data-collected-from-json-parsed-from-an-unstructured-pdf-file
# https://github.com/Unstructured-IO/unstructured/issues/2603
def get_metadata(raw_pdf_elements):
    metadata_text_doc_name = []
    metadata_text_pg_no = []
    metadata_table_doc_name = []
    metadata_table_pg_no = []
    base64_table_list = []
    metadata_image_doc_name = []
    metadata_image_pg_no = []
    base64_image_list = []
    
    ###### to iterate through each unstructured composite element        
    for each_pdf_element in raw_pdf_elements:
        ###### this if else section will understand if it's text or table for and create the respective metadata accordingly
        if each_pdf_element.to_dict()["type"] == "CompositeElement":
            ### text metadata is more straight forward, just extracting from metadata
            metadata_text_doc_name.append(each_pdf_element.to_dict()["metadata"]["filename"])
            metadata_text_pg_no.append(each_pdf_element.to_dict()["metadata"]["page_number"])
        elif each_pdf_element.to_dict()["type"] == "Table":
            ### table metadata requires splitting the composite element into original elements
            ### from the original elements metadata, we can then retrieve
            ### filename from composite element, page number and base64 format of table  from original element
            for each_og_element in each_pdf_element.metadata.orig_elements:
                metadata_table_doc_name.append(each_pdf_element.metadata.to_dict()["filename"])
                metadata_table_pg_no.append(each_og_element.to_dict()["metadata"]["page_number"])
                base64_table_list.append(each_og_element.to_dict()["metadata"]["image_base64"])

        ###### this section would then retrieve metadata for images
        ### this would dive straight into original elements as the composite elements are made up of different elements, combined together
        for each_og_element in each_pdf_element.metadata.orig_elements:
            ### after splitting into original elements, it will iterate through to find Image components
            if each_og_element.to_dict()["type"] == "Image":
                ### then retrieve filename from composite element, page number and base64 format of image from original element
                metadata_image_doc_name.append(each_pdf_element.metadata.to_dict()["filename"])
                metadata_image_pg_no.append(each_og_element.metadata.to_dict()["page_number"])
                base64_image_list.append(each_og_element.metadata.to_dict()["image_base64"])
    
    return metadata_text_doc_name, metadata_text_pg_no,\
           metadata_table_doc_name, metadata_table_pg_no, base64_table_list,\
           metadata_image_doc_name, metadata_image_pg_no, base64_image_list
    
    
def document_extractor(path_to_extract):
    docs_to_extract = []   
    combined_raw_pdf_elements = []   
    combined_texts = []   
    combined_tables = []
    combined_metadata_text_doc_name = []
    combined_metadata_text_pg_no = []
    combined_metadata_table_doc_name = []
    combined_metadata_table_pg_no = []
    combined_base64_table_list = []
    combined_metadata_image_doc_name = []
    combined_metadata_image_pg_no = []
    combined_base64_image_list = []


    for file in os.listdir(path_to_extract):
        if file.endswith(".pdf"):
            docs_to_extract.append(file)

    logger.log_info(f"Extracting the following documents: || {docs_to_extract}")

    for each_document in tqdm(docs_to_extract): 
        # Get raw elements
        raw_pdf_elements = extract_pdf_elements(path_to_extract, each_document) # we will get a list of unstructured components
        combined_raw_pdf_elements.append(raw_pdf_elements) # append the list of unstructured documents to 

        # Get text, tables
        metadata_text_doc_name, metadata_text_pg_no, metadata_table_doc_name, metadata_table_pg_no, base64_table_list, metadata_image_doc_name, metadata_image_pg_no, base64_image_list = get_metadata(raw_pdf_elements)
        texts, tables = categorize_elements(raw_pdf_elements)
        
        combined_texts.extend(texts)
        combined_tables.extend(tables)
        combined_metadata_text_doc_name.extend(metadata_text_doc_name)
        combined_metadata_text_pg_no.extend(metadata_text_pg_no)
        combined_metadata_table_doc_name.extend(metadata_table_doc_name)
        combined_metadata_table_pg_no.extend(metadata_table_pg_no)
        combined_base64_table_list.extend(base64_table_list)
        combined_metadata_image_doc_name.extend(metadata_image_doc_name)
        combined_metadata_image_pg_no.extend(metadata_image_pg_no)
        combined_base64_image_list.extend(base64_image_list)

    cat_counter(raw_pdf_elements)

    return combined_raw_pdf_elements,\
          combined_texts, combined_tables, combined_metadata_text_doc_name, combined_metadata_text_pg_no,\
          combined_metadata_table_doc_name, combined_metadata_table_pg_no, combined_base64_table_list,\
          combined_metadata_image_doc_name, combined_metadata_image_pg_no, combined_base64_image_list


# Generate summaries of text elements
def generate_text_summaries(texts, tables, summarize_texts=True):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    prompt_text = """Give a detailed summary of the following: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model_path = "google/flan-t5-large" #  flan-t5-large, xxl
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #max_length has typically been deprecated for max_new_tokens, max_length=512,
    pipe = pipeline(
        "summarization", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256, 
        device="cuda"
    )
    llm = HuggingFacePipeline(pipeline=pipe, verbose=True)
    
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1024})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1024})

    return text_summaries, table_summaries


def image_summarize(each_base64_image):
    """Make image summary"""

    prompt_template = "In the process of creating a vector store of image summaries. The assistant gives helpful, detailed and descriptive summaries of the user's input. USER: <image>\n"
    client = OpenAI(
            api_key="EMPTY",
            base_url=VLLM_API_URL,
        )

    chat_completion_from_url = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{each_base64_image}"
                    },
                },
            ],
        }],
        model=MODEL_ID,
        temperature=0.1, 
        stream=False,
        max_tokens=512,
        extra_body={
            'repetition_penalty': 1,
        }
    )
    return chat_completion_from_url.choices[0].message.content

def generate_img_summaries(base64_image_list):
    # Store image summaries
    image_summaries = []

    prompt = """Describe what is in the image. Be as detailed as you can be in providing texts and details from the image:"""

    for each_base64_image in tqdm(base64_image_list):
        image_summaries.append(image_summarize(each_base64_image, prompt))
            
    return image_summaries

def img_summariser_wrapper(base64_image_list):

    # Generate Image Summaries
    image_summaries = generate_img_summaries(base64_image_list)

    return image_summaries


def delete_all_temp_files():
    logger.log_info(f"Status || Deleting tmp folders...")
    folders_to_delete = [f'{HOME_PATH}/location_of_tmp_image_to_display/',
                         f'{HOME_PATH}/location_of_tmp_mm_rag_temp/misc/',
                         f'{HOME_PATH}/location_of_tmp_confirm_convert/',]
                         # f"{PROJ_HOME_PATH}/temp_result"]
    for each_folder in folders_to_delete:
        try:
            for f in os.listdir(each_folder):
                os.remove(os.path.join(each_folder, f))
        except Exception as e:
            info_error_log("Error", f"Error || An expectation occurred: {e}")


def confirm_upload_delete_fn(files, folder, type_of_upload, vector_stores_to_delete):
    if type_of_upload == "Delete Vector Store":
        if vector_stores_to_delete == "9_summarized":
            info_error_log("error", "You do not have permission to delete vector store '9_summarized'")
            return None, None, None, "You do not have permission to delete vector store '9_summarized'"#, None
            
        store_to_delete = f"{HOME_PATH}/location_of_vector_store"
        try:
            shutil.rmtree(store_to_delete)
            # updated_list_of_vector_stores = obtain_vector_stores()
            info_error_log("info", f"{vector_stores_to_delete} successfully deleted!")
            return None, None, None, f"{vector_stores_to_delete} successfully deleted!"#, [i for i in updated_list_of_vector_stores if i != vector_stores_to_delete]
            
        except:
            info_error_log("error", f"Unable to delete {vector_stores_to_delete}!")
            return None, None, None, f"Unable to delete {vector_stores_to_delete}!"#, None
            
    elif type_of_upload == "Upload Multiple Files":
        files = files
    elif type_of_upload == "Upload Folder":
        files = folder
    ### Section 1: Copy the files from the Gradio temp folder into our temp folder
    file_names = [f.name for f in files]
    for each_file in file_names:
        if each_file.endswith('.pdf'):# or each_file.endswith('.docx') or each_file.endswith('.pptx') or each_file.endswith('.png') or each_file.endswith('.jpg'):
            path = f"{HOME_PATH}/location_of_tmp_mm_rag_temp/" + os.path.basename(each_file)
            shutil.copyfile(each_file, path)
        else:
            info_error_log("Error", "Data type not supported. Please ensure only PDF files are uploaded")
            return None, None, None, None, None
    
    ### Section 2: Using PyMuPDF (fitz) to generate images from uploaded documents
    ### This section will also create a list of uploaded file names to be sent into the checkbox
    checkbox_names = []
    docs_not_previewed = []
    # video_df = pd.DataFrame(columns=["Name", "Duration", "FPS", "Frame Count", "Size"])
        
    for filename in os.listdir(f"{HOME_PATH}/location_of_tmp_mm_rag_temp"):
        checkbox_names.append(filename)
        if os.path.getsize(f"{HOME_PATH}/location_of_tmp_mm_rag_temp_misc_{filename}") > 12000000:# or filename.endswith('.docx') or filename.endswith('.pptx'):
            docs_not_previewed.append(filename)
            continue
        
        doc = fitz.open(f"{HOME_PATH}/location_of_tmp_mm_rag_temp_misc_{filename}")
        for i, page in enumerate(doc):
            pix = page.get_pixmap()  # render page to an image
            pix.save(f"{HOME_PATH}/location_of_tmp_mm_rag_temp_image_to_display/{filename}page_{i}.png")

    ### Section 3: Compiling created images paths and their names into a list of tuple for Gradio Gallery component
    list_images = [filename for filename in os.listdir(f"{HOME_PATH}/location_of_tmp_mm_rag_temp_image_to_display/")]
    images = [
        (f"{HOME_PATH}/location_of_tmp_mm_rag_temp_image_to_display/{list_images[i]}"
         , f"{list_images[i]}"
         )
        for i in range(len(list_images))
    ]
    return images,\
gr.CheckboxGroup(choices=checkbox_names, value=checkbox_names),\
gr.Textbox(value=docs_not_previewed), None#, None


def create_vector_store_fn(new_vector_store_name, file_names, progress=gr.Progress()):
    if len(file_names) == 0:
        info_error_log("error", "Unsuccessful! Please select at least 1 document to create a vector store.")
        return f"# Unsuccessful! \n### Please select at least 1 document to create a vector store."
    if not new_vector_store_name or new_vector_store_name == "":
        info_error_log("error", "Unsuccessful! Please type a name for the new vector store above.")
        return f"# Unsuccessful! \n### Please type a name for the new vector store above."
    list_of_vector_stores = obtain_vector_stores()
    if any(new_vector_store_name in x  for x in list_of_vector_stores):
        info_error_log("error", "Unsuccessful! Vector Store name selected already exists!.")
        return f"# Unsuccessful! \n### Vector Store name selected already exists!."
    progress(0, desc="Starting...")

    process_data_for_vector_store(file_names, new_vector_store_name, progress)

    info_error_log("info", f"Successful! Vector Store {new_vector_store_name} created.")
    return f"# Successful! \n### Vector Store {new_vector_store_name} created."

def process_data_for_vector_store(file_names, new_vector_store_name, progress):
    for each_file in file_names:
        if each_file.endswith(".pdf"):
            shutil.copyfile(os.path.join(f"{HOME_PATH}/location_of_tmp_mm_rag_temp_misc", each_file),
                            os.path.join(f"{HOME_PATH}/location_of_tmp_mm_rag_temp_confirm_convert", each_file))

    progress(0.1, desc="Extracting content from PDF...")
    info_error_log("info", "Starting vector store creation... \nExtracting content from PDF...")
    combined_raw_pdf_elements, combined_texts, combined_tables, metadata_text_doc_name, metadata_text_pg_no, metadata_table_doc_name, metadata_table_pg_no, base64_table_list, metadata_image_doc_name, metadata_image_pg_no, base64_image_list = document_extractor(f"{HOME_PATH}/tmp/mm_rag_temp/confirm_convert")
    info_error_log("info", f"PDF extraction complete. \nNumber of texts chunks: {len(combined_texts)}. \nNumber of tables: {len(combined_tables)}. \nNumber of images: {len(base64_image_list)}")

    progress(0.25, desc="Summarising of documents/tables. This could take a while...")
    info_error_log("info", "Starting summarisation of documents/tables")
    text_summaries, table_summaries = generate_text_summaries(
        combined_texts, combined_tables , summarize_texts=True  # texts_4k_token
        )
    info_error_log("info", "Summarisation of documents/tables completed")
    
    progress(0.65, desc="Summarising of images. This could take a while...")
    if len(base64_image_list) > 0:
        info_error_log("info", "Starting summarisation of images")
        image_summaries = img_summariser_wrapper(base64_image_list)
        info_error_log("info", "Summarisation of images completed.")
    else:
        info_error_log("info", "No images to be summarised")
        image_summaries = []

    progress(0.8, desc="Embeddings data...")
    text_summaries_embeddings = e5_embedding.embed_documents(text_summaries)
    table_summaries_embeddings = e5_embedding.embed_documents(table_summaries)
    if len(base64_image_list) > 0:
        image_summaries_embeddings = e5_embedding.embed_documents(image_summaries)

    progress(0.9, desc="Creating vector store...")
    info_error_log("info", "Creating vector store")
    schema = pa.schema(
        [
            pa.field("original_content", pa.string()),
            pa.field("summarized_content", pa.string()),
            pa.field("embeddings", pa.list_(pa.float32(), 1024)),
            pa.field("id", pa.int32()),
            pa.field("type", pa.string()),
            pa.field("doc_name", pa.string()),
            pa.field("pg_no", pa.int32()),
        ]
    )
    data = []
    iteration = 0

    for i in tqdm(range(0, len(combined_texts))):
        data.append(
            {"original_content": combined_texts[i],
            "summarized_content": text_summaries[i],
            "embeddings": text_summaries_embeddings[i],
            "id": iteration,
            "type": "text",
            "doc_name": metadata_text_doc_name[i],
            "pg_no": metadata_text_pg_no[i],
            }
        )
        iteration += 1

    for j in tqdm(range(0, len(combined_tables))):
        data.append(
            {"original_content": combined_tables[j],
            "summarized_content": table_summaries[j],
            "embeddings": table_summaries_embeddings[j],
            "id": iteration,
            "type": "table",
            "doc_name": metadata_table_doc_name[j],
            "pg_no": metadata_table_pg_no[j],
            }
        )
        iteration += 1

    for k in tqdm(range(0, len(base64_image_list))):
        data.append(
            {"original_content": base64_image_list[k],
            "summarized_content": image_summaries[k],
            "embeddings": image_summaries_embeddings[k],
            "id": iteration,
            "type": "image",
            "doc_name": metadata_image_doc_name[k],
            "pg_no": metadata_image_pg_no[k],
            }
        )
        iteration += 1
    tbl = db.create_table(new_vector_store_name, data, schema=schema, mode="overwrite")


###################################################################################
############################## Access Rights Demo #################################
###################################################################################


def login_ar_fn(login_username, login_password):
    global granted_level
    if login_username == "manager" and login_password == "12345":
        granted_level = "manager"
        info_error_log("info", "Successfully logged in as manager!")
    elif login_username == "normal" and login_password == "123":
        granted_level = "normal"
        info_error_log("info", "Successfully logged in as normal user!")
    else:
        granted_level = "denied"
        info_error_log("error", "No such account was found.")


def preview_vector_store_ar_fn(vector_store_name):
    preview_table = db.open_table(vector_store_name).to_pandas()
    preview_table = preview_table[preview_table["access_rights"] == granted_level]
    vector_store_length = len(preview_table)
    doc_list = set()
    doc_list_text = ""
    doc_types = {}

    for each_doc in preview_table["doc_name"].to_list():
        doc_list.add(each_doc)
        the_type = each_doc.split('.')[-1]
        if the_type in doc_types:
            doc_types[the_type] += 1
        else:
            doc_types[the_type] = 1
    total_doc_types = pd.DataFrame(doc_types, index=['i',])
    for each in doc_list:
        doc_list_text = doc_list_text + str(each) + " \n"
    return vector_store_length, len(doc_list), gr.Radio(choices=doc_list), total_doc_types#, preview_table


def show_doc_df_fn(vector_store_name, vector_store_doc_list_ar):
    preview_table = db.open_table(vector_store_name).to_pandas()
    preview_table = preview_table[preview_table["access_rights"] == granted_level]
    preview_table = preview_table[preview_table["doc_name"]==vector_store_doc_list_ar]
    preview_table = preview_table.drop(columns=["embeddings", "doc_name", "access_rights"])
    return preview_table



###################################################################################
################################ Main Application #################################
###################################################################################


theme = gr.themes.Default(text_size = sizes.Size(
            name="text_xl",
            xxs="16px",
            xs="24px",
            sm="26px",
            md="28px",
            lg="30px",
            xl="34px",
            xxl="38px",
        )).set(
    block_border_width="1px",
    border_color_primary="#cfcaca",
        )


css = """
footer {visibility: hidden}

"""


def multi_modal_demo():
    global vllm_api
    global db
    global lancedb_search
    global e5_embedding
    global VS_in_use
    global granted_level
    granted_level = "denied"


    vllm_api = prep_model()
    e5_embedding, db, lancedb_search, VS_in_use = preparing_vector_store()

    with gr.Blocks(analytics_enabled=False, 
                #    theme=theme, 
                   css = css) as multi_modal_interface:
        with gr.Row():
            with gr.Column(scale=15):
                gr.Markdown("""
                            # Multi Modal RAG
                            """)

            with gr.Column(scale=1, min_width=200):
                toggle_dark = gr.Button(value="💡🌙")
        with gr.Tab("Multi-Modal LLM"):
            with gr.Row():
                gr.Markdown("# LLM with vector store inferencing")
            with gr.Row():
                with gr.Column():
                    LLM_types = gr.Dropdown(
                        choices=["llava-v1.6-vicuna-7b-hf"], type="value",
                        multiselect=False, allow_custom_value=False,
                        label="Available LLMs", interactive=False, value="llava-v1.6-vicuna-7b-hf"
                    )
                with gr.Column():
                    list_of_vector_stores = obtain_vector_stores()
                    vector_stores_for_LLM = gr.Dropdown(
                        choices=list_of_vector_stores, type="value",
                        multiselect=False, allow_custom_value=False,
                        label="Available Vector Stores", interactive=True
                    )
                with gr.Column():
                    refresh_vector_store_for_llm = gr.Button(value="Refresh", scale=1)

            with gr.Accordion("Prompt Template", open=False):
                prompt_template_tb = gr.Textbox(
                    value="""You are a smart question answering Assistant. 
                             You will be given a mix of text, tables, and image(s) usually of charts or graphs.
                             Use these information to provide a helpful answer related to the user question without mentioning them (E.g., do not say 'This image shows'... or 'Based on the provided image').
                             If you don't know the answer, just say that you don't know, don't try to make up an answer.
                             
                             User question: {question}
                             
                             Text and / or tables:
                             {retrieved_documents}
                             Assistant:
                             """,
                    lines=9, max_lines=15, interactive=True)
            chatbot1 = gr.Chatbot(label="Multi-modal chatbot with vector store referencing",
                                    height=600, show_copy_button=True,
                                    value=[["Hello, how can you help me?",
                                            "Hello, I am an assistant powered by llava-v1.6-vicuna-7b-hf. You may ask me questions related to the vector store you have selected above and I will try my best to answer them. \n\nHow can I help you? 😊"]])
            with gr.Row():
                msg = gr.Textbox(label="Input prompt", interactive=True, placeholder="Ask a question!")
            with gr.Row():
                with gr.Column():
                    clear = gr.ClearButton([msg, chatbot1])
                with gr.Column():
                    btn = gr.Button(value="Submit", variant="primary")

            with gr.Row():
                gr.Examples(label="Example Questions", 
                            # run_on_click=True,
                    examples=[
                    "What is Retrieval Augmented Generation (RAG) and the process?",
                    "I would like to implement a generative model pipeline for a Decision intelligence and Planning. Do you think this is advisable? If not, please advise other better techniques.",
                    "What percentage of respondents are piloting or productionizing Generative AI?",
                    "How can Gen AI use-cases be categorized?"
                ], inputs=msg)

            with gr.Accordion("Documents/Images used for chatbot with vector store inferencing", open=False):
                with gr.Row():
                    doc_table = gr.Dataframe(label="Referenced Documents",
                                                        interactive=False,
                                                        wrap=True,
                                                        height=900, headers=["original_content",
                                                                            "doc_name",
                                                                            "pg_no"])
                with gr.Row():
                    retrieved_images = gr.Gallery(label="Referenced Pictures", interactive=False, columns=2, allow_preview=True)

            
        with gr.Tab("Vector Store Management"):
            with gr.Row():
                list_of_vector_stores = obtain_vector_stores()
                available_vector_stores = gr.Dropdown(choices=list_of_vector_stores,
                                                      type="value",
                                                      multiselect=False,
                                                      allow_custom_value=False,
                                                      label="Available Vector Stores")
            with gr.Row():
                with gr.Column():
                    refresh_vector_store = gr.Button(value="Refresh for updated vector stores")
                with gr.Column():
                    view_vector_store = gr.Button(value="Preview Store",
                                                  variant="primary")
                    
            with gr.Row(visible=False) as preview_vector_store1:
                with gr.Column(scale=4):
                    with gr.Row():
                        vector_store_size = gr.Textbox(
                            label="No. of chunks",
                            lines=1
                            )
                        vector_store_doc_count = gr.Textbox(
                            label="No. of documents",
                            lines=1
                            )
                    with gr.Row():
                        file_counter_display = gr.Dataframe(label="Count of chunk(s)")
                with gr.Column(scale=5):
                    vector_store_doc_list = gr.Textbox(
                        label="Names of documents in vector store",
                        max_lines=10)

            with gr.Row(visible=False) as preview_vector_store2:
                check_vs_image = gr.MultimodalTextbox(label="Check Vector Store for Image",
                                                        info="Ask a question or upload a picture to find relevant pictures",
                                                        interactive=True,
                                                        placeholder="Ask a question!",
                                                        file_types=["image"],
                                                        file_count="single",
                                                        submit_btn=True
                                                        )
                        
            with gr.Row(visible=False) as preview_vector_store3:
                similar_img = gr.Gallery(label="Similar Pictures", interactive=False, columns=2, allow_preview=True,
                                         height=600)

            def preview_store():
                return {preview_vector_store1: gr.update(visible=True), preview_vector_store2: gr.update(visible=True)}
            def preview_store3():
                return {preview_vector_store3: gr.update(visible=True)}

        with gr.Tab("Create/Delete New Vector Store"):
            with gr.Group():
                with gr.Row():
                    type_of_upload = gr.Radio(
                        label="Select choice of action",
                        choices=["Upload Multiple Files", "Upload Folder", "Delete Vector Store"],
                        value="Upload Multiple Files",
                        )
                with gr.Row(visible=True) as multiple_file_upload:
                    upload_pdf = gr.Files(label="Upload data here", 
                                        show_label=True,
                                        file_types=['.pdf'],
                                        interactive=True)

                with gr.Row(visible=False) as folder_upload:
                    upload_pdf_folder = gr.File(label="Upload data here",
                                                show_label=True,
                                                file_count="directory",
                                                interactive=True)
                with gr.Row(visible=False) as delete_vs:
                    with gr.Column():
                        list_of_vector_stores_to_delete = obtain_vector_stores()
                        vector_stores_to_delete = gr.Dropdown(
                            choices=list_of_vector_stores_to_delete, type="value",
                            multiselect=False, allow_custom_value=False,
                            label="Available Vector Stores", interactive=True
                        )
                    with gr.Column():
                        refresh_vector_store_to_delete = gr.Button(value="Refresh for updated vector stores", scale=1)

                with gr.Row():
                    with gr.Column():
                        delete_uploaded_docs = gr.ClearButton()
                    with gr.Column():
                        confirm_upload_delete = gr.Button(value="Confirm",
                                                        variant="primary")
                        
                def select_uploader(choice):
                    if choice == "Upload Multiple Files":
                        return {multiple_file_upload: gr.update(visible=True),
                                folder_upload: gr.update(visible=False),
                                delete_vs: gr.update(visible=False),
                                preview_docs_row1: gr.update(visible=False),
                                not_previewed_row: gr.update(visible=False),
                                preview_docs_row2: gr.update(visible=False),
                                preview_docs_row3: gr.update(visible=False),
                                delete_vs_row1: gr.update(visible=False), 
                                upload_complete_create: gr.update(visible=False)}
                    elif choice == "Upload Folder":
                        return {multiple_file_upload: gr.update(visible=False),
                                folder_upload: gr.update(visible=True),
                                delete_vs: gr.update(visible=False),
                                preview_docs_row1: gr.update(visible=False),
                                not_previewed_row: gr.update(visible=False),
                                preview_docs_row2: gr.update(visible=False),
                                preview_docs_row3: gr.update(visible=False),
                                delete_vs_row1: gr.update(visible=False), 
                                upload_complete_create: gr.update(visible=False)}
                    elif choice == "Delete Vector Store":
                        return {multiple_file_upload: gr.update(visible=False),
                                folder_upload: gr.update(visible=False),
                                delete_vs: gr.update(visible=True),
                                preview_docs_row1: gr.update(visible=False),
                                not_previewed_row: gr.update(visible=False),
                                preview_docs_row2: gr.update(visible=False),
                                preview_docs_row3: gr.update(visible=False),
                                delete_vs_row1: gr.update(visible=False), 
                                upload_complete_create: gr.update(visible=False)}

                

            with gr.Group():
                with gr.Row(visible=False) as preview_docs_row1:
                    pdf_preview = gr.Gallery(label="Preview of PDF files to upload", rows=2, columns=4)

                def delete_gallery():
                    return {pdf_preview: gr.update(label="Preview of PDF files to upload", rows=2, columns=4, value=None)}

                with gr.Row(visible=False) as not_previewed_row:
                    not_previewed = gr.Textbox(value="None", lines=3, max_lines=10,
                                            label="Documents not previewed above",
                                            info="PDFs of size > 12Mb or other types of documents (pptx, docx)")
                with gr.Row(visible=False) as preview_docs_row2:
                    double_confirm_upload = gr.CheckboxGroup(
                        label="Confirm Documents",
                        info="Checked documents would be uploaded",
                        interactive=True
                    )
            with gr.Row(visible=False) as preview_docs_row3:
                new_vector_store_name = gr.Textbox(
                    label="Create a new Vector Store",
                    info="Input name for the new vector store",
                    lines=1
                )
                create_vector_store = gr.Button(value="Create Vector Store with Uploaded Documents", variant="primary")
            with gr.Row(visible=False) as delete_vs_row1:
                delete_message = gr.Markdown()

            def upload_part2(type_of_upload):
                if type_of_upload == "Upload Multiple Files" or type_of_upload == "Upload Folder":
                    return {preview_docs_row1: gr.update(visible=True),
                            not_previewed_row: gr.update(visible=True),
                            preview_docs_row2: gr.update(visible=True),
                            preview_docs_row3: gr.update(visible=True),
                            delete_vs_row1: gr.update(visible=False)}
                elif type_of_upload == "Delete Vector Store":
                    return {preview_docs_row1: gr.update(visible=False),
                            not_previewed_row: gr.update(visible=False),
                            preview_docs_row2: gr.update(visible=False),
                            preview_docs_row3: gr.update(visible=False),
                            delete_vs_row1: gr.update(visible=True)}

            def opposite_of_upload_part2():
                return {preview_docs_row1: gr.update(visible=False),
                        not_previewed_row: gr.update(visible=False),
                        preview_docs_row2: gr.update(visible=False),
                        preview_docs_row3: gr.update(visible=False),
                        delete_vs_row1: gr.update(visible=False)}

            with gr.Row():
                with gr.Column(visible=False) as upload_complete_merge:
                    upload_complete_message_merge = gr.Markdown()

                with gr.Column(visible=False) as upload_complete_create:
                    upload_complete_message_create = gr.Markdown()

            def upload_part3():
                return {upload_complete_merge: gr.update(visible=True)}

            def upload_part4():
                return {upload_complete_create: gr.update(visible=True)}

        with gr.Tab("Notes"):
            gr.Markdown("""
                        # Models tried: <br> 
                        1. llava-hf/llava-v1.6-vicuna-7b-hf (most detailed) <br>
                        2. THUDM/cogvlm-chat-hf 
                        3. Qwen/Qwen-VL
                        """
                        )
        with gr.Tab("Access Rights Demo", visible=True):
            with gr.Group(visible=True) as login_ar:
                with gr.Row():
                    login_username = gr.Textbox(label="Username", value="manager", interactive=True)
                with gr.Row():
                    login_password = gr.Textbox(label="Password", type="password", interactive=True)
                with gr.Row():
                    login_btn_ar = gr.Button(value="Login")
            
            with gr.Row(visible=False) as logout_ar:
                with gr.Column(scale=10):
                    gr.Markdown("")
                with gr.Column(scale=1):
                    logout_btn_ar = gr.Button("Logout")
            with gr.Row(visible=False) as choose_vs_ar:
                list_of_vector_stores = obtain_vector_stores()
                available_vector_stores_ar = gr.Dropdown(choices=list_of_vector_stores,
                                                      type="value",
                                                      multiselect=False,
                                                      allow_custom_value=False,
                                                      label="Available Vector Stores",
                                                      value="9_summarized_access_rights")
            with gr.Row(visible=False) as confirm_vs_ar:
                with gr.Column():
                    refresh_vector_store_ar = gr.Button(value="Refresh for updated vector stores")
                with gr.Column():
                    view_vector_store_ar = gr.Button(value="Preview Store",
                                                  variant="primary")
                    
            with gr.Row(visible=False) as view_vector_store_ar1:
                with gr.Column(scale=4):
                    with gr.Row():
                        vector_store_size_ar = gr.Textbox(
                            label="No. of chunks",
                            lines=1
                            )
                        vector_store_doc_count_ar = gr.Textbox(
                            label="No. of documents",
                            lines=1
                            )
                    with gr.Row():
                        file_counter_display_ar = gr.Dataframe(label="Count of chunk(s)")
                with gr.Column(scale=5):
                    vector_store_doc_list_ar = gr.Radio(
                        label="Names of documents in vector store", interactive=True)
                    
            with gr.Row(visible=False) as view_vector_store_df_ar1:
                ar_df = gr.Dataframe(interactive=False, wrap=True, headers=["original_content",
                                                                            "summarized_content",
                                                                            "id",
                                                                            "type",
                                                                            "pg_no"])

                        
            with gr.Row(visible=False) as view_vector_store_ar3:
                similar_img_ar = gr.Gallery(label="Similar Pictures", interactive=False, columns=2, allow_preview=True,
                                         height=600)

            def login_ar_gui():
                if granted_level == "manager" or granted_level == "normal":
                    return {login_ar: gr.update(visible=False),
                            logout_ar: gr.update(visible=True),
                            choose_vs_ar: gr.update(visible=True),
                            confirm_vs_ar: gr.update(visible=True)}
                else:
                    return {login_ar: gr.update(visible=True),
                            logout_ar: gr.update(visible=False),
                            choose_vs_ar: gr.update(visible=False),
                            confirm_vs_ar: gr.update(visible=False)}
            def logout_ar_gui():
                global granted_level
                info_error_log("info", f"Successfully logged out of {granted_level} account")
                granted_level = "denied"
                return {login_ar: gr.update(visible=True),
                        logout_ar: gr.update(visible=False),
                        choose_vs_ar: gr.update(visible=False),
                        confirm_vs_ar: gr.update(visible=False),
                        view_vector_store_ar1: gr.update(visible=False),
                        view_vector_store_df_ar1: gr.update(visible=False),
                        view_vector_store_ar3: gr.update(visible=False),}

            def view_store_ar1():
                return {view_vector_store_ar1: gr.update(visible=True),
                        view_vector_store_df_ar1: gr.update(visible=True),
                        view_vector_store_df_ar1: gr.update(visible=True)}
            def view_store_ar2():
                return {view_vector_store_ar3: gr.update(visible=True)}


        ###################################################################################
        ################################ Dark mode toggle #################################
        ###################################################################################
        toggle_dark.click(
            None,
            js="""
            () => {
                document.body.classList.toggle('dark');
                document.body.classList.toggle('vsc-initialized dark');
                document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
            }
            """, )

        ###################################################################################
        ################################# Chatbot Gradio ##################################
        ###################################################################################
        msg.submit(fn=obtaining_chatbot1_msg, inputs=[msg, chatbot1], outputs=[msg, chatbot1], queue=False) \
                .then(fn=generating_output_store, inputs=[chatbot1, prompt_template_tb], outputs=[chatbot1]) \
                .then(fn=fetch_retrieved_docs, inputs=[chatbot1], outputs=[doc_table, retrieved_images])
        btn.click(fn=obtaining_chatbot1_msg, inputs=[msg, chatbot1], outputs=[msg, chatbot1], queue=False) \
                .then(fn=generating_output_store, inputs=[chatbot1, prompt_template_tb], outputs=[chatbot1]) \
                .then(fn=fetch_retrieved_docs, inputs=[chatbot1], outputs=[doc_table, retrieved_images])
        refresh_vector_store_for_llm.click(fn=update_vector_store_list, outputs=vector_stores_for_LLM)
        vector_stores_for_LLM.change(fn=change_vector_store_for_llm, 
                            inputs=vector_stores_for_LLM)
        
        ###################################################################################
        ######################## Vector Store Management Gradio ###########################
        ###################################################################################
        refresh_vector_store.click(fn=update_vector_store_list, outputs=available_vector_stores)
        # refresh_integrate_vector_store.click(fn=update_vector_store_list, outputs=vector_db_for_integration1)
        view_vector_store.click(fn=preview_store, outputs=[preview_vector_store1, preview_vector_store2])\
                         .then(fn=preview_vector_store_fn, inputs=available_vector_stores,
                               outputs=[vector_store_size, vector_store_doc_count,
                                        vector_store_doc_list, file_counter_display])
        check_vs_image.submit(fn=preview_store3, outputs=[preview_vector_store3]) \
                        .then(fn=find_similar_images, inputs=[available_vector_stores, check_vs_image], outputs=[similar_img])
        ###################################################################################
        ####################### Creating New Vector Store Gradio ##########################
        ###################################################################################
        type_of_upload.change(fn=select_uploader, inputs=type_of_upload,
                                  outputs=[multiple_file_upload, 
                                           folder_upload, 
                                           delete_vs,
                                           preview_docs_row1,
                                           not_previewed_row,
                                           preview_docs_row2,
                                           preview_docs_row3,
                                           delete_vs_row1,
                                           upload_complete_create])
        delete_uploaded_docs.click(lambda: None, None, upload_pdf) \
                            .then(lambda: None, None, upload_pdf_folder) \
                            .then(fn=delete_all_temp_files) \
                            .then(fn=delete_gallery, outputs=pdf_preview) \
                            .then(fn=opposite_of_upload_part2,
                                  outputs=[preview_docs_row1,
                                           not_previewed_row,
                                           preview_docs_row2,
                                           preview_docs_row3, 
                                           delete_vs_row1])
        confirm_upload_delete.click(fn=upload_part2,
                                    inputs=[type_of_upload],
                                    outputs=[preview_docs_row1,
                                             not_previewed_row,
                                             preview_docs_row2,
                                             preview_docs_row3,
                                             delete_vs_row1]) \
                            .then(fn=confirm_upload_delete_fn, 
                                  inputs=[upload_pdf, upload_pdf_folder, type_of_upload, vector_stores_to_delete],
                                  outputs=[pdf_preview, double_confirm_upload,
                                           not_previewed, delete_message])
        create_vector_store.click(fn=upload_part4, outputs=upload_complete_create) \
                            .then(fn=create_vector_store_fn,
                                  inputs=[new_vector_store_name, double_confirm_upload],
                                  outputs=[upload_complete_message_create])
        refresh_vector_store_to_delete.click(fn=update_vector_store_list, outputs=vector_stores_to_delete)

        ###################################################################################
        ############################## Access Rights Demo #################################
        ###################################################################################
        

        refresh_vector_store_ar.click(fn=update_vector_store_list, outputs=available_vector_stores_ar)
        login_btn_ar.click(fn=login_ar_fn, inputs=[login_username, login_password]) \
                    .then(fn=login_ar_gui, outputs=[login_ar, logout_ar, choose_vs_ar, confirm_vs_ar])
        logout_btn_ar.click(fn=logout_ar_gui, outputs=[login_ar, logout_ar, choose_vs_ar, confirm_vs_ar,
                                                       view_vector_store_ar1, view_vector_store_df_ar1, 
                                                       view_vector_store_ar3])
        view_vector_store_ar.click(fn=view_store_ar1, outputs=[view_vector_store_ar1, view_vector_store_df_ar1])\
                         .then(fn=preview_vector_store_ar_fn, inputs=available_vector_stores_ar,
                               outputs=[vector_store_size_ar, vector_store_doc_count_ar,
                                        vector_store_doc_list_ar, file_counter_display_ar])
        vector_store_doc_list_ar.change(fn=show_doc_df_fn, inputs=[available_vector_stores_ar, vector_store_doc_list_ar], 
                                        outputs=[ar_df])
        

    return multi_modal_interface


if __name__ == "__main__":
    demo = multi_modal_demo()
    demo.queue(
        max_size=10,
    ).launch(server_name='0.0.0.0', share=False)