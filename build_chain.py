import io
import re
import base64
import streamlit as st
import certifi

from get_external_ip import get_external_ip

st.title('🏥 FellowsGPT')

## For use if need to whitelist certain IP addresses
# external_ip = get_external_ip()
# st.write("External IP: ", external_ip)

# from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.storage import InMemoryStore

# Get secret keys from environment variables
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
MONGODB_CONN_STRING=st.secrets["MONGODB_CONN_STRING"]
DB_NAME=st.secrets["DB_NAME"]
VECTOR_COLLECTION_NAME=st.secrets["VECTOR_COLLECTION_NAME"]
KEYVALUE_COLLECTION_NAME=st.secrets["KEYVALUE_COLLECTION_NAME"]
VECTOR_INDEX_NAME=st.secrets["VECTOR_INDEX_NAME"]

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\xFF\xD8\xFF": "jpeg",
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
    

def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a healthcare consultant tasking with providing training to newly hired early career healthcare consultant trainees called 'Fellows.' .\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide advice related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1920, 1080))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", openai_api_key=OPENAI_API_KEY)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }   
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

id_key = "doc_id"
store = InMemoryStore()

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize MongoDB clients
client = MongoClient(MONGODB_CONN_STRING, tlsCAFile=certifi.where())
mongoDB = client[DB_NAME]
vector_collection = client[DB_NAME][VECTOR_COLLECTION_NAME]
kv_collection = client[DB_NAME][KEYVALUE_COLLECTION_NAME]

print ("MongoDB connected: ", mongoDB)

vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_CONN_STRING,
    DB_NAME + "." + VECTOR_COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()), # embeddings, # OpenAIEmbeddings(disallowed_special=()),
    index_name=VECTOR_INDEX_NAME
)

# kv_collection.find_one()
mongoDict = {}
for docObject in kv_collection.find():
    mongoDict.update(docObject)

mongoObjArr = []
# Flag to skip the first entry
skip_first = True
for key, value in mongoDict.items():
    if skip_first:
        skip_first = False
        continue  # Skip the first iteration
    mongoObjArr.append((key, value))

store.mset(mongoObjArr)

# Create the multi-vector retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Create RAG chain
chain_multimodal_rag = multi_modal_rag_chain(retriever)

print(chain_multimodal_rag.invoke("How can I convince an IT stakeholder to invest resources into my initiative to build regimen margin projections into Allscripts?"))