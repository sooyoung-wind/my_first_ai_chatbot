__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

load_dotenv()

### title
st.title("ChatPDF")
st.write("---")

### file uploader
uploaded_file = st.file_uploader("PDF 파일을 업로드 해주세요.", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

### 업로드 되면 동작하는 코드 
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)


    ### Loader
    loader = PyPDFLoader('sample.pdf')
    pages = loader.load_and_split()

    ### Split
    text_splitter = RecursiveCharacterTextSplitter(
        ### Set a really small chunk size, just to show.
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)

    ### Embedding
    embeddings_model = OpenAIEmbeddings()

    ### load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)
    
    ### Question
    st.header("PDF에게 질문해보세요")
    question = st.text_input('질문을 입력하세요')
    
    st.button("Reset", type="primary")
    if st.button('질문하기'):
        with st.spinner("Wait for it..."):
            ### db에서 가져오기
            # question = "아내가 먹고 싶어하는 음식은 무엇이야?"
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({'query': question})
            st.write(result["result"])


