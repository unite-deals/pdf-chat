import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
key= st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=key)


hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_github_link_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 




def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, search the pdf first , then add your own context , make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context",try your own knowledge , try to display chart or any graphical presentation & mathematical equation also. don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(google_api_key = key, model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key = key, model = "models/embedding-001") 
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])



# Application

st.header("PDF Insurance Wizard 🧙🏻‍♀️")
st.markdown("""
This app retrieves the information from the uploaded PDFs and answers your questions related to the PDFs using AI.
""")
with st.expander("Help"):
    st.subheader("Upload PDFs:")
    st.write('''In the sidebar browse or drag and drop the PDFsyou want to analyze.
        You can upload multiple PDFs simultaneously.''')
    st.subheader("Process PDFs:")
    st.write('''In the sidebar browse or drag and drop the PDFsyou want to analyze.
        You can upload multiple PDFs simultaneously.''')
    st.subheader("Process PDFs:")
    st.write('''After uploading, hit the "Process" button to initiate the processing of your PDFs. Once the processing is complete, you'll see the success sign with done.''')
    st.subheader("Ask Your Questions:")
    st.write('''Now type your questions and let PDFQueryWizard generate insightful responses.''')

user_question = st.text_input("Ask any Question from the PDF Files")
ask = st.button("Ask")
if ask or user_question:
    user_input(user_question)



with st.sidebar:
    st.image("images/download.png")
    st.title("Your PDFs goes here!")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")


