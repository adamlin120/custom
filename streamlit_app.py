from pathlib import Path

import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAIChat
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from tqdm import tqdm


_template = """根據以下對話及後續問題，將後續問題改寫為獨立的問題。

對話記錄：
{chat_history}
後續問題輸入：{question}
獨立問題："""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
prompt_template = """使用以下上下文片段來回答最後的問題。
你是台灣政府幫忙解決民眾進出口報關問題的幫手
一定要使用台灣繁體中文回答

{context}

問題：{question}
有用的回答："""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def load_embedding():
    EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v1"
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_data():
    dir =  Path('data/Merged')
    # random sample 100 pdf files under dir
    paths = [p for p in dir.glob('**/*.pdf')]
    st.info(f"There are {len(paths)} pdf files in the data folder.")
    # paths = random.sample(paths, NUM_DOCS)
    # st.info(f"For demo purpose, we only use {NUM_DOCS} pdf files.")
    loaders = [PyPDFLoader(str(p)) for p in paths]
    documents = []
    for loader in tqdm(loaders):
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(e)
    return documents


@st.cache_resource()
def load_vectorstore():
    persist_directory = 'db'
    embeddings = load_embedding()
    from pathlib import Path
    if Path(persist_directory).exists():
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    documents = load_data()
    # check if the vectorstore is already built
    return Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

vectorstore = load_vectorstore()

llm = OpenAIChat(model_name="gpt-3.5-turbo-16k")


qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    qa_prompt=QA_PROMPT,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
)


def on_input_change():
    chat_history = []
    for i in range(len(st.session_state['generated'])):
        chat_history.append((st.session_state['past'][i], st.session_state['generated'][i]))

    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)

    query = f"我的文件: {uploaded_file}\n\n{user_input}"

    with st.spinner('Calculating...'):
        result = qa({"question": query, "chat_history": chat_history})
    st.session_state.generated.append(result["answer"])


def on_btn_click():
    del st.session_state.past[1:]
    del st.session_state.generated[1:]


st.session_state.setdefault(
    'past',
    ['我需要您協助我，一定要使用台灣正體繁體中文回答。']
)
st.session_state.setdefault(
    'generated',
    ['我是報關小幫手，請在上方上傳您的文件。']
)
# Page title
# st.set_page_config(page_title='報關小幫手 - 進口貨物分類查詢')
st.title('進口貨物分類查詢')

# Big text area
uploaded_file = st.text_area('請輸入文件內容', height=200, max_chars=1000000)

chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        if i > 0:
            message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i],
            key=f"{i}",
            allow_html=True,
            is_table=False
        )

st.button("重啟對話", on_click=on_btn_click)

with st.container():
    st.text_input("請輸入問題：", on_change=on_input_change, key="user_input", placeholder='我應該要報哪一項？')
