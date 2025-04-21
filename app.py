import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# 🔑 API 키
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

st.title("📊 설문 응답 챗봇")
uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx"])

if uploaded_file:
    # 🔄 파일 로딩
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("데이터 미리보기", df.head())

    # 🧩 텍스트로 변환
    docs = []
    for _, row in df.iterrows():
        row_text = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        docs.append(row_text)

    # 📎 텍스트 나누기
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(docs)

    # ✅ 여기서 문서 수 줄이기 (RateLimit 회피용)
    texts = texts[:10]  # 🔥 처음 10개 문서만 벡터화

    # 🧠 벡터 DB 생성
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # 💬 질문 응답 체인
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    query = st.text_input("무엇이든 물어보세요 (예: '경영진 만족도는 어땠나요?')")

    if query:
        result = qa({"query": query})
        st.subheader("💡 답변")
        st.write(result["result"])

        with st.expander("🔍 관련 데이터 보기"):
            for doc in result["source_documents"]:
                st.markdown(f"- {doc.page_content}")
