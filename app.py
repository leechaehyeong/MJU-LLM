#streaming app 만들기

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda

def run_chains(question, lang):
    # Chain1
    llm1 = ChatOllama(model="qwen2:1.5b", temperature=0, base_url="http://127.0.0.1:11434/") #http://127.0.0.1:11434
    prompt1 = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("당신은 인공지능 전문가입니다. 사용자의 질문에 답해주세요."),
                HumanMessagePromptTemplate.from_template("{input}")
                                                ])
    output_parser = StrOutputParser()
    chain1 = prompt1 | llm1 | output_parser


    llm2 = ChatOllama(model="qwen2:1.5b", temperature=0, base_url="http://127.0.0.1:11434/") #http://127.0.0.1:11434
    prompt2 = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template("""
                                                # Instruction
                                                당신은 번역가입니다. 다음에 주어진 문장을 {lang}로 번역해주세요.                                                                                      
                                                """),
                                                HumanMessagePromptTemplate.from_template("""
                                                # Text
                                                {text}                                          
                                                # Result""")])
    output_parser = StrOutputParser()
    chain2 = RunnableMap({"text": RunnableLambda(lambda x: chain1), "lang": RunnableLambda(lambda x: lang)}) | prompt2 | llm2 | output_parser


    result = chain2.invoke({"input": question})
    return result

#스트림잇 만들기

st.title("Ollama 정보 번역 app")

# User inputs
question = st.text_input("질문:", "여기에 질문을 입력하세요")
lang = st.selectbox("언어를 선택해주세요:", ["English", "한국어", "프랑스어", "Español"])

if st.button("실행"):
    if question and lang:
        with st.spinner("잠시만 기다려 주세요..."):
            result = run_chains(question, lang)
            st.success("완료!")
            st.write("번역결과 : ", result)