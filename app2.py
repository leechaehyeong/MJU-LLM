#streaming app 만들기

import streamlit as st
import json
import requests
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings



# 인증키는 개인 디코딩키 사용
def get_air_quality_data(sido):
    url = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'
    params ={'serviceKey' : '5G6VhMJvAkXVled6gzJvIYhsXaVONUMxx4N1AKkOemtEd1yCfRN0dp1oLmiRmHt7lJEzcvKBpQCeOTS5+RKzQg==', 'returnType' : 'json', 'numOfRows' : '100', 'pageNo' : '1', 'sidoName' : sido, 'ver' : '1.0' }    
    response = requests.get(url, params=params)
    content = response.content.decode('utf-8')
    data = json.loads(content)
    return data


def parse_air_quality_data(data):
    items = data['response']['body']['items']
    air_quality_info = []
    for item in items:
        info = {
            '측정소명': item.get('stationName'),
            '날짜': item.get('dataTime'),
            '미세먼지농도': item.get('pm10Value'),
            '초미세먼지농도': item.get('pm25Value'),
            'so2농도': item.get('so2Value'),
            'co농도': item.get('coValue'),
            'o3농도': item.get('o3Value'),
            'no2농도': item.get('no2Value'),
            '통합대기환경수치': item.get('khaiValue'),
            '통합대기환경지수': item.get('khaiGrade'),
            'pm10등급': item.get('pm10Grade'),
            'pm25등급': item.get('pm25Grade')
        }
        air_quality_info.append(info)
    return air_quality_info

st.title("대기질 정보 제공 챗봇 app")

# User inputs
sido = st.text_input("도시의 이름을 입력하세요. : ", "서울")
query = st.text_input("궁금한 지역을 입력하세요. : ", "서대문구")

if st.button("실행"):
    if sido and query:
        with st.spinner("잠시만 기다려 주세요..."):
            result = get_air_quality_data(sido)
            air_quality_info = parse_air_quality_data(result)
            text_splitter = RecursiveCharacterTextSplitter( separators = "',", )
            documents = [Document(page_content=", ".join([f"{key}: {str(info[key])}" for key in ['측정소명', '날짜', '미세먼지농도', '초미세먼지농도', '통합대기환경수치']])) for info in air_quality_info]
            docs = text_splitter.split_documents(documents)
            embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
            db = FAISS.from_documents(docs, embedding_function)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={'k':5, 'fetch_k': 100})
            template = """
                당신은 대기질을 안내하는 챗봇입니다. 사용자에게 가능한 많은 정보를 친절하게 제공하십시오.
                미세먼지농도와 초미세먼지 농도를 다음 기준을 통해 좋음, 보통, 나쁨, 매우 나쁨으로 나누어서 각각 표시해줘.

                #기준

                미세먼지 농도
                좋음: 0~30
                보통: 31~80
                나쁨: 81~100
                매우 나쁨: 151 이상

                초미세먼지 농도
                좋음: 0~15
                보통: 16~35
                나쁨: 36~75
                매우 나쁨: 76 이상

                Answer the question as based only on the following context:
                {context}


                Question: {question}
                """
            prompt = ChatPromptTemplate.from_template(template)
            from langchain_community.vectorstores import FAISS
            llm = ChatOllama(model="qwen2:1.5b", temperature=0, base_url="http://127.0.0.1:11434/")
            chain = RunnableMap({
                "context": lambda x: retriever.get_relevant_documents(x['question']),
                "question": lambda x: x['question']
            }) | prompt | llm
            response = chain.invoke({'question': query})
            st.markdown(response.content)