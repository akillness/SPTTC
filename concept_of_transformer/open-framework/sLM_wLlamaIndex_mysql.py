import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, BLOB, text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

# MySQL 연결 설정 (root 계정 사용)
DATABASE_URL = "mysql+mysqlconnector://root:0000@localhost/rag_db"
engine = create_engine(DATABASE_URL)

# 데이터베이스가 없는 경우 생성
def create_database():
    temp_engine = create_engine("mysql+mysqlconnector://root:0000@localhost/")
    with temp_engine.connect() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS rag_db"))
        conn.commit()

create_database()

Base = declarative_base()

# MySQL 테이블 정의
class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    embedding = Column(BLOB)  # 벡터 임베딩 저장
    doc_metadata = Column(Text)   # 메타데이터 JSON 저장
    created_at = Column(DateTime, default=datetime.utcnow)  # 생성 시간 추가

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# 임베딩 모델 초기화
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

class MySQLVectorStore:
    def __init__(self):
        self.embed_model = Settings.embed_model
        self.batch_size = 32
    
    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성 (캐싱 적용)"""
        embedding = np.array(self.embed_model.get_text_embedding(text))
        return embedding.astype('float32')
    
    def _batch_process(self, items: List, batch_size: int):
        """배치 처리 유틸리티"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def add_documents(self, documents: List[Document]):
        session = Session()
        try:
            # 기존 문서 삭제
            session.query(DocumentModel).delete()
            session.commit()
            
            # 배치 처리로 문서 추가
            for batch in self._batch_process(documents, self.batch_size):
                docs_to_add = []
                
                for doc in batch:
                    embedding = self._get_embedding(doc.text)
                    embedding_blob = json.dumps(embedding.tolist()).encode('utf-8')
                    
                    new_doc = DocumentModel(
                        text=doc.text,
                        embedding=embedding_blob,
                        doc_metadata=json.dumps(doc.metadata if hasattr(doc, 'metadata') else {})
                    )
                    docs_to_add.append(new_doc)
                
                # 데이터베이스에 배치 저장
                session.bulk_save_objects(docs_to_add)
                session.commit()
            
            print(f"✅ {len(documents)} 개의 문서가 성공적으로 저장되었습니다.")
        finally:
            session.close()
    
    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[float, DocumentModel]]:
        session = Session()
        try:
            # 쿼리 임베딩 생성
            query_embedding = self._get_embedding(query_text).reshape(1, -1)
            
            # 모든 문서 가져오기
            all_docs = session.query(DocumentModel).all()
            if not all_docs:
                return []
            
            # 문서 임베딩 행렬 생성
            doc_embeddings = []
            for doc in all_docs:
                embedding = np.array(json.loads(doc.embedding.decode('utf-8')))
                doc_embeddings.append(embedding)
            doc_embeddings = np.array(doc_embeddings)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # 결과 정렬 및 중복 제거
            results = []
            seen_texts = set()
            
            # 유사도 기준으로 정렬된 인덱스
            sorted_indices = np.argsort(similarities)[::-1]
            
            for idx in sorted_indices:
                doc = all_docs[idx]
                if doc.text not in seen_texts:
                    seen_texts.add(doc.text)
                    results.append((float(similarities[idx]), doc))
                    
                    if len(results) >= top_k:
                        break
            
            return results
        finally:
            session.close()

    def visualize_document_stats(self):
        """문서 통계 시각화"""
        session = Session()
        try:
            # 데이터 조회
            docs = session.query(DocumentModel).all()
            
            # 기본 정보 수집
            doc_lengths = [len(doc.text) for doc in docs]
            created_times = [doc.created_at for doc in docs]
            
            # 1. 문서 길이 분포 시각화
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            sns.histplot(doc_lengths)
            plt.title('문서 길이 분포')
            plt.xlabel('문서 길이 (글자 수)')
            plt.ylabel('문서 수')
            
            # 2. 시간별 문서 추가 추이
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=range(len(created_times)), y=created_times)
            plt.title('문서 추가 시간 추이')
            plt.xlabel('문서 인덱스')
            plt.ylabel('생성 시간')
            
            plt.tight_layout()
            plt.savefig('document_stats.png')
            plt.close()
            
            # 3. 데이터베이스 통계 출력
            total_docs = len(docs)
            avg_length = np.mean(doc_lengths)
            
            print("\n📊 데이터베이스 통계:")
            print(f"- 총 문서 수: {total_docs}")
            print(f"- 평균 문서 길이: {avg_length:.2f} 글자")
            print(f"- 가장 긴 문서: {max(doc_lengths)} 글자")
            print(f"- 가장 짧은 문서: {min(doc_lengths)} 글자")
            
            # 4. 데이터프레임으로 변환하여 표 형태로 출력
            df = pd.DataFrame({
                'ID': [doc.id for doc in docs],
                '텍스트 (일부)': [doc.text[:50] + '...' for doc in docs],
                '길이': doc_lengths,
                '생성 시간': created_times
            })
            
            print("\n📋 문서 목록:")
            print(df.to_string(index=False))
            
        finally:
            session.close()

# 샘플 데이터 생성 및 처리
def create_sample_documents():
    try:
        # PDF 파일에서 문서 로드
        documents = SimpleDirectoryReader(
            input_files=["data/resume.pdf"]
        ).load_data()
        
        if not documents:
            print("⚠️ PDF 파일에서 데이터를 불러오지 못했습니다. 샘플 데이터를 사용합니다.")
            return _create_fallback_documents()
            
        print(f"✅ PDF 파일에서 {len(documents)}개의 문서를 성공적으로 불러왔습니다.")
        return documents
    except Exception as e:
        print(f"⚠️ PDF 파일 로드 중 오류 발생: {e}")
        print("샘플 데이터를 대신 사용합니다.")
        return _create_fallback_documents()


def _create_fallback_documents():
    """PDF 파일 로드 실패 시 사용할 샘플 데이터"""
    sample_texts = [
        "기계 학습은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 기술입니다.",
        "딥러닝은 여러 층의 인공 신경망을 사용하여 데이터를 학습하는 기계 학습의 한 방법입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술을 연구하는 분야입니다.",
        "강화학습은 에이전트가 환경과 상호작용하면서 보상을 최대화하는 방향으로 학습하는 방법입니다.",
        "전이학습은 하나의 작업에서 학습한 지식을 다른 작업에 적용하는 기계학습 기법입니다."
    ]
    
    documents = [Document(text=text) for text in sample_texts]
    return documents

def process_documents():
    # 샘플 데이터 생성
    documents = create_sample_documents()
    
    vector_store = MySQLVectorStore()
    vector_store.add_documents(documents)
    
    # 데이터베이스 통계 시각화
    vector_store.visualize_document_stats()

# RAG 질의 처리
def rag_query(query_text: str) -> List[Tuple[float, DocumentModel]]:
    vector_store = MySQLVectorStore()
    results = vector_store.query(query_text)
    
    print("\n🔍 검색 결과 상위 3개 문서:")
    for i, (score, doc) in enumerate(results, 1):
        print(f"{i}. (유사도: {score:.4f}) {doc.text[:200]}...")
    
    return results

# 테스트 실행
if __name__ == "__main__":
    # 샘플 데이터 초기화
    process_documents()
    
    # 테스트 질의들
    test_queries = [
        "기계 학습의 주요 개념은 무엇인가요?",
        "딥러닝이란 무엇인가요?",
        "자연어 처리는 어떤 분야인가요?",
        "강화학습에 대해 설명해주세요",
        "전이학습이란 무엇인가요?"
    ]
    
    for query in test_queries:
        print(f"\n📝 질의: {query}")
        rag_query(query) 