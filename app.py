from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

# Load environment variables
load_dotenv()

# Configuration
URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "12345678"))
INDEX_NAME = "content_vector_index"

app = FastAPI(title="Maritime GraphRAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
driver = None
rag = None

# Request/Response Models
class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    id: int
    shortName: str
    title: str
    category: str
    date: str
    url: str
    summary: str
    icon: str

class Section(BaseModel):
    title: str
    content: str
    sourceIds: List[int]

class QueryResponse(BaseModel):
    sections: List[Section]
    sources: List[Source]

def get_schema(driver):
    """Neo4j 데이터베이스의 스키마 정보를 가져옵니다"""
    with driver.session() as session:
        node_info = session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName, propertyTypes
            RETURN nodeType, collect(propertyName) as properties
        """).data()

        patterns = session.run("""
            MATCH (n)-[r]->(m)
            RETURN DISTINCT labels(n)[0] as source, type(r) as relationship, labels(m)[0] as target
            LIMIT 20
        """).data()

        schema_text = "=== Neo4j Schema ===\n"
        schema_text += "\n노드 타입:\n"
        for node in node_info:
            schema_text += f"- {node['nodeType']}: {node['properties']}\n"

        schema_text += "\n관계 패턴:\n"
        for pattern in patterns:
            schema_text += f"- ({pattern['source']})-[:{pattern['relationship']}]->({pattern['target']})\n"

        return schema_text

def initialize_graphrag():
    """GraphRAG 시스템 초기화"""
    global driver, rag
    
    try:
        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("✓ Neo4j 연결 성공")
    except Exception as e:
        print(f"✗ Neo4j 연결 실패: {e}")
        return False

    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={"temperature": 0}
    )
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 벡터 임베딩 생성 (없는 경우)
    print("벡터 임베딩 확인 중...")
    from neo4j_graphrag.indexes import create_vector_index
    
    with driver.session() as session:
        # 임베딩 없는 Content 노드 확인
        result = session.run("MATCH (c:Content) WHERE c.embedding IS NULL RETURN elementId(c) AS id, c.chunk AS text")
        records = result.data()
        
        if records:
            print(f"  → {len(records)}개 청크에 임베딩 생성 중...")
            for i, record in enumerate(records):
                node_id = record["id"]
                text = record["text"]
                try:
                    vector = embedder.embed_query(text)
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    
                    session.run("""
                        MATCH (c) WHERE elementId(c) = $id
                        SET c.embedding = $embedding
                        """, {"id": node_id, "embedding": vector})
                    
                    if (i+1) % 10 == 0:
                        print(f"  → 처리됨: {i+1}/{len(records)}")
                except Exception as e:
                    print(f"  ✗ 청크 {node_id} 임베딩 오류: {e}")
            print("✓ 임베딩 생성 완료")
        else:
            print("✓ 모든 청크에 임베딩 존재")
    
    # 벡터 인덱스 생성
    try:
        create_vector_index(
            driver,
            INDEX_NAME,
            label="Content",
            embedding_property="embedding",
            dimensions=1536,
            similarity_fn="cosine",
        )
        print("✓ 벡터 인덱스 생성/확인 완료")
    except Exception as e:
        print(f"  ℹ 인덱스 정보: {e}")

    # Vector Retriever (결과 개수 증가)
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder
    )
    
    # VectorCypher Retriever
    retrieval_query = """
    WITH node AS content, score
    MATCH (content)<-[:HAS_CHUNK]-(article:Article)
    OPTIONAL MATCH (article)-[:BELONGS_TO]->(category:Category)
    OPTIONAL MATCH (article)-[:PUBLISHED_BY]->(media:Media)
    OPTIONAL MATCH (article)-[:MENTIONS]->(entity)
    OPTIONAL MATCH (entity)-[rel:OPERATES|CALLS_AT|APPLIES_TO|INVOLVED_IN|OCCURRED_AT]-(neighbor)
    WITH content, score, article, category, media,
        collect(DISTINCT coalesce(entity.name, entity.incident_id)) AS mentioned_entities,
        collect(DISTINCT CASE WHEN neighbor IS NULL THEN NULL ELSE
            coalesce(entity.name, entity.incident_id) + ' -' + type(rel) + '- ' +
            coalesce(neighbor.name, neighbor.incident_id) END) AS entity_facts
    RETURN
        content.content_id AS content_id,
        content.chunk AS chunk,
        article.article_id AS article_id,
        article.title AS article_title,
        article.url AS article_url,
        article.published_date AS article_date,
        category.name AS category_name,
        media.name AS media_name,
        score AS similarity_score,
        [m IN mentioned_entities WHERE m IS NOT NULL] AS mentioned_entities,
        [f IN entity_facts WHERE f IS NOT NULL] AS entity_facts
    """
    
    vector_cypher_retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        retrieval_query=retrieval_query,
        embedder=embedder
    )

    # Text2Cypher Retriever — curated schema (auto-extracted schema produced
    # malformed Cypher and empty retrievals; this text is what the benchmark validated)
    neo4j_schema = """
Nodes:
  Article {article_id, title, url, published_date}
  Content {content_id, chunk}
  Media {name}, Category {name}
  Vessel {name, type}   # type: 컨테이너선|유조선|LNG운반선|벌크선
  Company {name}, Port {name}
  Regulation {name, applies_to_type, description}
  Incident {incident_id, description}
Naming rules:
  - 선박 이름 뒤의 '호'는 저장된 name에 포함되지 않는다 (질문의 '대양스피릿호' -> Vessel name '대양스피릿')
  - RETURN 절에는 답이 되는 값과 함께 그 주체의 이름(v.name 등)도 반환하라
Relationships:
  (Article)-[:HAS_CHUNK]->(Content)
  (Article)-[:PUBLISHED_BY]->(Media)
  (Article)-[:BELONGS_TO]->(Category)
  (Article)-[:MENTIONS]->(Vessel|Company|Port|Regulation|Incident)
  (Company)-[:OPERATES]->(Vessel)
  (Vessel)-[:CALLS_AT]->(Port)
  (Regulation)-[:APPLIES_TO]->(Vessel)
  (Vessel)-[:INVOLVED_IN]->(Incident)
  (Incident)-[:OCCURRED_AT]->(Port)
"""
    
    examples = [
        """
        USER INPUT: 한서파이오니어호를 운영하는 선사는 어디인가요?
        CYPHER QUERY:
        MATCH (c:Company)-[:OPERATES]->(v:Vessel {name: "한서파이오니어"})
        RETURN c.name
        """,
        """
        USER INPUT: 부산항에 기항하는 컨테이너선을 운영하는 선사를 알려주세요
        CYPHER QUERY:
        MATCH (c:Company)-[:OPERATES]->(v:Vessel {type: "컨테이너선"})-[:CALLS_AT]->(:Port {name: "부산항"})
        RETURN DISTINCT c.name, v.name
        """,
        """
        USER INPUT: 울산항에서 발생한 사고와 관련 기사를 알려주세요
        CYPHER QUERY:
        MATCH (i:Incident)-[:OCCURRED_AT]->(:Port {name: "울산항"})
        OPTIONAL MATCH (a:Article)-[:MENTIONS]->(i)
        RETURN i.incident_id, i.description, a.title, a.url
        """,
        """
        USER INPUT: 규제/환경 분야 기사 개수를 알려주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category)
        RETURN c.name as category, count(a) as article_count
        ORDER BY article_count DESC
        """,
    ]
    
    text2cypher_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    # Tools Setup
    vector_tool = vector_retriever.convert_to_tool(
        name="vector_retriever",
        description="벡터 기반 검색. 해양 뉴스 본문 내용(사건 경위, 정책 설명 등)을 의미 기반으로 찾을 때 사용합니다."
    )
    vector_cypher_tool = vector_cypher_retriever.convert_to_tool(
        name="vectorcypher_retriever",
        description="벡터 검색 결과 기사에 언급된 엔티티(선박/선사/항만/규제/사고)와 그 관계(운영, 기항, 적용, 사고)를 함께 반환합니다. 관계를 따라가야 하는 질문에 사용합니다."
    )
    text2cypher_tool = text2cypher_retriever.convert_to_tool(
        name="text2cypher_retriever",
        description="자연어를 Cypher로 변환해 그래프를 직접 질의합니다. 선사-선박-항만-규제-사고를 잇는 멀티홉 질문이나 개수 집계에 사용합니다."
    )

    tools_retriever = ToolsRetriever(
        driver=driver,
        llm=llm,
        tools=[vector_tool, vector_cypher_tool, text2cypher_tool],
    )

    # GraphRAG Setup
    prompt_template = RagTemplate(
        template="""당신은 해양 산업(해운·항만·규제) 정보를 제공하는 전문 어시스턴트입니다.

질문: {query_text}

검색된 문서 정보:
{context}

지침:
1. 제공된 검색 결과에 근거해 질문에 직접 답한 뒤, 근거 기사를 함께 제시하세요.
2. 답의 근거가 되는 엔티티(선사/선박/항만/규제)를 명시하세요.
3. 각 근거 기사마다 제목, URL, 발행일, 카테고리, 언론사, 요약(1-2문장)을 포함하세요.
4. 검색 결과에 없는 내용은 절대 추측하거나 지어내지 마세요. 출처(제목/URL/날짜)를 창작하는 것은 금지됩니다.
5. 검색 결과가 비어 있거나 질문과 무관하면 content에 "관련 정보를 찾을 수 없습니다"라고 쓰고 sources는 빈 배열로 두세요.
6. 다음 JSON 형식으로만 답변하세요 (마크다운 코드 블록 없이):

{{
  "sections": [
    {{
      "title": "검색 결과",
      "content": "",
      "sources": [
        {{
          "title": "기사 제목",
          "url": "기사 URL",
          "date": "발행일",
          "category": "카테고리",
          "media": "언론사",
          "summary": "기사 요약 (2-3문장)"
        }}
      ]
    }}
  ]
}}

답변:""",
        expected_inputs=["context", "query_text"]
    )

    rag = GraphRAG(
        llm=llm,
        retriever=tools_retriever,
        prompt_template=prompt_template
    )
    
    print("✓ GraphRAG 시스템 초기화 완료")
    return True

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 GraphRAG 초기화"""
    success = initialize_graphrag()
    if not success:
        print("⚠ Warning: GraphRAG 초기화 실패")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 연결 해제"""
    global driver
    if driver:
        driver.close()
        print("✓ Neo4j 연결 종료")

@app.get("/")
async def root():
    return {"message": "RAG Search API is running"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    global driver
    try:
        if driver:
            driver.verify_connectivity()
            return {"status": "healthy", "database": "connected"}
        return {"status": "unhealthy", "database": "not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """검색 쿼리 처리"""
    global rag
    
    if not rag:
        raise HTTPException(status_code=503, detail="GraphRAG system not initialized")
    
    try:
        # GraphRAG 검색 실행
        result = rag.search(query_text=request.query, return_context=True)
        
        # 응답에서 마크다운 코드 블록 제거
        answer_text = result.answer.strip()
        
        # ```json ... ``` 형식 제거
        if answer_text.startswith('```'):
            # 첫 줄 제거 (```json)
            lines = answer_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # 마지막 줄 제거 (```)
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            answer_text = '\n'.join(lines).strip()
        
        # JSON 파싱
        import json
        try:
            parsed_result = json.loads(answer_text)
        except Exception as e:
            print(f"JSON 파싱 실패: {e}")
            print(f"응답 내용: {answer_text[:500]}")
            # JSON 파싱 실패 시 기본 형태로 반환
            parsed_result = {
                "sections": [{
                    "title": "검색 결과",
                    "content": result.answer,
                    "sources": []
                }]
            }
        
        # 출처 정보 변환
        sources = []
        source_id = 1
        
        for section in parsed_result.get("sections", []):
            source_ids = []
            for source_data in section.get("sources", []):
                sources.append({
                    "id": source_id,
                    "shortName": source_data.get("media", "unknown"),
                    "title": source_data.get("title", ""),
                    "category": source_data.get("category", "기타"),
                    "date": source_data.get("date", ""),
                    "url": source_data.get("url", ""),
                    "summary": source_data.get("summary", ""),
                    "icon": get_icon_for_category(source_data.get("category", ""))
                })
                source_ids.append(source_id)
                source_id += 1
            
            section["sourceIds"] = source_ids
            # sources 키 제거 (프론트엔드에서 sourceIds 사용)
            section.pop("sources", None)
        
        return {
            "sections": parsed_result.get("sections", []),
            "sources": sources
        }
        
    except Exception as e:
        print(f"검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def get_icon_for_category(category: str) -> str:
    """카테고리에 따른 아이콘 반환"""
    icons = {
        "정치": "🏛️",
        "경제": "💼",
        "사회": "👥",
        "생활/문화": "🎭",
        "IT/과학": "💻",
        "세계": "🌍",
    }
    return icons.get(category, "📰")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)