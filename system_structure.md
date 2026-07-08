# Maritime GraphRAG 시스템 구조

이 문서는 해양 지식그래프 RAG 시스템(FastAPI 백엔드 + React 프론트엔드)의 구조를 시각화하여 설명합니다.

## 1. 전체 아키텍처 (Architecture)

```mermaid
graph TD
    subgraph "Frontend (React + Vite)"
        UI[User Interface]
        SB[SearchBar]
        RS[ResultSection]
        ST[SourceTooltip]
        UI --> SB
        UI --> RS
        RS --> ST
    end

    subgraph "Backend (FastAPI)"
        API[FastAPI Server]
        RAG[GraphRAG Manager]
        LLM[OpenAI GPT-4o]
        
        API --> RAG
        RAG --> LLM
        
        subgraph "Retrievers"
            VR[Vector Retriever]
            VCR[VectorCypher Retriever]
            T2C[Text2Cypher Retriever]
        end
        
        RAG --> VR
        RAG --> VCR
        RAG --> T2C
    end

    subgraph "Database (Neo4j)"
        DB[(Neo4j Graph Database)]
        VEC[Vector Index]
    end

    API -- "HTTP POST /search" --- UI
    VR --- DB
    VCR --- DB
    T2C --- DB
    DB --- VEC
```

## 2. 뉴스 데이터 모델 (Neo4j Schema)

```mermaid
erDiagram
    Article ||--o{ Content : HAS_CHUNK
    Article ||--|| Media : PUBLISHED
    Article ||--|| Category : BELONGS_TO
    
    Article {
        string article_id
        string title
        string url
        string published_date
    }
    
    Content {
        string content_id
        string chunk
        float[] embedding
    }
    
    Media {
        string name
    }
    
    Category {
        string name
    }
```

## 3. 검색 데이터 흐름 (Search Data Flow)

```mermaid
sequenceDiagram
    participant User
    participant React as Frontend
    participant FastAPI as Backend
    participant RAG as GraphRAG
    participant Neo4j as Database
    participant OpenAI as LLM (GPT-4o)

    User->>React: 검색어 입력 (예: "경제 뉴스")
    React->>FastAPI: POST /search {query}
    FastAPI->>RAG: search(query)
    
    par Retrieval Phase
        RAG->>Neo4j: 벡터 검색 (Vector)
        RAG->>Neo4j: 관계 기반 검색 (Cypher)
        RAG->>Neo4j: 자연어 쿼리 변환 (Text2Cypher)
    end
    
    Neo4j-->>RAG: 검색된 문서군 (Context)
    
    RAG->>OpenAI: Prompt + Context 전송
    OpenAI-->>RAG: 구조화된 JSON 응답
    
    RAG-->>FastAPI: 정제된 결과 데이터
    FastAPI-->>React: Response {sections, sources}
    React-->>User: 애니메이션과 함께 결과 표시
```

## 4. 디렉토리 구조 (Directory Structure)

```text
RAG_SYS/
├── app.py                    # FastAPI 백엔드 엔트리포인트
├── README.md                 # 프로젝트 개요
├── system_structure.md       # (현재 파일) 시스템 구조 시각화
├── .env                      # API Key 및 DB 접속 정보
└── frontend/                 # React 프론트엔드 프로젝트
    ├── index.html            # 메인 HTML
    ├── vite.config.js        # Vite 설정
    └── src/
        ├── index.jsx         # React 진입점
        ├── App.jsx           # 메인 애플리케이션 컴포넌트
        ├── App.css           # 전체 스타일 및 애니메이션
        └── components/       # UI 컴포넌트
            ├── SearchBar.jsx # 검색창 컴포넌트
            ├── ResultSection.jsx # 결과 섹션 컴포넌트
            └── SourceTooltip.jsx # 출처 툴팁 컴포넌트
```
