import React, { useState } from 'react';
import SearchBar from './components/SearchBar.jsx';
import ResultSection from './components/ResultSection.jsx';
import GraphView from './components/GraphView.jsx';
import './App.css';

function App() {
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isSearching, setIsSearching] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    const handleSearch = async (query) => {
        if (!query.trim()) return;

        setIsSearching(true);
        setLoading(true);
        setError(null);
        setSearchQuery(query);

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';
            const response = await fetch(`${apiUrl}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            console.error('Search error:', err);
            setError('검색 중 오류가 발생했습니다. 다시 시도해주세요.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            <div className="app-content">
                <header className={`app-header ${isSearching ? 'searching' : ''}`}>
                    <h1>Maritime GraphRAG</h1>
                    <p>해양안전심판원 재결서 139건의 지식그래프 기반 사고 분석 Q&A</p>
                </header>

                <div className={isSearching ? 'searching' : ''}>
                    <SearchBar onSearch={handleSearch} loading={loading} />
                </div>

                {error && (
                    <div className="error-message">
                        <span>⚠️</span>
                        <p>{error}</p>
                    </div>
                )}

                {loading && (
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <p>검색 중입니다...</p>
                    </div>
                )}

                {results && !loading && (
                    <div className="results-container">
                        <div className="search-query-header">
                            <h2>"{searchQuery}"에 대한 검색 결과</h2>
                        </div>
                        {results.sections.map((section, idx) => (
                            <ResultSection
                                key={idx}
                                section={section}
                                sources={results.sources}
                            />
                        ))}
                        <GraphView graph={results.graph} />
                    </div>
                )}

                {!results && !loading && !error && (
                    <div className="welcome-message">
                        <h2>해양사고 재결 지식그래프</h2>
                        <p>실제 재결서에서 추출한 사고-원인-선박-처분 그래프를 근거로 답합니다.</p>
                        <div className="example-queries">
                            <p>예시 질문:</p>
                            <ul>
                                <li>충돌 사고에서 가장 흔한 원인 카테고리는?</li>
                                <li>경계 소홀이 원인으로 판시된 사고와 선박을 알려줘</li>
                                <li>어선이 관련된 사고의 원인 분포는 어떻게 되나?</li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
