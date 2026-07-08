import React, { useState } from 'react';
import SearchBar from './components/SearchBar.jsx';
import ResultSection from './components/ResultSection.jsx';
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
                    <p>Neo4j 지식그래프 기반 해양 산업(해운·항만·규제) 질의응답</p>
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
                    </div>
                )}

                {!results && !loading && !error && (
                    <div className="welcome-message">
                        <h2>해양 지식그래프 검색</h2>
                        <p>선사·선박·항만·규제·사고를 잇는 관계 질문에 답합니다.</p>
                        <div className="example-queries">
                            <p>예시 질문:</p>
                            <ul>
                                <li>부산항에 기항하는 컨테이너선을 운영하는 선사는?</li>
                                <li>울산항에서 사고를 낸 선박에 적용되는 환경 규제는?</li>
                                <li>한서해운 선대가 기항하는 항만을 모두 알려줘</li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
