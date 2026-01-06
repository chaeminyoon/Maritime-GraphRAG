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
                    <h1>AI 뉴스 검색 시스템</h1>
                    <p>Neo4j GraphRAG 기반 지능형 뉴스 검색</p>
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
                        <h2>환영합니다! 👋</h2>
                        <p>궁금한 내용을 검색해보세요.</p>
                        <div className="example-queries">
                            <p>예시 질문:</p>
                            <ul>
                                <li>경제 분야 최신 뉴스</li>
                                <li>IT/과학 관련 소식</li>
                                <li>정치 동향은?</li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
