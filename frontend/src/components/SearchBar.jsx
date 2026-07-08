import React, { useState } from 'react';
import './SearchBar.css';

function SearchBar({ onSearch, loading }) {
    const [query, setQuery] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim() && !loading) {
            onSearch(query);
        }
    };

    return (
        <form className="search-bar" onSubmit={handleSubmit}>
            <div className="search-input-container">
                <input
                    type="text"
                    className="search-input"
                    placeholder="해양 산업 질문을 입력하세요... (예: 사고 이력이 있는 선박의 운영 선사는?)"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={loading}
                />
                <button
                    type="submit"
                    className="search-button"
                    disabled={loading || !query.trim()}
                >
                    {loading ? '검색 중...' : '🔍 검색'}
                </button>
            </div>
        </form>
    );
}

export default SearchBar;
