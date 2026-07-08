import React from 'react';
import SourceTooltip from './SourceTooltip.jsx';
import './ResultSection.css';

function ResultSection({ section, sources }) {
    // sourceIds로 실제 source 객체들을 찾기
    const sectionSources = section.sourceIds
        ? section.sourceIds.map(id => sources.find(s => s.id === id)).filter(Boolean)
        : [];

    return (
        <div className="result-section">
            <h2 className="section-title">{section.title}</h2>
            <div className="section-content">
                {section.content && (
                    <p className="answer-text">{section.content}</p>
                )}
                {sectionSources.length > 0 ? (
                    <ul className="article-list">
                        {sectionSources.map((source, idx) => (
                            <li key={idx} className="article-item">
                                <span className="article-summary">{source.summary}</span>
                                <SourceTooltip source={source} />
                            </li>
                        ))}
                    </ul>
                ) : (
                    !section.content && <p className="no-sources">관련 기사가 없습니다.</p>
                )}
            </div>
        </div>
    );
}

export default ResultSection;
