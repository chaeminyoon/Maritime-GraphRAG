import React, { useState } from 'react';
import './SourceTooltip.css';

function SourceTooltip({ source }) {
    const [showTooltip, setShowTooltip] = useState(false);

    return (
        <span
            className="source-badge-container"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
        >
            <span className="source-badge">
                {source.shortName}
            </span>

            {showTooltip && (
                <div className="source-tooltip">
                    <div className="tooltip-header">
                        <div className="tooltip-title-group">
                            <strong className="tooltip-source">{source.shortName}</strong>
                            <span className="tooltip-category">{source.category}</span>
                        </div>
                    </div>

                    <h4 className="tooltip-title">{source.title}</h4>

                    <p className="tooltip-summary">{source.summary}</p>

                    <div className="tooltip-footer">
                        <span className="tooltip-date">{source.date}</span>
                        <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="tooltip-link"
                            onClick={(e) => e.stopPropagation()}
                        >
                            원문 보기
                        </a>
                    </div>
                </div>
            )}
        </span>
    );
}

export default SourceTooltip;
