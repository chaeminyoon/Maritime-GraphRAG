import React, { useEffect, useRef, useState } from 'react';
import {
    forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide,
} from 'd3-force';
import './GraphView.css';

// node type -> color / display name (marine palette, consistent with docs figures)
const NODE_STYLE = {
    Chunk:         { color: '#94A3B8', name: '재결서 본문' },
    Accident:      { color: '#C4453C', name: '사고' },
    AVessel:       { color: '#2E6F9E', name: '선박' },
    ALocation:     { color: '#0E7490', name: '장소' },
    CauseCategory: { color: '#7C3AED', name: '원인 유형' },
    Court:         { color: '#475569', name: '심판원' },
};
const styleOf = (t) => NODE_STYLE[t] || { color: '#94A3B8', name: t };

function GraphView({ graph }) {
    const [layout, setLayout] = useState(null);
    const [hover, setHover] = useState(null);
    const wrapRef = useRef(null);
    const W = 920, H = 420;

    useEffect(() => {
        if (!graph || !graph.nodes || graph.nodes.length === 0) return;
        const nodes = graph.nodes.map(n => ({ ...n }));
        const links = graph.edges.map(e => ({ ...e }));

        const sim = forceSimulation(nodes)
            .force('link', forceLink(links).id(d => d.id).distance(95).strength(0.6))
            .force('charge', forceManyBody().strength(-260))
            .force('center', forceCenter(W / 2, H / 2))
            .force('collide', forceCollide(34))
            .stop();
        for (let i = 0; i < 250; i += 1) sim.tick();

        nodes.forEach(n => {
            n.x = Math.max(40, Math.min(W - 40, n.x));
            n.y = Math.max(26, Math.min(H - 26, n.y));
        });
        setLayout({ nodes, links });
    }, [graph]);

    if (!graph || !layout) return null;
    const typesUsed = [...new Set(layout.nodes.map(n => n.type))];

    return (
        <div className="graph-view" ref={wrapRef}>
            <div className="graph-header">
                <h3>지식그래프 연결 — 이 답이 어떻게 이어졌는가</h3>
                <div className="graph-legend">
                    {typesUsed.map(t => (
                        <span key={t} className="legend-item">
                            <span className="legend-dot" style={{ background: styleOf(t).color }} />
                            {styleOf(t).name}
                        </span>
                    ))}
                </div>
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} className="graph-svg">
                <defs>
                    <pattern id="dotgrid" width="22" height="22" patternUnits="userSpaceOnUse">
                        <circle cx="1" cy="1" r="1" fill="#E3E9F2" />
                    </pattern>
                    <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4"
                        markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 8 4 L 0 8 z" fill="#B9C4D4" />
                    </marker>
                    <marker id="arrow-active" viewBox="0 0 8 8" refX="7" refY="4"
                        markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 8 4 L 0 8 z" fill="#0F4C81" />
                    </marker>
                </defs>
                <rect width={W} height={H} fill="url(#dotgrid)" />
                {layout.links.map((l, i) => {
                    const active = hover && (l.source.id === hover || l.target.id === hover);
                    return (
                        <g key={i}>
                            <line
                                x1={l.source.x} y1={l.source.y}
                                x2={l.target.x} y2={l.target.y}
                                stroke={active ? '#0F4C81' : '#C6D0DE'}
                                strokeWidth={active ? 1.8 : 1.1}
                                markerEnd={active ? 'url(#arrow-active)' : 'url(#arrow)'}
                            />
                            <text
                                x={(l.source.x + l.target.x) / 2}
                                y={(l.source.y + l.target.y) / 2 - 4}
                                className={`edge-label ${active ? 'active' : ''}`}
                            >
                                {l.type}
                            </text>
                        </g>
                    );
                })}
                {layout.nodes.map(n => (
                    <g
                        key={n.id}
                        transform={`translate(${n.x},${n.y})`}
                        onMouseEnter={() => setHover(n.id)}
                        onMouseLeave={() => setHover(null)}
                    >
                        <circle
                            r={(n.type === 'Chunk' ? 7 : 11) + 3.5}
                            fill={styleOf(n.type).color}
                            opacity={hover === n.id ? 0.18 : 0}
                            style={{ transition: 'opacity 0.15s' }}
                        />
                        <circle
                            r={n.type === 'Chunk' ? 7 : 11}
                            fill={styleOf(n.type).color}
                            opacity={hover && hover !== n.id ? 0.38 : 1}
                            stroke="#fff" strokeWidth="2"
                            style={{ transition: 'opacity 0.15s' }}
                        />
                        <text y={n.type === 'Chunk' ? 18 : 24} className="node-label">
                            {n.label.length > 14 ? `${n.label.slice(0, 13)}…` : n.label}
                        </text>
                    </g>
                ))}
            </svg>
            <p className="graph-caption">
                재결서 본문(검색 진입점) → 사고 → 원인 유형·선박·장소로 이어지는 근거 경로입니다.
                노드에 마우스를 올리면 연결이 강조됩니다.
            </p>
        </div>
    );
}

export default GraphView;
