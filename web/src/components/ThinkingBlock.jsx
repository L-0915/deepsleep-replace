import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';

export default function ThinkingBlock({ thinking, isStreaming }) {
  const [expanded, setExpanded] = useState(true);

  if (!thinking && !isStreaming) return null;

  return (
    <div
      style={{
        margin: '8px 0 12px',
        borderRadius: 10,
        border: '1px solid rgba(99, 102, 241, 0.3)',
        background: 'var(--thinking-bg)',
        overflow: 'hidden',
      }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          width: '100%',
          padding: '10px 14px',
          background: 'none',
          border: 'none',
          color: 'var(--accent)',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 13,
          fontWeight: 500,
        }}
      >
        <span>💭 思考过程</span>
        {isStreaming && (
          <span className="thinking-dots">
            <span /><span /><span />
          </span>
        )}
        <motion.div
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center' }}
        >
          <ChevronDown size={16} />
        </motion.div>
      </button>
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            style={{ overflow: 'hidden' }}
          >
            <div
              style={{
                padding: '0 14px 12px',
                fontSize: 13,
                color: 'var(--text-secondary)',
                lineHeight: 1.6,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {thinking || (isStreaming ? '' : '')}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
