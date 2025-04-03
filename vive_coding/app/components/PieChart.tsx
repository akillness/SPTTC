const calculateFontSize = (value: number) => {
  // 값의 제곱근에 비례하는 폰트 크기 (기준 크기 14px, 최소 10px ~ 최대 24px)
  return Math.min(Math.max(Math.sqrt(value) * 14, 10), 24);
};

const renderKeywords = (keywords: Keyword[]) => (
  <div className="keyword-container" style={{ 
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    padding: '16px',
    backgroundColor: '#f8f9fa',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  }}>
    {keywords.map(({ label, value }) => (
      <span 
        key={label}
        style={{
          fontSize: `${calculateFontSize(value)}px`,
          padding: '6px 12px',
          borderRadius: '20px',
          backgroundColor: '#e9ecef',
          color: '#495057',
          transition: 'all 0.2s ease',
          cursor: 'pointer',
          ':hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }
        }}
      >
        {label}
      </span>
    ))}
  </div>
); 