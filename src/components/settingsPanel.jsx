import React, { useState, useEffect } from 'react';

export function SettingsPanel() {
  const [useLocalBackend, setUseLocalBackend] = useState(
    () => localStorage.getItem('use_local_backend') === 'true'
  );

  useEffect(() => {
    localStorage.setItem('use_local_backend', useLocalBackend);
    window.dispatchEvent(new CustomEvent('backend-toggle-changed', { detail: useLocalBackend }));
  }, [useLocalBackend]);

  return (
    <div style={{ padding: '15px', background: '#222', color: '#fff', borderRadius: '8px' }}>
      <h3>Tracking Configuration</h3>
      <hr style={{ borderColor: '#444' }} />
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '10px' }}>
        <div>
          <strong style={{ display: 'block' }}>Local Processing Only</strong>
          <small style={{ color: '#aaa' }}>Bypasses cloud server completely (Prevents buffer loops)</small>
        </div>
        <input 
          type="checkbox" 
          checked={useLocalBackend} 
          onChange={(e) => setUseLocalBackend(e.target.checked)}
          style={{ width: '20px', height: '20px', cursor: 'pointer' }}
        />
      </div>
    </div>
  );
}