import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-vision'],
  },
  server: {
    headers: {
      // Required for SharedArrayBuffer used by MediaPipe WASM
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})