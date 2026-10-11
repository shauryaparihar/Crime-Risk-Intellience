import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Local port used by the docs, docker-compose and the backend's default CORS origin.
  server: { port: 5174, strictPort: true },
  preview: { port: 5174, strictPort: true },
})
