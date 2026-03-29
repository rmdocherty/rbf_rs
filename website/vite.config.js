import { defineConfig } from 'vite';

export default defineConfig({
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        // REMOVE format: 'iife'
      }
    }

  },
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  assetsInclude: ['**/*.wasm'],
  publicDir: 'public'
});
