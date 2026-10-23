import { defineConfig } from 'vite';

// relative base so the build works from a GitHub Pages project path
export default defineConfig({
  base: './',
  build: { target: 'es2022' },
});
