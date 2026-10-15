import {defineConfig} from "vite";

export default defineConfig({
  build: {
    outDir: "../src/expr_tracker/_web",
    emptyOutDir: true,
  },
  server: {
    proxy: {"/api": "http://127.0.0.1:9110"},
  },
});
