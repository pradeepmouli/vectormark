import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const input = process.env.INPUT;
if (!input) {
  throw new Error("INPUT environment variable is not set");
}

const isDevelopment = process.env.NODE_ENV === "development";

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    cssMinify: !isDevelopment,
    emptyOutDir: true,
    minify: !isDevelopment,
    outDir: "dist",
    rollupOptions: {
      input,
    },
    sourcemap: isDevelopment ? "inline" : undefined,
  },
});
