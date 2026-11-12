import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react-swc"
import { defineConfig } from "vite"
import { detectorApiPlugin } from "./vite-detector-plugin"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), detectorApiPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})