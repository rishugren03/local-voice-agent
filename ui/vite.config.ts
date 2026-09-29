import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The UI is served on 5173 and the API on 8080, which makes every request a
// cross-origin one. That works — the API allows both ports — but the dev proxy
// below is the better default because it removes the CORS round trip from the
// critical path of an edit-to-test loop and means a failure shows up as one
// clean proxy error instead of a browser CORS message that hides the real
// status.
//
// VITE_API_BASE still wins: set it to an absolute URL to point the built bundle
// at a control plane on another machine.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8080", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8080", changeOrigin: true },
    },
  },
  build: {
    // livekit-client is ~520kB and only the Talk screen needs it. It is already
    // split out by a dynamic import there; this ceiling exists so that if a
    // future dependency lands in the entry chunk, the build says so instead of
    // the app quietly getting slower.
    chunkSizeWarningLimit: 400,
  },
});
