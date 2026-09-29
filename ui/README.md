# Voice Agent UI

The control-plane web app: seven screens for editing agents, talking to them, and
reading what happened. Full documentation is in the [repository
README](../README.md#the-web-ui).

```bash
npm install
npm run dev        # http://127.0.0.1:5173, proxies /api to 127.0.0.1:8080
npm run build      # -> dist/
npm run preview    # serve the built bundle
npx tsc -b         # typecheck
npx oxlint src     # lint
```

The control plane has to be running separately — `python3 -m control_plane.app`
from the repository root.

Set `VITE_API_BASE` to an absolute URL to point a built bundle at an API running
somewhere other than the same origin (the API allows ports 5173 and 3000 by
default).
