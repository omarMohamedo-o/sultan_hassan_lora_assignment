"""Local HTTP review interface for human-in-the-loop candidate screening."""

import json
import logging
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.enums import CategoryEnum, ReviewDecisionEnum
from sultan_hassan.review.decisions import DecisionStore

logger = logging.getLogger(__name__)

REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sultan Hassan Dataset Review Interface</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --accent: #58a6ff;
            --green: #238636;
            --red: #da3633;
            --orange: #d29922;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        .main-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: #000;
        }
        .image-container {
            max-width: 90%;
            max-height: 80vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 80vh;
            object-fit: contain;
            border-radius: 6px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.8);
        }
        .side-panel {
            width: 380px;
            background: var(--surface);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px;
            overflow-y: auto;
        }
        h2 { font-size: 1.2rem; color: var(--text-heading); margin-bottom: 16px; }
        .meta-box {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 20px;
            font-size: 0.85rem;
            line-height: 1.6;
        }
        .section-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-heading);
            margin: 12px 0 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-bottom: 16px;
        }
        button {
            padding: 10px;
            border: 1px solid var(--border);
            background: #21262d;
            color: var(--text-heading);
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        button:hover { background: #30363d; }
        button.btn-keep { background: var(--green); border-color: #2ea043; color: #fff; }
        button.btn-keep:hover { background: #2ea043; }
        button.btn-reject { background: var(--red); border-color: #f85149; color: #fff; }
        button.btn-reject:hover { background: #f85149; }
        button.btn-rifai { background: var(--orange); border-color: #e3b341; color: #000; }
        button.active { outline: 2px solid var(--accent); }
        .nav-controls {
            display: flex;
            justify-content: space-between;
            margin-top: auto;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }
        .nav-btn { flex: 1; margin: 0 4px; }
        .progress-bar {
            height: 6px;
            background: #30363d;
            border-radius: 3px;
            margin-bottom: 12px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: var(--accent);
            width: 0%;
            transition: width 0.2s ease;
        }
    </style>
</head>
<body>
    <div class="main-panel">
        <div class="image-container">
            <img id="currentImage" src="" alt="Candidate Preview" />
        </div>
    </div>
    <div class="side-panel">
        <h2>Human Review Portal</h2>
        <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        <div id="progressText" style="font-size: 0.8rem; margin-bottom: 12px; color: #8b949e;">Loading...</div>

        <div class="meta-box" id="metaBox">Select or load a candidate image.</div>

        <div class="section-title">Decision</div>
        <div class="btn-grid">
            <button class="btn-keep" onclick="submitDecision('KEEP')">Keep (K)</button>
            <button class="btn-reject" onclick="submitDecision('REJECT')">Reject (R)</button>
            <button class="btn-rifai" onclick="submitDecision('AL_RIFAI')">Al-Rifa'i (A)</button>
            <button onclick="submitDecision('MIXED')">Mixed</button>
            <button onclick="submitDecision('WRONG_BUILDING')">Wrong Bldg</button>
            <button onclick="submitDecision('PEOPLE')">People Subject</button>
            <button onclick="submitDecision('WATERMARK')">Watermark/Text</button>
            <button onclick="submitDecision('LOW_QUALITY')">Low Quality</button>
        </div>

        <div class="section-title">Architectural Category</div>
        <div class="btn-grid" id="categoryGrid">
            <button onclick="setCategory('EXTERIOR')">Exterior Facade</button>
            <button onclick="setCategory('ENTRANCE')">Entrance Portal</button>
            <button onclick="setCategory('COURTYARD')">Courtyard</button>
            <button onclick="setCategory('IWAN')">Four Iwans</button>
            <button onclick="setCategory('MINARET')">Minarets</button>
            <button onclick="setCategory('DOME')">Dome</button>
            <button onclick="setCategory('MUQARNAS')">Muqarnas</button>
            <button onclick="setCategory('ORNAMENT')">Ornament</button>
            <button onclick="setCategory('HANGING_LAMP')">Hanging Lamp</button>
            <button onclick="setCategory('OTHER')">Other</button>
        </div>

        <div class="nav-controls">
            <button class="nav-btn" onclick="prevItem()">&larr; Prev (P)</button>
            <button class="nav-btn" onclick="nextItem()">Next (N) &rarr;</button>
        </div>
    </div>

    <script>
        let candidates = [];
        let currentIndex = 0;
        let selectedCategory = 'OTHER';

        async function init() {
            const res = await fetch('/api/candidates');
            candidates = await res.json();
            if (candidates.length > 0) {
                renderCurrent();
            } else {
                document.getElementById('metaBox').innerText = 'No candidate images found.';
            }
        }

        function setCategory(cat) {
            selectedCategory = cat;
            document.querySelectorAll('#categoryGrid button').forEach(b => {
                b.classList.toggle('active', b.innerText.toUpperCase().includes(cat));
            });
        }

        function renderCurrent() {
            if (candidates.length === 0) return;
            const item = candidates[currentIndex];
            document.getElementById('currentImage').src = '/image/' + encodeURIComponent(item.filename);
            document.getElementById('progressFill').style.width = ((currentIndex + 1) / candidates.length * 100) + '%';
            document.getElementById('progressText').innerText = `Item ${currentIndex + 1} of ${candidates.length}`;

            document.getElementById('metaBox').innerHTML = `
                <div><strong>ID:</strong> ${item.frame_id}</div>
                <div><strong>Filename:</strong> ${item.filename}</div>
                <div><strong>Resolution:</strong> ${item.width}x${item.height}</div>
                <div><strong>Current Decision:</strong> <span style="color:var(--accent)">${item.current_decision || 'Pending'}</span></div>
                <div><strong>Current Category:</strong> ${item.current_category || 'None'}</div>
            `;
            if (item.current_category) setCategory(item.current_category);
        }

        async function submitDecision(decision) {
            if (candidates.length === 0) return;
            const item = candidates[currentIndex];
            await fetch('/api/decide', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_id: item.frame_id,
                    decision: decision,
                    category: selectedCategory,
                })
            });
            item.current_decision = decision;
            item.current_category = selectedCategory;
            nextItem();
        }

        function nextItem() {
            if (currentIndex < candidates.length - 1) {
                currentIndex++;
                renderCurrent();
            }
        }

        function prevItem() {
            if (currentIndex > 0) {
                currentIndex--;
                renderCurrent();
            }
        }

        document.addEventListener('keydown', (e) => {
            const key = e.key.toUpperCase();
            if (key === 'K') submitDecision('KEEP');
            else if (key === 'R') submitDecision('REJECT');
            else if (key === 'A') submitDecision('AL_RIFAI');
            else if (key === 'N') nextItem();
            else if (key === 'P') prevItem();
        });

        init();
    </script>
</body>
</html>
"""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for candidate image inspection and decision logging."""

    config: AppConfig
    store: DecisionStore
    candidates_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(REVIEW_HTML.encode("utf-8"))

        elif url.path == "/api/candidates":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            files = sorted([p for p in self.candidates_dir.glob("*.jpg") if p.is_file()])
            items = []
            for p in files:
                stem_parts = p.stem.split("_")
                frame_id = "_".join(stem_parts[-2:]) if len(stem_parts) >= 2 else p.stem
                existing = self.store.get_decision(frame_id)
                items.append(
                    {
                        "frame_id": frame_id,
                        "filename": p.name,
                        "width": 1024,
                        "height": 1024,
                        "current_decision": existing.decision.value if existing else None,
                        "current_category": existing.category.value if existing else None,
                    }
                )
            self.wfile.write(json.dumps(items).encode("utf-8"))

        elif url.path.startswith("/image/"):
            filename = urllib.parse.unquote(url.path[len("/image/") :])
            img_path = self.candidates_dir / filename
            if img_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(img_path.read_bytes())
            else:
                self.send_error(404, "Image not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/decide":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))

            frame_id = payload.get("frame_id")
            dec_str = payload.get("decision", "REVIEW")
            cat_str = payload.get("category", "OTHER")

            try:
                dec_enum = ReviewDecisionEnum(dec_str)
                cat_enum = CategoryEnum(cat_str)
                rec = self.store.save_decision(
                    frame_id=frame_id,
                    decision=dec_enum,
                    category=cat_enum,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(rec.model_dump()).encode("utf-8"))
            except Exception as exc:
                self.send_error(400, f"Invalid decision: {exc}")
        else:
            self.send_error(404)


def run_review_server(config: AppConfig, port: int = 8080) -> None:
    """Launch lightweight review server on localhost."""
    store = DecisionStore(config)
    candidates_dir = Path(config.paths.candidates_images)

    ReviewRequestHandler.config = config
    ReviewRequestHandler.store = store
    ReviewRequestHandler.candidates_dir = candidates_dir

    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, ReviewRequestHandler)
    print(f"\n[INFO] Sultan Hassan Review Server running at: http://127.0.0.1:{port}")
    print("[INFO] Press Ctrl+C in terminal to stop server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Review server stopped.")
    finally:
        httpd.server_close()
