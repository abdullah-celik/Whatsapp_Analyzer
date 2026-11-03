"""A fake Ollama server for tests: same HTTP API, deterministic JSON answers."""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wa_analyzer.lexicon_engine import WordClassifier

CLF = WordClassifier()


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "qwen2.5:3b"}, {"name": "qwen2.5:7b"}, {"name": "mock:latest"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        Handler.calls.append(req)
        user = req["messages"][-1]["content"]
        system = req["messages"][0]["content"]
        if "Participants:" in user:
            names = [x.strip() for x in re.search(r"Participants: (.*)", user).group(1).split(",")]
            # first speaker gets warmer scores early in the alphabet of months: deterministic but varied
            month = re.search(r"Date: \w+ \d+ (\w+)", user).group(1)
            w = 2 + (len(month) % 4)
            out = {"people": [{"name": nm.split()[0], "warmth": min(5, w + (i == 0)), "engagement": 3 + (i == 0),
                               "interest_in_other": max(1, w - i)} for i, nm in enumerate(names)],
                   "drives_conversation": names[0].split()[0], "tone": ["friendly", "affectionate", "practical", "tense"][len(month) % 4],
                   "conflict": len(month) % 4 == 3, "topics": ["plans", "daily life"], "summary": f"They chat about plans in {month}."}
        elif "Label these messages" in user:
            items = []
            for line in user.splitlines():
                mm = re.match(r"(\d+)\. .*MESSAGE: \"(.*)\"$", line)
                if mm:
                    labs = [k for k, v in CLF.classify(mm.group(2)).items() if v and k in json.dumps(req["format"])]
                    items.append({"id": int(mm.group(1)), "labels": labs, "sentiment": "positive" if "affection" in labs else "neutral"})
            out = {"items": items}
        elif "name" in json.dumps(req.get("format", "")) and "topic" in system.lower():
            out = {"name": "Mock topic"}
        else:
            out = {"summary": "A calm month with regular, friendly contact."}
        self._send({"model": req["model"], "message": {"role": "assistant", "content": json.dumps(out)}, "done": True})


def start(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
