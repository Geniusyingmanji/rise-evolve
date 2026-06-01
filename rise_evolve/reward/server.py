from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

from rise_evolve.reward.critic import score_agent_result, score_verifier_item


class RewardHandler(BaseHTTPRequestHandler):
    use_programmatic_priors = False

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return
        if self.path == "/score_verifier":
            self._send_json(200, score_verifier_item(payload, use_programmatic_priors=self.use_programmatic_priors))
            return
        if self.path == "/score_agent":
            self._send_json(200, score_agent_result(payload, use_programmatic_priors=self.use_programmatic_priors))
            return
        self._send_json(404, {"error": "expected /score_verifier or /score_agent"})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the lightweight RISE-Critic scoring API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--use-programmatic-priors", action="store_true")
    args = parser.parse_args(argv)
    RewardHandler.use_programmatic_priors = args.use_programmatic_priors
    server = HTTPServer((args.host, args.port), RewardHandler)
    print(f"RISE-Critic server listening on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
