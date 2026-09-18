import http.server
import socketserver
import urllib.request
import urllib.error

BACKEND = "https://document-screening-system-3tvd.onrender.com"
PORT = 3000
ROOT = "."

EXCLUDED_HEADERS = {"host", "origin", "connection", "content-length", "accept-encoding", "http2-settings", "proxy-connection", "upgrade"}


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_PATCH(self):
        self._handle("PATCH")

    def do_DELETE(self):
        self._handle("DELETE")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _handle(self, method):
        if self.path.startswith("/api/") or self.path == "/health":
            self._proxy(method)
        else:
            super().do_GET()

    def _proxy(self, method):
        data = None
        if self.headers.get("Content-Length"):
            try:
                data = self.rfile.read(int(self.headers["Content-Length"]))
            except ValueError:
                data = None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in EXCLUDED_HEADERS}
        req = urllib.request.Request(BACKEND + self.path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            payload = ('{"success":false,"error":{"code":"BAD_GATEWAY","message":"proxy: %s"}}' % str(exc)).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, fmt, *args):
        import sys
        sys.stdout.write("[%s] %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("", PORT), ProxyHandler) as httpd:
        print("Dashboard + API proxy on port %d -> %s" % (PORT, BACKEND), flush=True)
        httpd.serve_forever()