"""
Potree-based 3D point cloud viewer for (N, 3) .npy arrays.

Converts the point cloud to a LAS file, builds a Potree 2.0 octree, and
serves an interactive WebGL viewer in the browser. Handles any cloud size
well (Potree uses level-of-detail streaming, so even 100M+ points stay
interactive).

Click anywhere in the 3D view to pick the nearest point. Its original array
index and decoded token are shown in the top-left overlay.

Assets required in vendor/potree/ (run scripts/fetch_potree.sh):
    PotreeConverter    – built from github.com/potree/PotreeConverter
    viewer_build/      – Potree 1.8.2 build/ directory
    viewer_libs/       – Potree 1.8.2 libs/ directory

Usage:
    lift-potree data/reduced/1_umap.npy
    lift-potree points.npy --port 8080 --out-dir ./out
    lift-potree points.npy --no-tokenizer
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np

from lift.paths import POTREE_BUILD, POTREE_CONVERTER, POTREE_LIBS

VIEWER_BUILD = POTREE_BUILD
VIEWER_LIBS  = POTREE_LIBS

MIMETYPES = {
    ".html": "text/html",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".json": "application/json",
    ".bin":  "application/octet-stream",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".svg":  "image/svg+xml",
    ".woff": "font/woff",
    ".woff2":"font/woff2",
    ".ttf":  "font/ttf",
    ".glsl": "text/plain",
}


def npy_to_las(pts: np.ndarray, out_path: Path):
    n = pts.shape[0]
    norm = (np.arange(n) / max(n - 1, 1)).astype(np.float32)
    rgb16 = (plt.colormaps["turbo"](norm)[:, :3] * 65535).astype(np.uint16)

    las = laspy.LasData(header=laspy.LasHeader(point_format=2, version="1.4"))
    scale = 1e-5
    offset = pts.min(axis=0).astype(np.float64)
    las.header.offsets = offset
    las.header.scales = np.array([scale, scale, scale])
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    las.red   = rgb16[:, 0]
    las.green = rgb16[:, 1]
    las.blue  = rgb16[:, 2]
    las.write(str(out_path))


def build_octree(las_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(POTREE_CONVERTER.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(
        [str(POTREE_CONVERTER), str(las_path), "-o", str(out_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def write_positions_bin(pts: np.ndarray, out_dir: Path):
    pts.astype(np.float32).tofile(out_dir / "positions.bin")


def generate_html(out_dir: Path, n_points: int, title: str):
    turbo_lut = (plt.colormaps["turbo"](np.linspace(0, 1, 256))[:, :3]).tolist()
    turbo_lut_json = json.dumps(turbo_lut)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <link rel="stylesheet" href="/build/potree/potree.css">
  <link rel="stylesheet" href="/libs/jquery-ui/jquery-ui.min.css">
  <link rel="stylesheet" href="/libs/openlayers3/ol.css">
  <link rel="stylesheet" href="/libs/spectrum/spectrum.css">
  <link rel="stylesheet" href="/libs/jstree/themes/mixed/style.css">
  <style>
    #token-overlay {{
      position: absolute;
      top: 10px; right: 10px;
      background: rgba(0,0,0,0.75);
      color: #ffe000;
      font: bold 15px/1.6 monospace;
      padding: 10px 16px;
      border-radius: 6px;
      border: 1px solid rgba(255,224,0,0.5);
      pointer-events: none;
      display: none;
      z-index: 9999;
      white-space: pre;
      max-width: 420px;
    }}
  </style>
</head>
<body>
  <script src="/libs/jquery/jquery-3.1.1.min.js"></script>
  <script src="/libs/spectrum/spectrum.js"></script>
  <script src="/libs/jquery-ui/jquery-ui.min.js"></script>
  <script src="/libs/other/BinaryHeap.js"></script>
  <script src="/libs/tween/tween.min.js"></script>
  <script src="/libs/d3/d3.js"></script>
  <script src="/libs/proj4/proj4.js"></script>
  <script src="/libs/openlayers3/ol.js"></script>
  <script src="/libs/i18next/i18next.js"></script>
  <script src="/libs/jstree/jstree.js"></script>
  <script src="/build/potree/potree.js"></script>
  <script src="/libs/plasio/js/laslaz.js"></script>

  <div class="potree_container" style="position:absolute;width:100%;height:100%;left:0;top:0">
    <div id="potree_render_area"></div>
    <div id="potree_sidebar_container"></div>
    <div id="token-overlay"></div>
    <div id="token-search" style="position:absolute;bottom:20px;left:20px;z-index:9999;
         display:flex;gap:6px;align-items:center;background:rgba(0,0,0,0.75);
         padding:8px 12px;border-radius:6px;border:1px solid rgba(255,255,255,0.15);">
      <input id="search-input" type="text" placeholder='find token, e.g. " garden"'
        style="font:14px monospace;background:transparent;color:#fff;border:none;
               outline:none;width:200px;" />
      <button id="search-btn"
        style="background:none;border:none;color:#ffe000;cursor:pointer;
               font:bold 14px monospace;padding:0 4px;">→</button>
      <button id="search-clear"
        style="background:none;border:none;color:#888;cursor:pointer;
               font:14px monospace;padding:0 4px;" title="hide marker">✕</button>
    </div>
    <div id="search-status" style="position:absolute;bottom:62px;left:20px;z-index:9999;
         color:#aaa;font:12px monospace;display:none;background:rgba(0,0,0,0.6);
         padding:4px 10px;border-radius:4px;"></div>
  </div>

  <script type="module">
    import * as THREE from "/libs/three.js/build/three.module.js";

    const N_POINTS = {n_points};
    const overlay  = document.getElementById("token-overlay");

    // ── Viewer setup ──────────────────────────────────────────────────────────
    window.viewer = new Potree.Viewer(document.getElementById("potree_render_area"));
    viewer.setEDLEnabled(true);
    viewer.setFOV(60);
    viewer.setPointBudget(3_000_000);
    viewer.setMinNodeSize(0);
    viewer.setBackground("black");

    viewer.loadGUI(() => {{
      viewer.setLanguage("en");
      viewer.toggleSidebar();

      // ── Inject "Examine Token" into the Potree measurement toolbar ──────────
      // No inline filter — potree.css .button-icon:hover adds drop-shadow and
      // an inline filter would permanently override it.
      let iconURL = Potree.resourcePath + "/icons/magnifying_glass.svg";
      let elExamine = $(`
        <img src="${{iconURL}}" style="width:32px;height:32px;"
             class="button-icon" title="Examine Token — click points to decode vocab tokens" />
      `);
      elExamine.click(() => {{
        examineActive = true;
        startExamine();   // startExamine calls startInsertion which cancels any other active tool
      }});
      $("#tools").append(elExamine);
    }});

    // ── Load cloud ────────────────────────────────────────────────────────────
    let highlightRadius = 0.05;
    let bbDiag = 1;
    let pcoReady = false;
    Potree.loadPointCloud("/metadata.json", "{title}", function(e) {{
      viewer.scene.addPointCloud(e.pointcloud);
      let mat = e.pointcloud.material;
      mat.activeAttributeName = "rgba";
      mat.size = 2.5;
      mat.pointSizeType = Potree.PointSizeType.FIXED;
      mat.shape = Potree.PointShape.CIRCLE;
      viewer.fitToScreen();
      bbDiag = e.pointcloud.boundingBox.getSize(new THREE.Vector3()).length();
      highlightRadius = bbDiag * 0.002;
      pcoReady = true;
      tryBuildAllPoints();
    }});

    // ── Load original float32 positions for nearest-neighbour search ──────────
    let positions = null;
    fetch("/positions.bin")
      .then(r => r.arrayBuffer())
      .then(buf => {{
        positions = new Float32Array(buf);
        tryBuildAllPoints();
      }});

    const TURBO_LUT = {turbo_lut_json};
    function turboColor(t) {{
      let i = Math.min(255, Math.max(0, Math.floor(t * 255)));
      return TURBO_LUT[i];
    }}

    let allPointsObj = null;

    function tryBuildAllPoints() {{
      if (!pcoReady || !positions) return;
      let geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      let colors = new Float32Array(N_POINTS * 3);
      for (let i = 0; i < N_POINTS; i++) {{
        let t = i / (N_POINTS - 1);
        let [r, g, b] = turboColor(t);
        colors[i*3]   = r;
        colors[i*3+1] = g;
        colors[i*3+2] = b;
      }}
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      let ptsMat = new THREE.PointsMaterial({{size: 2, vertexColors: true, sizeAttenuation: false}});
      allPointsObj = new THREE.Points(geo, ptsMat);
      viewer.scene.scene.add(allPointsObj);
    }}

    function nearestIndex(px, py, pz) {{
      let best = -1, bestD = Infinity;
      for (let i = 0; i < N_POINTS; i++) {{
        let dx = positions[i*3]   - px;
        let dy = positions[i*3+1] - py;
        let dz = positions[i*3+2] - pz;
        let d  = dx*dx + dy*dy + dz*dz;
        if (d < bestD) {{ bestD = d; best = i; }}
      }}
      return best;
    }}

    // ── Examine Token tool ────────────────────────────────────────────────────
    // Uses THREE.js Raycaster against the full 262k-point cloud so every token
    // is reachable, not just the sparse Potree LOD subset.
    let examineActive   = false;
    let examineHoverIdx = -1;

    let examineGhost = (() => {{
      let mesh = new THREE.Mesh(
        new THREE.SphereGeometry(1, 12, 8),
        new THREE.MeshBasicMaterial({{color: 0xff3333, opacity: 0.8, transparent: true}})
      );
      mesh.visible = false;
      viewer.scene.scene.add(mesh);
      return mesh;
    }})();

    let examineRaycaster = new THREE.Raycaster();

    function deactivateExamine() {{
      examineActive = false;
      examineGhost.visible = false;
      examineHoverIdx = -1;
    }}

    viewer.renderer.domElement.addEventListener("contextmenu", (e) => {{
      if (examineActive) {{ deactivateExamine(); e.stopImmediatePropagation(); }}
    }}, true);

    function startExamine() {{
      examineActive = true;
      viewer.dispatchEvent({{type: "cancel_insertions"}});
    }}

    viewer.renderer.domElement.addEventListener("mousemove", (e) => {{
      if (!examineActive || !allPointsObj) return;
      let rect = viewer.renderer.domElement.getBoundingClientRect();
      let mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width)  * 2 - 1,
        -((e.clientY - rect.top)  / rect.height) * 2 + 1
      );
      examineRaycaster.params.Points = {{ threshold: viewer.scene.view.radius * 0.012 }};
      examineRaycaster.setFromCamera(mouse, viewer.scene.getActiveCamera());
      let hits = examineRaycaster.intersectObject(allPointsObj);
      if (hits.length > 0) {{
        let idx = hits[0].index;
        examineGhost.position.fromArray(positions, idx * 3);
        examineGhost.scale.setScalar(highlightRadius * 0.1);
        examineGhost.visible = true;
        examineHoverIdx = idx;
      }} else {{
        examineGhost.visible = false;
        examineHoverIdx = -1;
      }}
    }});

    viewer.renderer.domElement.addEventListener("click", (e) => {{
      if (!examineActive || examineHoverIdx < 0) return;
      let idx = examineHoverIdx;
      overlay.style.display = "block";
      overlay.textContent   = `index ${{idx}}  …`;
      fetch(`/decode?idx=${{idx}}`)
        .then(r => r.json())
        .then(d => {{
          let tok = d.token !== undefined ? `  token=${{JSON.stringify(d.token)}}` : "";
          overlay.textContent = `index ${{idx}}${{tok}}`;
        }})
        .catch(() => {{ overlay.textContent = `index ${{idx}}`; }});
    }});

    // ── Find Token search box ─────────────────────────────────────────────────
    let searchHighlight = null;

    function flyToIndex(idx) {{
      if (!positions || idx < 0 || idx >= N_POINTS) return;
      let px = positions[idx * 3], py = positions[idx * 3 + 1], pz = positions[idx * 3 + 2];

      if (!searchHighlight) {{
        let wf  = new THREE.WireframeGeometry(new THREE.SphereGeometry(1, 16, 10));
        let mat = new THREE.LineBasicMaterial({{color: 0xffff00, opacity: 0.55, transparent: true}});
        searchHighlight = new THREE.LineSegments(wf, mat);
        viewer.scene.scene.add(searchHighlight);
      }}
      searchHighlight.visible = true;
      searchHighlight.scale.setScalar(highlightRadius * 0.5);
      searchHighlight.position.set(px, py, pz);

      viewer.controls.stop();
      let view   = viewer.scene.view;
      let tr     = bbDiag * 0.04;
      let dir    = view.direction.clone();
      // Camera must end up tr distance from token, not at it
      let endPos = new THREE.Vector3(px, py, pz).sub(dir.multiplyScalar(tr));
      let startPos = view.position.clone();
      let startR   = view.radius;
      let tw = {{p: 0}};
      new TWEEN.Tween(tw).to({{p: 1}}, 800)
        .easing(TWEEN.Easing.Quadratic.InOut)
        .onUpdate(() => {{
          view.position.set(
            startPos.x + (endPos.x - startPos.x) * tw.p,
            startPos.y + (endPos.y - startPos.y) * tw.p,
            startPos.z + (endPos.z - startPos.z) * tw.p
          );
          view.radius = startR + (tr - startR) * tw.p;
        }})
        .start();
    }}

    function doSearch() {{
      // Strip surrounding quotes so the user can type " garden" to mean the
      // space-prefixed token without the tokenizer seeing the quote characters
      let raw    = document.getElementById("search-input").value;
      let tok    = raw.replace(/^["'`]|["'`]$/g, "");
      let status = document.getElementById("search-status");
      if (!tok && !raw) return;
      status.style.display = "block";
      status.textContent   = "searching…";
      fetch("/encode?token=" + encodeURIComponent(tok))
        .then(r => r.json())
        .then(d => {{
          if (d.idx < 0) {{ status.textContent = "token not found"; return; }}
          let info = `idx ${{d.idx}}`;
          if (d.decoded !== undefined) info += `  →  ${{JSON.stringify(d.decoded)}}`;
          if (d.n_tokens > 1)         info += `  (${{d.n_tokens}} tokens, showing first)`;
          status.textContent = info;
          flyToIndex(d.idx);
        }})
        .catch(() => {{ status.textContent = "error"; }});
    }}

    document.getElementById("search-input").addEventListener("keydown", e => {{
      if (e.key === "Enter") doSearch();
    }});
    document.getElementById("search-btn").addEventListener("click", doSearch);
    document.getElementById("search-clear").addEventListener("click", () => {{
      if (searchHighlight) searchHighlight.visible = false;
      document.getElementById("search-status").style.display = "none";
    }});
  </script>
</body>
</html>"""
    (out_dir / "index.html").write_text(html)


def make_server_class(out_dir: Path, tokenizer):
    """Returns an HTTPServer-compatible handler class."""

    class Handler(BaseHTTPRequestHandler):

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path   = urllib.parse.unquote(parsed.path)

            # ── /encode?token=TEXT ────────────────────────────────────────────
            if path == "/encode":
                qs  = urllib.parse.parse_qs(parsed.query)
                tok = qs.get("token", [""])[0]
                result = {"idx": -1, "n_tokens": 0, "decoded": ""}
                if tokenizer is not None and tok:
                    try:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        # print(tok)
                        result["n_tokens"] = len(ids)
                        if ids:
                            result["idx"] = int(ids[0])
                            result["decoded"] = tokenizer.decode([ids[0]])
                    except Exception:
                        pass
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            # ── /decode?idx=N ─────────────────────────────────────────────────
            if path == "/decode":
                qs  = urllib.parse.parse_qs(parsed.query)
                idx = int(qs.get("idx", [-1])[0])
                tok = ""
                if tokenizer is not None and idx >= 0:
                    try:
                        tok = tokenizer.decode([idx])
                    except Exception:
                        pass
                body = json.dumps({"token": tok}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            # ── static files ──────────────────────────────────────────────────
            rel = path.lstrip("/") or "index.html"
            candidates = [
                out_dir / rel,
                VIEWER_BUILD / rel.removeprefix("build/"),
                VIEWER_LIBS  / rel.removeprefix("libs/"),
            ]
            for cand in candidates:
                if cand.is_file():
                    mime = MIMETYPES.get(cand.suffix, "application/octet-stream")
                    data = cand.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

            self.send_error(404, f"Not found: {rel}")

        def log_message(self, fmt, *args):
            pass  # silence access log

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npy_path", nargs="?", default=None,
                        help="path to (N, 3) .npy point cloud (not needed with --serve-only)")
    parser.add_argument("--out-dir", default=None,
                         help="output directory for octree + HTML (default: <npy_path>_potree/)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-tokenizer", action="store_true",
                         help="skip loading the tokenizer (faster startup for non-token clouds)")
    parser.add_argument("--serve-only", action="store_true",
                         help="skip build steps and serve an existing pre-built octree directory "
                              "(requires --out-dir)")
    args = parser.parse_args()

    if args.serve_only:
        if not args.out_dir:
            sys.exit("--serve-only requires --out-dir pointing to a pre-built octree directory")
        out_dir = Path(args.out_dir)
        if not (out_dir / "metadata.json").exists():
            sys.exit(f"No metadata.json found in {out_dir} — directory may not be a built octree")
        for asset in (VIEWER_BUILD, VIEWER_LIBS):
            if not asset.exists():
                sys.exit(f"Missing required asset: {asset}")
        tokenizer = None
        if not args.no_tokenizer:
            from lift.models.tokenizer import load_tokenizer
            print("loading tokenizer...")
            tokenizer = load_tokenizer()
        print(f"\nServing pre-built octree from {out_dir}")
        print(f"Open http://localhost:{args.port} in your browser\n")
        server = HTTPServer((args.host, args.port), make_server_class(out_dir, tokenizer))
        server.serve_forever()
        return

    for asset in (VIEWER_BUILD, VIEWER_LIBS):
        if not asset.exists():
            sys.exit(f"Missing required asset: {asset}\n"
                     f"Run scripts/fetch_potree.sh to install the Potree viewer assets.")

    if not args.npy_path:
        sys.exit("npy_path is required unless --serve-only is used")

    npy_path = Path(args.npy_path)
    out_dir  = Path(args.out_dir) if args.out_dir else npy_path.parent / (npy_path.stem + "_potree")
    out_dir.mkdir(parents=True, exist_ok=True)

    has_converter  = POTREE_CONVERTER.exists()
    has_prebuilt   = (out_dir / "metadata.json").exists() and (out_dir / "positions.bin").exists()

    if not has_converter and not has_prebuilt:
        sys.exit(f"PotreeConverter not found at {POTREE_CONVERTER} and no pre-built octree in {out_dir}.\n"
                 f"Run scripts/fetch_potree.sh for the viewer assets; build PotreeConverter\n"
                 f"from source as described in that script's header.")

    tokenizer = None
    if not args.no_tokenizer:
        from lift.models.tokenizer import load_tokenizer
        print("loading tokenizer...")
        tokenizer = load_tokenizer()

    if has_converter:
        pts = np.load(npy_path)
        if pts.ndim != 2 or pts.shape[1] != 3:
            sys.exit(f"Expected (N, 3) array, got shape {pts.shape}")
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        n   = pts.shape[0]
        title = npy_path.stem

        print(f"converting {n:,} points to LAS…")
        las_path = out_dir / f"{title}.las"
        npy_to_las(pts, las_path)

        print("building Potree octree…")
        build_octree(las_path, out_dir)
        las_path.unlink()   # clean up intermediate LAS

        print("writing positions.bin…")
        write_positions_bin(pts, out_dir)

        print("generating index.html…")
        generate_html(out_dir, n, npy_path.stem)
    else:
        print(f"PotreeConverter not found — using pre-built octree in {out_dir}")

    print(f"\nServing at http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    print("(VS Code Remote-SSH forwards the port automatically)\n")

    server = HTTPServer((args.host, args.port), make_server_class(out_dir, tokenizer))
    server.serve_forever()


if __name__ == "__main__":
    main()
