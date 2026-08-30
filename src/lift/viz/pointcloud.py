"""
Fast interactive 3D point cloud viewer for large (N, 3) .npy arrays.

GPU-accelerated via PyVista/VTK -- handles hundreds of thousands of points
smoothly (matplotlib's 3D scatter cannot). Supports orbit/pan/zoom with the
mouse, and left-click on a point to show its index in the original array,
right there in the view (and optionally dump picks to a file).

By default points are colored by their index in the array using the
'turbo' colormap; pass --color/--color-axis to color by something else.

Since these arrays are usually per-vocab-token embeddings (row index ==
token id, see gemma4.get_contextual_embeddings), the gemma tokenizer is
loaded by default so a click also shows the decoded token. Pass
--no-tokenizer to skip this (e.g. for non-token point clouds).

Works in two modes:
  native  - opens a regular window. Use this on a laptop with a display.
  web     - serves an interactive view in a browser via trame. Use this on
            a remote server with no display: the points are rendered
            server-side and streamed as images, so it stays fast even over
            an SSH connection. With VS Code Remote-SSH the port forwards
            automatically; otherwise add `-L 8080:localhost:8080` to your
            ssh command and open http://localhost:8080 locally.

By default the mode is auto-detected from the $DISPLAY environment variable.

Usage:
    python view_pointcloud.py points.npy
    python view_pointcloud.py points.npy --mode web --port 8080
    python view_pointcloud.py points.npy --color points.npy --color-axis 2
    python view_pointcloud.py points.npy --color colors.npy   # (N,) or (N,3)
    python view_pointcloud.py points.npy --point-size 3 --save-picks picks.txt

Controls:
    left-drag        orbit
    right-drag       zoom
    middle-drag      pan
    left-click       pick nearest point, print its index + coords
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyvista as pv


def load_points(path: str) -> np.ndarray:
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) array, got shape {pts.shape}")
    return np.ascontiguousarray(pts, dtype=np.float32)


def build_scalars(pts: np.ndarray, color_path: str | None, color_axis: int | None):
    n = pts.shape[0]
    if color_path is not None:
        color_arr = np.load(color_path)
        if color_arr.shape == (n, 3):
            rgb_arr = color_arr.astype(np.float32)
            if rgb_arr.max() <= 1.0:
                rgb_arr = rgb_arr * 255.0
            return np.ascontiguousarray(rgb_arr.clip(0, 255), dtype=np.uint8), True
        elif color_arr.shape == (n,):
            return color_arr, False
        else:
            raise ValueError(f"--color array shape {color_arr.shape} doesn't match (N,) or (N,3) for N={n}")
    elif color_axis is not None:
        return pts[:, color_axis], False
    return np.arange(n), False


def build_plotter(pts: np.ndarray, args, off_screen: bool, picks_file, tokenizer) -> pv.Plotter:
    cloud = pv.PolyData(pts)
    scalars, rgb = build_scalars(pts, args.color, args.color_axis)

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(
        cloud,
        scalars=scalars,
        rgb=rgb,
        cmap=None if rgb else args.cmap,
        point_size=args.point_size,
        render_points_as_spheres=True,
        show_scalar_bar=scalars is not None and not rgb,
    )
    plotter.add_axes()

    def on_pick(point, picker):
        idx = picker.GetPointId()
        if idx is None or idx < 0:
            return
        token_part = ""
        if tokenizer is not None:
            try:
                token_part = f"  token={tokenizer.decode([idx])!r}"
            except Exception as e:
                token_part = f"  token=<decode error: {e}>"
        msg = f"index {idx}{token_part}    ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})"
        print(f"picked {msg}")
        if picks_file:
            picks_file.write(f"{idx} {point[0]} {point[1]} {point[2]}\n")
            picks_file.flush()
        plotter.add_text(msg, position="lower_left", font_size=16, color="yellow",
                          shadow=True, name="pick_label")
        plotter.render()

    plotter.enable_point_picking(
        callback=on_pick,
        picker="point",
        use_picker=True,
        left_clicking=True,
        tolerance=0.04,
        show_message="Click a point to print its index",
        show_point=True,
        point_size=args.point_size * 6,
        color="red",
        render_points_as_spheres=True,
    )
    return plotter


def run_native(plotter: pv.Plotter, title: str):
    plotter.show(title=title)


def run_web(plotter: pv.Plotter, title: str, host: str, port: int):
    import logging
    logging.getLogger("aiohttp.server").setLevel(logging.CRITICAL)

    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import vuetify3
    from pyvista.trame.ui import plotter_ui

    server = get_server()
    server.client_type = "vue3"
    state = server.state
    state.trame__title = title

    with SinglePageLayout(server) as layout:
        layout.title.set_text(title)
        with layout.content:
            with vuetify3.VContainer(fluid=True, classes="pa-0 fill-height"):
                plotter_ui(plotter, mode="server")

    print(f"\nServing at http://{host}:{port}  (forward this port if connecting over SSH)\n")
    server.start(host=host, port=port, open_browser=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npy_path", help="path to (N, 3) .npy point cloud")
    parser.add_argument("--color", default=None, help="path to .npy used for coloring: (N,) scalars or (N,3) rgb")
    parser.add_argument("--color-axis", type=int, default=None, choices=[0, 1, 2],
                         help="color by one coordinate of the point cloud itself (0=x,1=y,2=z)")
    parser.add_argument("--cmap", default="turbo", help="matplotlib colormap name for scalar coloring")
    parser.add_argument("--point-size", type=float, default=5.0, help="rendered point size in pixels")
    parser.add_argument("--save-picks", default=None, help="if set, append every picked (index, x, y, z) to this file")
    parser.add_argument("--mode", choices=["auto", "native", "web"], default="auto",
                         help="auto picks 'web' when $DISPLAY is unset (e.g. SSH), else 'native'")
    parser.add_argument("--host", default="0.0.0.0", help="bind host for --mode web")
    parser.add_argument("--port", type=int, default=8080, help="bind port for --mode web")
    parser.add_argument("--no-tokenizer", action="store_true",
                         help="don't load the tokenizer / decode picked indices as tokens")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "native" if os.environ.get("DISPLAY") else "web"

    pts = load_points(args.npy_path)
    n = pts.shape[0]
    print(f"loaded {n:,} points from {args.npy_path}  [mode={mode}]")

    tokenizer = None
    if not args.no_tokenizer:
        from lift.models.tokenizer import load_tokenizer
        print("loading tokenizer...")
        tokenizer = load_tokenizer()

    picks_file = open(args.save_picks, "a") if args.save_picks else None
    title = f"{Path(args.npy_path).name}  ({n:,} points)"

    plotter = build_plotter(pts, args, off_screen=(mode == "web"), picks_file=picks_file, tokenizer=tokenizer)

    if mode == "native":
        run_native(plotter, title)
    else:
        run_web(plotter, title, args.host, args.port)

    if picks_file:
        picks_file.close()


if __name__ == "__main__":
    main()
