"""Command line entry.

    python -m shogunbot collect               save settled game frames (for perception work)
    python -m shogunbot selfplay -n 50        offline bench: planner vs simulator
    python -m shogunbot report                cross-run report -> runs/report.html
    python -m shogunbot review RUN_ID [--last N]
    python -m shogunbot play [--no-autostart] [--stage1 | --full]   play (hold F8 to stop)
    python -m shogunbot watch [--run ID]      live decision viewer on http://127.0.0.1:8765
    python -m shogunbot replay collect/<ts>   score perception + rules on a recorded run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shogunbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    sp = sub.add_parser("selfplay")
    sp.add_argument("-n", type=int, default=50)
    sp.add_argument("--depth", type=int, default=0, help="0 = adaptive")
    sub.add_parser("report")
    rv = sub.add_parser("review")
    rv.add_argument("run_id")
    rv.add_argument("--last", type=int, default=None)
    pl = sub.add_parser("play")
    pl.add_argument("--runs", type=int, default=1)
    pl.add_argument("--frames", choices=["events", "all", "none"], default="events")
    pl.add_argument("--no-autostart", action="store_true",
                    help="do not click through title/camp/map; start the run yourself")
    mode = pl.add_mutually_exclusive_group()
    mode.add_argument("--stage1", dest="stage1", action="store_true", default=True,
                      help="end each run after Stage 1's boss (the default until the map past Stage 1 works)")
    mode.add_argument("--full", dest="stage1", action="store_false",
                      help="go on after Stage 1's boss (M4, in progress)")
    wt = sub.add_parser("watch", help="live decision viewer in the browser (local, zero tokens)")
    wt.add_argument("--run", default=None, help="a run id; default: follow the newest run")
    wt.add_argument("--port", type=int, default=8765)
    wt.add_argument("--no-browser", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("run_id", nargs="?")
    rp = sub.add_parser("replay")
    rp.add_argument("folder", help="a collect/<timestamp> folder")
    args = ap.parse_args(argv)

    if args.cmd == "collect":
        from .collect import run
        run()
    elif args.cmd == "selfplay":
        from .selfplay import bench
        print(bench(args.n, args.depth))
    elif args.cmd == "report":
        from .report import report
        print(report(Path("runs")))
    elif args.cmd == "review":
        from .report import review
        print(review(Path("runs"), args.run_id, args.last))
    elif args.cmd == "play":
        from .bot import play
        play(runs=args.runs, frame_policy=args.frames, autostart=not args.no_autostart, stage1=args.stage1)
    elif args.cmd == "watch":
        from .watch import serve
        serve("runs", args.run, args.port, open_browser=not args.no_browser)
    elif args.cmd == "analyze":
        from .analyze import main as amain
        amain([args.run_id] if args.run_id else [])
    elif args.cmd == "replay":
        from .replay import replay
        replay(args.folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
