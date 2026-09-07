import argparse
import sys
from pathlib import Path

from img2vid.generate import DEFAULT_MODEL, GenerationError, generate_video


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="img2vid", description="Generate a video from an image + prompt.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the desired motion/scene")
    parser.add_argument("--output", default=None, help="Output video path (default: outputs/generated.mp4)")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-low-ram", action="store_true", help="Disable --low-ram (needs more unified memory)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = generate_video(
            Path(args.image),
            args.prompt,
            output_path=Path(args.output) if args.output else None,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            guidance=args.guidance,
            flow_shift=args.flow_shift,
            fps=args.fps,
            seed=args.seed,
            model=args.model,
            low_ram=not args.no_low_ram,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except GenerationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
