import argparse
import sys
from pathlib import Path

from img2vid.generate import DEFAULT_MODEL, GenerationError, generate_video


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img2vid",
        description="Generate a video from a text prompt (T2V), or a prompt + image (I2V), via LTX-2.5.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt describing the desired scene/motion")
    parser.add_argument("--image", default=None, help="Optional input image path (enables image-to-video)")
    parser.add_argument("--output", default=None, help="Output video path (default: outputs/generated.mp4)")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", "-f", type=int, default=97, help="Must satisfy (frames - 1) %% 8 == 0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--no-low-ram", action="store_true", help="Disable --low-ram (needs more unified memory)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = generate_video(
            args.prompt,
            image_path=Path(args.image) if args.image else None,
            output_path=Path(args.output) if args.output else None,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            frame_rate=args.frame_rate,
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
