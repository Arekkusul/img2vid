import argparse
import sys
from pathlib import Path

from img2vid.image_generate import DEFAULT_SNAPSHOT_DIR, ImageGenerationError, generate_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img2vid-image", description="Generate or edit an image with Krea 2."
    )
    parser.add_argument("--prompt", required=True, help="Prompt (or edit instruction in edit mode)")
    parser.add_argument("--output", default=None, help="Output image path (default: outputs/krea_generated.png)")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=52)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edit-source", default=None, help="Reference image path (enables edit mode)")
    parser.add_argument("--lora", default=None, help="Identity-edit LoRA safetensors path (edit mode only)")
    parser.add_argument("--turbo-edit", action="store_true", help="Use the distilled CFG-free Turbo edit path")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output_path = generate_image(
            args.prompt,
            output_path=Path(args.output) if args.output else None,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            edit_image_path=Path(args.edit_source) if args.edit_source else None,
            lora_path=Path(args.lora) if args.lora else None,
            turbo_edit=args.turbo_edit,
            snapshot=Path(args.snapshot),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ImageGenerationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
