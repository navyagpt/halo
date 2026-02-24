"""Command-line interface for the end-to-end surgical tool pipeline."""

from __future__ import annotations

import argparse
import os
from typing import List

from .config import (
    AudioConfig,
    ClassifierConfig,
    DEFAULT_EXTENSIONS,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DetectorConfig,
    PipelineConfig,
    RobotConfig,
)
from .helpers import parse_candidate_labels, parse_json_dict_arg
from .pipeline import run_pipeline


def create_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI arguments for image, audio, classifier, and robot settings."""
    parser = argparse.ArgumentParser(
        description="Run self-contained audio-guided surgical instrument localization pipeline."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Manual input image/folder path. Required when --robot is false. "
            "When --robot is true, this can be a fallback path."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output root for crops, visualizations, and merged reports.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input folder.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="File extensions for folder scanning (default: common image formats).",
    )
    parser.add_argument("--save_masks", action="store_true", help="Save ROI and object masks.")
    parser.add_argument("--crop_pad", type=int, default=0, help="Extra crop padding in pixels.")

    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Saved classifier directory (copied locally under surgical_tool_pipeline/models/best_model).",
    )
    parser.add_argument(
        "--model_id_override",
        default=None,
        help="Override backbone model id (default comes from model_dir/config.json).",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Crop classification batch size.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Dataloader workers for crop classification.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device selection.",
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP even on CUDA.")
    parser.add_argument(
        "--candidate_labels",
        nargs="+",
        default=None,
        help=(
            "Optional label subset to predict among. "
            "Accepts space-separated and/or comma-separated labels."
        ),
    )
    parser.add_argument(
        "--audio_input",
        default=None,
        help="Optional audio path used to infer target instrument before image matching.",
    )
    parser.add_argument(
        "--audio_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Audio ASR device selection.",
    )
    parser.add_argument(
        "--audio_model_id",
        type=str,
        default="google/medasr",
        help="ASR model id for audio instrument extraction (default: google/medasr).",
    )
    parser.add_argument(
        "--audio_chunk_length_s",
        type=float,
        default=0.0,
        help="Optional chunk length in seconds for audio ASR (0 = disabled).",
    )
    parser.add_argument(
        "--audio_stride_length_s",
        type=float,
        default=None,
        help="Optional stride length in seconds for audio ASR chunking.",
    )

    parser.add_argument(
        "--robot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, call robot API to obtain input source before running pipeline.",
    )
    parser.add_argument(
        "--robot_api_url",
        type=str,
        default=None,
        help="Robot API URL (required when --robot is true).",
    )
    parser.add_argument(
        "--robot_api_method",
        type=str,
        default="POST",
        choices=["GET", "POST"],
        help="Robot API method for --robot mode.",
    )
    parser.add_argument(
        "--robot_timeout_sec",
        type=float,
        default=10.0,
        help="Robot API timeout in seconds.",
    )
    parser.add_argument(
        "--robot_payload_json",
        type=str,
        default=None,
        help="Optional JSON payload string for robot API POST requests.",
    )
    parser.add_argument(
        "--robot_response_image_key",
        type=str,
        default="image_path",
        help="Robot API JSON field containing a path to the image or image folder.",
    )
    parser.add_argument(
        "--robot_response_image_list_key",
        type=str,
        default="image_paths",
        help="Robot API JSON field containing a list of image paths (first item used).",
    )

    parser.add_argument("--v_dark", type=int, default=None, help="ROI dark threshold (manual override).")
    parser.add_argument("--roi_close_k", type=int, default=31)
    parser.add_argument("--roi_open_k", type=int, default=17)
    parser.add_argument("--delta_v", type=int, default=35)
    parser.add_argument("--s_min", type=int, default=25)
    parser.add_argument("--obj_close_k", type=int, default=23)
    parser.add_argument("--obj_open_k", type=int, default=7)
    parser.add_argument("--min_area", type=int, default=None)
    parser.add_argument("--max_area_frac", type=float, default=0.60)
    parser.add_argument("--max_aspect", type=float, default=20.0)
    parser.add_argument(
        "--split_merged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable conservative split for merged components.",
    )
    parser.add_argument("--split_erode_max_iter", type=int, default=8)
    parser.add_argument(
        "--auto_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable auto-thresholding for the classical CV detector.",
    )
    return parser


def namespace_to_config(args: argparse.Namespace) -> PipelineConfig:
    """Map parsed CLI arguments into strongly-typed pipeline configuration."""
    detector = DetectorConfig(
        v_dark=int(args.v_dark) if args.v_dark is not None else 110,
        roi_close_k=int(args.roi_close_k),
        roi_open_k=int(args.roi_open_k),
        delta_v=int(args.delta_v),
        s_min=int(args.s_min),
        obj_close_k=int(args.obj_close_k),
        obj_open_k=int(args.obj_open_k),
        min_area=args.min_area,
        max_area_frac=float(args.max_area_frac),
        max_aspect=float(args.max_aspect),
        split_merged=bool(args.split_merged),
        split_erode_max_iter=int(args.split_erode_max_iter),
        auto_thresholds=bool(args.auto_thresholds),
        manual_v_dark_override=(args.v_dark is not None),
        crop_pad=int(args.crop_pad),
    )

    classifier = ClassifierConfig(
        model_dir=str(args.model_dir),
        model_id_override=args.model_id_override,
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        use_amp=(not bool(args.no_amp)),
        candidate_labels=parse_candidate_labels(args.candidate_labels),
    )

    robot = RobotConfig(
        enabled=bool(args.robot),
        api_url=args.robot_api_url,
        api_method=str(args.robot_api_method),
        timeout_sec=float(args.robot_timeout_sec),
        payload=parse_json_dict_arg(args.robot_payload_json, "--robot_payload_json"),
        response_image_key=str(args.robot_response_image_key),
        response_image_list_key=str(args.robot_response_image_list_key),
    )
    audio = AudioConfig(
        enabled=bool(args.audio_input),
        input_path=args.audio_input,
        model_id=str(args.audio_model_id),
        device=str(args.audio_device),
        chunk_length_s=float(args.audio_chunk_length_s),
        stride_length_s=(float(args.audio_stride_length_s) if args.audio_stride_length_s is not None else None),
    )

    return PipelineConfig(
        input_path=args.input,
        output_dir=str(args.output_dir),
        recursive=bool(args.recursive),
        extensions=[str(x) for x in args.extensions],
        save_masks=bool(args.save_masks),
        detector=detector,
        classifier=classifier,
        robot=robot,
        audio=audio,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments from argv or sys.argv when omitted."""
    parser = create_arg_parser()
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    config = namespace_to_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
