"""Robot API client helpers for resolving runtime input image paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib import error as urllib_error, request as urllib_request

from .config import RobotConfig


def call_robot_api(robot: RobotConfig) -> Dict[str, Any]:
    """Call the configured robot API endpoint and normalize its JSON response."""
    method_upper = robot.api_method.strip().upper()
    if method_upper not in {"GET", "POST"}:
        raise ValueError("robot_api_method must be GET or POST.")
    if not robot.api_url:
        raise ValueError("robot_api_url is required when robot mode is enabled.")

    body: bytes | None = None
    headers: Dict[str, str] = {"Accept": "application/json"}
    if method_upper == "POST":
        body = json.dumps(robot.payload or {}).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(
        url=robot.api_url,
        data=body,
        headers=headers,
        method=method_upper,
    )

    try:
        with urllib_request.urlopen(req, timeout=float(robot.timeout_sec)) as resp:
            status = int(getattr(resp, "status", resp.getcode()))
            text = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        raise RuntimeError(f"Robot API HTTP error: {exc.code} {exc.reason}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Robot API connection error: {exc.reason}") from exc

    parsed: Any
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        parsed = {"raw_response": text}

    if not isinstance(parsed, dict):
        parsed = {"response": parsed}

    parsed["_http_status"] = status
    parsed["_api_url"] = robot.api_url
    parsed["_api_method"] = method_upper
    return parsed


def resolve_input_path(robot: RobotConfig, manual_input_path: str | None) -> Tuple[Path, Dict[str, Any] | None]:
    """Resolve final input path from robot mode or manual CLI/API input."""
    robot_meta: Dict[str, Any] | None = None
    selected_input = manual_input_path

    if robot.enabled:
        robot_meta = call_robot_api(robot)
        print(f"[robot] api_call_ok status={robot_meta.get('_http_status')} url={robot.api_url}")

        if robot.response_image_key in robot_meta and robot_meta[robot.response_image_key]:
            selected_input = str(robot_meta[robot.response_image_key])
            print(f"[robot] using path from response key '{robot.response_image_key}'")
        else:
            response_list = robot_meta.get(robot.response_image_list_key)
            if isinstance(response_list, list) and response_list:
                selected_input = str(response_list[0])
                print(f"[robot] using first path from response key '{robot.response_image_list_key}'")
            elif selected_input:
                print("[robot] response had no image path; using --input fallback")
            else:
                raise ValueError(
                    "Robot response did not provide an image path and --input fallback is missing."
                )
    else:
        if not selected_input:
            raise ValueError("--input is required when --robot is false.")

    input_path = Path(str(selected_input)).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return input_path.resolve(), robot_meta
