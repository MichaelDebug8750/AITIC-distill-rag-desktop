"""Static contract: every business endpoint exposed by WebUI has a native route."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code")]

from desktop_app.backend import DesktopBackend
from desktop_app.main_window import MainWindow


ROUTE_TO_NATIVE = {
    "api_status": "status",
    "api_ask": "ask",
    "api_ask_stream": "ask_stream",
    "api_ask_stream_post": "ask_stream",
    "api_brief": "brief",
    "api_questions": "questions",
    "api_feedback": "feedback",
    "api_feedback_list": "feedback_list",
    "api_feedback_rerun": "rerun_feedback",
    "api_feedback_regression": "export_regression",
    "api_library_health": "library_health",
    "api_concept": "concept",
    "api_compare": "compare",
    "api_batch": "batch",
    "api_source_page": "source_page_png",
    "api_source": "source_blocks",
    "api_retrieve_only": "retrieve",
    "api_library_chunks": "library_chunks",
    "api_libraries": "libraries",
    "api_activate_library": "activate_library",
    "api_build_job": "build_status",
    "api_active_build": "active_build",
    "api_build": "start_build",
    "api_upload": "start_build",
}


def route_functions() -> set[str]:
    tree = ast.parse((ROOT / "code" / "webui.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            attr = call.func if call and isinstance(call.func, ast.Attribute) else None
            if attr and isinstance(attr.value, ast.Name) and attr.value.id == "app":
                names.add(node.name)
    return names


def main() -> int:
    routes = route_functions()
    business_routes = {name for name in routes if name.startswith("api_")}
    missing_map = sorted(business_routes - set(ROUTE_TO_NATIVE))
    missing_methods = sorted(
        "%s -> %s" % (route, method) for route, method in ROUTE_TO_NATIVE.items()
        if route in business_routes and not hasattr(DesktopBackend, method))
    required_ui = {
        "_choose_and_build", "_resume_active_build", "_open_retrieval_settings",
        "_toggle_focus_mode", "_show_session_metrics", "_choose_materials",
        "_copy_conversation", "_export_conversation", "_run_brief", "_run_quiz",
        "_run_concept", "_run_compare", "_run_batch", "_run_retrieve",
        "_refresh_feedback", "_rerun_feedback", "_export_feedback",
        "_append_rich_answer", "_rich_answer_block", "_open_chat_link",
        "_copy_latest_answer", "_toggle_latest_favorite", "_regenerate_latest",
    }
    missing_ui = sorted(name for name in required_ui if not hasattr(MainWindow, name))
    if missing_map or missing_methods or missing_ui:
        raise AssertionError({"missing_map": missing_map, "missing_methods": missing_methods,
                              "missing_ui": missing_ui, "routes": sorted(business_routes)})
    print("WEB_BUSINESS_ROUTES=%d" % len(business_routes))
    print("NATIVE_BACKEND_PARITY=PASS")
    print("NATIVE_UI_PARITY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
