"""Static contract for the public bilingual Inno Setup installer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iss", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    text = Path(args.iss).resolve().read_text(encoding="utf-8")
    checks = {
        "modern_dynamic_ui": "WizardStyle=modern dynamic windows11 includetitlebar" in text,
        "chinese_language": 'Name: "zhcn"' in text and "ChineseSimplified.isl" in text,
        "english_language": 'Name: "en"' in text and "Default.isl" in text,
        "language_dialog": "ShowLanguageDialog=yes" in text,
        "destination_page": "DisableDirPage=no" in text,
        "start_menu_page": "DisableProgramGroupPage=no" in text,
        "start_menu_optional": "AllowNoIcons=yes" in text,
        "desktop_optional_default_on": (
            'Name: "desktopicon"' in text and "Flags: unchecked" not in text
        ),
        "ready_page": "DisableReadyPage=no" in text,
        "progress_page": "[Files]" in text,
        "launch_after_install": "postinstall" in text,
        "per_user_default": "DefaultDirName={localappdata}\\Programs\\{#AppName}" in text,
        "x64_installer": "SetupArchitecture=x64" in text,
    }
    report = {"ok": all(checks.values()), "checks": checks}
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
