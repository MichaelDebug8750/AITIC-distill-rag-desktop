"""Check the generated WiX source exposes the public installer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


NS = {"w": "http://wixtoolset.org/schemas/v4/wxs",
      "ui": "http://wixtoolset.org/schemas/v4/wxs/ui"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wxs", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = ET.parse(Path(args.wxs).resolve()).getroot()
    package = root.find("w:Package", NS)
    assert package is not None
    ui = package.find("ui:WixUI", NS)
    main_feature = package.find("w:Feature[@Id='MainFeature']", NS)
    desktop = package.find(".//w:Feature[@Id='DesktopShortcutFeature']", NS)
    start_link = package.find(".//w:Shortcut[@Id='StartMenuShortcutLink']", NS)
    desktop_link = package.find(".//w:Shortcut[@Id='DesktopShortcutLink']", NS)
    license_var = package.find("w:WixVariable[@Id='WixUILicenseRtf']", NS)
    progress_ref = package.find("w:UIRef[@Id='WixUI_ErrorProgressText']", NS)
    report = {
        "ok": True,
        "wizard": ui.get("Id") if ui is not None else "",
        "install_directory": ui.get("InstallDirectory") if ui is not None else "",
        "main_configurable_directory": (
            main_feature.get("ConfigurableDirectory") if main_feature is not None else ""),
        "main_required": main_feature.get("AllowAbsent") if main_feature is not None else "",
        "desktop_default_on": desktop.get("Level") if desktop is not None else "",
        "desktop_separate_feature": desktop is not None,
        "start_menu_shortcut": start_link is not None,
        "desktop_shortcut": desktop_link is not None,
        "license_dialog_source": license_var is not None,
        "progress_text": progress_ref is not None,
        "per_user": package.get("Scope"),
    }
    report["ok"] = all((
        report["wizard"] == "WixUI_FeatureTree",
        report["install_directory"] == "INSTALLFOLDER",
        report["main_configurable_directory"] == "INSTALLFOLDER",
        report["main_required"] == "no",
        report["desktop_default_on"] == "1",
        report["desktop_separate_feature"], report["start_menu_shortcut"],
        report["desktop_shortcut"], report["license_dialog_source"],
        report["progress_text"], report["per_user"] == "perUser",
    ))
    target = Path(args.report).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
