"""为 PyInstaller onedir 目录生成确定性的 WiX v4/v5/v6 MSI 源文件。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import uuid


NS = "http://wixtoolset.org/schemas/v4/wxs"
UI_NS = "http://wixtoolset.org/schemas/v4/wxs/ui"
# Public desktop packages use a namespace distinct from the pre-release MSI
# prototypes. Those prototypes were installed during acceptance and one early
# package left stale Windows Installer component clients behind; reusing their
# GUIDs makes a later package report success while retaining shared files.
# Keep this namespace stable for every public 1.x upgrade from now on.
COMPONENT_NAMESPACE = uuid.UUID("8bd4aa26-91e1-4cfe-a8c4-207f7df86cd2")
REGISTRY_BASE = "Software\\AITIC\\Desktop\\PublicV1"
ET.register_namespace("", NS)
ET.register_namespace("ui", UI_NS)


def tag(name: str) -> str:
    return "{%s}%s" % (NS, name)


def ui_tag(name: str) -> str:
    return "{%s}%s" % (UI_NS, name)


def stable_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def stable_guid(value: str) -> str:
    return str(uuid.uuid5(COMPONENT_NAMESPACE, value)).upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--icon", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    dist = Path(args.dist).resolve()
    output = Path(args.output).resolve()
    icon = Path(args.icon).resolve()
    license_rtf = Path(args.license).resolve()
    main_exe = dist / "AITIC Desktop.exe"
    if not main_exe.is_file():
        raise SystemExit("missing packaged executable: %s" % main_exe)
    if not license_rtf.is_file():
        raise SystemExit("missing installer license: %s" % license_rtf)

    wix = ET.Element(tag("Wix"))
    package = ET.SubElement(wix, tag("Package"), {
        "Name": "AITIC Desktop", "Manufacturer": "AITIC", "Version": args.version,
        "Language": "2052", "UpgradeCode": "6D332707-13AE-4C93-A110-1FB8D59F4B71",
        "Scope": "perUser", "InstallerVersion": "500", "Compressed": "yes",
    })
    ET.SubElement(package, tag("MajorUpgrade"), {
        "DowngradeErrorMessage": "已安装更高版本的 AITIC Desktop。",
    })
    ET.SubElement(package, tag("MediaTemplate"), {
        "EmbedCab": "yes", "CompressionLevel": "high",
    })
    ET.SubElement(package, tag("Icon"), {"Id": "AITICIcon", "SourceFile": str(icon)})
    ET.SubElement(package, tag("Property"), {"Id": "ARPPRODUCTICON", "Value": "AITICIcon"})
    # WixUI_FeatureTree provides a normal wizard, destination browsing, feature
    # selection and installation progress. The desktop shortcut is a separate
    # default-on feature so users can opt out before copying files.
    ET.SubElement(package, ui_tag("WixUI"), {
        "Id": "WixUI_FeatureTree", "InstallDirectory": "INSTALLFOLDER",
    })
    ET.SubElement(package, tag("UIRef"), {"Id": "WixUI_ErrorProgressText"})
    ET.SubElement(package, tag("WixVariable"), {
        "Id": "WixUILicenseRtf", "Value": str(license_rtf),
    })
    ET.SubElement(package, tag("Property"), {
        "Id": "WIXUI_EXITDIALOGOPTIONALTEXT",
        "Value": "安装完成后，可从开始菜单或桌面快捷方式启动 AITIC Desktop。首次启动会引导配置模型。",
    })
    # Directory properties are recalculated from their defaults during
    # maintenance/uninstall. Persist the actual path chosen in the wizard and
    # restore it before CostFinalize, otherwise a custom-path uninstall cleans
    # the default directory and leaves the real program behind.
    install_folder_property = ET.SubElement(package, tag("Property"), {
        "Id": "INSTALLFOLDER",
    })
    ET.SubElement(install_folder_property, tag("RegistrySearch"), {
        "Id": "RememberedInstallFolder", "Root": "HKCU",
        "Key": REGISTRY_BASE, "Name": "InstallLocation", "Type": "raw",
    })

    local = ET.SubElement(package, tag("StandardDirectory"), {"Id": "LocalAppDataFolder"})
    # 程序文件与用户知识库必须分开。桌面后端把数据放在
    # %LOCALAPPDATA%\AITIC Desktop；若安装器也写入该目录，卸载时就可能
    # 删除用户建好的库。程序按常见的 per-user 布局安装到 Programs 子目录。
    programs = ET.SubElement(local, tag("Directory"), {
        "Id": "LocalProgramsFolder", "Name": "Programs",
    })
    install = ET.SubElement(programs, tag("Directory"), {
        "Id": "INSTALLFOLDER", "Name": "AITIC Desktop",
    })
    menu = ET.SubElement(package, tag("StandardDirectory"), {"Id": "ProgramMenuFolder"})
    ET.SubElement(menu, tag("Directory"), {"Id": "ProgramMenuDir", "Name": "AITIC Desktop"})
    desktop_dir = ET.SubElement(package, tag("StandardDirectory"), {"Id": "DesktopFolder"})

    directory_ids = {Path("."): "INSTALLFOLDER"}

    def ensure_directory(relative: Path) -> str:
        if relative in directory_ids:
            return directory_ids[relative]
        parent = ensure_directory(relative.parent)
        wanted = stable_id("dir_", relative.as_posix())
        # 找到父 Directory 节点并挂入子节点。
        parent_node = install if parent == "INSTALLFOLDER" else next(
            node for node in package.iter(tag("Directory")) if node.get("Id") == parent)
        ET.SubElement(parent_node, tag("Directory"), {"Id": wanted, "Name": relative.name})
        directory_ids[relative] = wanted
        return wanted

    files = sorted((path for path in dist.rglob("*") if path.is_file()),
                   key=lambda p: p.relative_to(dist).as_posix().casefold())
    component_ids = []
    for path in files:
        relative = path.relative_to(dist)
        directory_id = ensure_directory(relative.parent)
        ref = ET.SubElement(package, tag("DirectoryRef"), {"Id": directory_id})
        component_id = stable_id("cmp_", relative.as_posix())
        file_id = stable_id("fil_", relative.as_posix())
        component = ET.SubElement(ref, tag("Component"), {
            "Id": component_id, "Guid": stable_guid("file:" + relative.as_posix()),
        })
        ET.SubElement(component, tag("File"), {
            "Id": file_id, "Source": str(path), "Name": relative.name,
        })
        # LocalAppDataFolder 属于用户配置目录。Windows Installer 的 ICE38 要求
        # 其中每个组件以 HKCU 注册表值为 KeyPath；用文件作 KeyPath 会生成一个
        # 能构建却无法通过官方校验的 MSI。
        ET.SubElement(component, tag("RegistryValue"), {
            "Root": "HKCU", "Key": REGISTRY_BASE + "\\Components",
            "Name": component_id, "Type": "integer", "Value": "1", "KeyPath": "yes",
        })
        component_ids.append(component_id)

    install_location_ref = ET.SubElement(package, tag("DirectoryRef"), {
        "Id": "INSTALLFOLDER",
    })
    install_location_id = "InstallLocationRegistry"
    install_location_component = ET.SubElement(install_location_ref, tag("Component"), {
        "Id": install_location_id,
        "Guid": stable_guid("registry:install-location"),
    })
    ET.SubElement(install_location_component, tag("RegistryValue"), {
        "Root": "HKCU", "Key": REGISTRY_BASE, "Name": "InstallLocation",
        "Type": "string", "Value": "[INSTALLFOLDER]", "KeyPath": "yes",
    })
    component_ids.append(install_location_id)

    # Intermediate directories can contain only child directories and no file of their own.
    # A RemoveFolder attached to file components therefore cannot cover the full tree (ICE64).
    # Give every generated directory one deterministic cleanup component.
    for relative, directory_id in sorted(directory_ids.items(), key=lambda item: item[0].as_posix()):
        ref = ET.SubElement(package, tag("DirectoryRef"), {"Id": directory_id})
        relative_name = relative.as_posix()
        component_id = stable_id("drc_", relative_name)
        component = ET.SubElement(ref, tag("Component"), {
            "Id": component_id, "Guid": stable_guid("directory:" + relative_name),
        })
        ET.SubElement(component, tag("RegistryValue"), {
            "Root": "HKCU", "Key": REGISTRY_BASE + "\\Directories",
            "Name": component_id, "Type": "integer", "Value": "1", "KeyPath": "yes",
        })
        ET.SubElement(component, tag("RemoveFolder"), {
            "Id": stable_id("rmf_", relative_name), "On": "uninstall",
        })
        component_ids.append(component_id)

    # LocalProgramsFolder is another user-profile directory introduced above.
    # ICE64 requires every such directory to have an uninstall entry. RemoveFolder
    # removes it only when empty, so other per-user programs remain untouched.
    programs_ref = ET.SubElement(package, tag("DirectoryRef"), {"Id": "LocalProgramsFolder"})
    programs_cleanup_id = "LocalProgramsCleanup"
    programs_cleanup = ET.SubElement(programs_ref, tag("Component"), {
        "Id": programs_cleanup_id, "Guid": stable_guid("directory:local-programs"),
    })
    ET.SubElement(programs_cleanup, tag("RegistryValue"), {
        "Root": "HKCU", "Key": REGISTRY_BASE + "\\Directories",
        "Name": programs_cleanup_id, "Type": "integer", "Value": "1", "KeyPath": "yes",
    })
    ET.SubElement(programs_cleanup, tag("RemoveFolder"), {
        "Id": "RemoveLocalProgramsFolder", "On": "uninstall",
    })
    component_ids.append(programs_cleanup_id)

    shortcut_ref = ET.SubElement(package, tag("DirectoryRef"), {"Id": "ProgramMenuDir"})
    shortcut = ET.SubElement(shortcut_ref, tag("Component"), {
        "Id": "StartMenuShortcut", "Guid": stable_guid("shortcut:start-menu"),
    })
    ET.SubElement(shortcut, tag("Shortcut"), {
        "Id": "StartMenuShortcutLink", "Name": "AITIC Desktop",
        "Target": "[INSTALLFOLDER]AITIC Desktop.exe", "WorkingDirectory": "INSTALLFOLDER",
        "Icon": "AITICIcon",
    })
    ET.SubElement(shortcut, tag("RemoveFolder"), {"Id": "RemoveStartMenuDir", "On": "uninstall"})
    ET.SubElement(shortcut, tag("RegistryValue"), {
        "Root": "HKCU", "Key": REGISTRY_BASE, "Name": "StartMenuShortcut",
        "Type": "integer", "Value": "1", "KeyPath": "yes",
    })
    component_ids.append("StartMenuShortcut")

    desktop_component = ET.SubElement(
        desktop_dir, tag("Component"), {
            "Id": "DesktopShortcut", "Guid": stable_guid("shortcut:desktop"),
        })
    ET.SubElement(desktop_component, tag("Shortcut"), {
        "Id": "DesktopShortcutLink", "Name": "AITIC Desktop",
        "Target": "[INSTALLFOLDER]AITIC Desktop.exe", "WorkingDirectory": "INSTALLFOLDER",
        "Icon": "AITICIcon",
    })
    ET.SubElement(desktop_component, tag("RegistryValue"), {
        "Root": "HKCU", "Key": REGISTRY_BASE, "Name": "DesktopShortcut",
        "Type": "integer", "Value": "1", "KeyPath": "yes",
    })
    desktop_component_id = "DesktopShortcut"

    feature = ET.SubElement(package, tag("Feature"), {
        "Id": "MainFeature", "Title": "AITIC Desktop 主程序", "Level": "1",
        "Description": "原生桌面程序、检索运行时、Ollama 运行环境和开始菜单入口",
        "AllowAbsent": "no",
        "ConfigurableDirectory": "INSTALLFOLDER",
    })
    for component_id in component_ids:
        ET.SubElement(feature, tag("ComponentRef"), {"Id": component_id})
    desktop_feature = ET.SubElement(feature, tag("Feature"), {
        "Id": "DesktopShortcutFeature", "Title": "桌面快捷方式", "Level": "1",
        "Description": "在当前用户桌面创建 AITIC Desktop 快捷方式（可取消）",
    })
    ET.SubElement(desktop_feature, tag("ComponentRef"), {"Id": desktop_component_id})

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(wix, space="  ")
    ET.ElementTree(wix).write(output, encoding="utf-8", xml_declaration=True)
    print("WIX_FILES=%d" % len(files))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
