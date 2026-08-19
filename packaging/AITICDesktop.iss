#define AppName "AITIC Desktop"
#define AppVersion "1.0.0"
#ifndef AppIdValue
  #define AppIdValue "{{8BD4AA26-91E1-4CFE-A8C4-207F7DF86CD2}"
#endif
#ifndef OutputBaseName
  #define OutputBaseName "AITIC-Desktop-1.0.0-Setup-x64"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\AITIC Desktop"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

[Setup]
AppId={#AppIdValue}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=AITIC
AppCopyright=Copyright (C) 2026 AITIC
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
VersionInfoDescription=AITIC Desktop bilingual installer
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
DisableDirPage=no
DisableProgramGroupPage=no
DisableWelcomePage=yes
DisableReadyPage=no
DisableReadyMemo=no
DisableFinishedPage=no
ShowLanguageDialog=yes
LanguageDetectionMethod=uilanguage
UsePreviousAppDir=yes
UsePreviousGroup=yes
UsePreviousTasks=yes
UsePreviousLanguage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
SetupArchitecture=x64
MinVersion=10.0.19045
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseName}
SetupIconFile=aitic.ico
UninstallDisplayIcon={app}\AITIC Desktop.exe
UninstallDisplayName={#AppName}
WizardStyle=modern dynamic windows11 includetitlebar
WizardResizable=yes
WizardSizePercent=125,120
WizardKeepAspectRatio=yes
WizardSmallImageFile=generated\aitic_setup_64.png,generated\aitic_setup_96.png,generated\aitic_setup_128.png,generated\aitic_setup_160.png,generated\aitic_setup_192.png
WizardSmallImageFileDynamicDark=generated\aitic_setup_64.png,generated\aitic_setup_96.png,generated\aitic_setup_128.png,generated\aitic_setup_160.png,generated\aitic_setup_192.png
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
InternalCompressLevel=max
DiskSpanning=no
SetupLogging=yes
CloseApplications=yes
RestartApplications=no
RestartIfNeededByRun=no
ChangesAssociations=no
ChangesEnvironment=no

[Languages]
Name: "zhcn"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
zhcn.DesktopShortcut=创建桌面快捷方式
zhcn.ShortcutGroup=附加快捷方式：
zhcn.LaunchAITIC=立即运行 AITIC Desktop
zhcn.ModelGuide=模型配置教程
en.DesktopShortcut=Create a desktop shortcut
en.ShortcutGroup=Additional shortcuts:
en.LaunchAITIC=Launch AITIC Desktop
en.ModelGuide=Model setup guide

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopShortcut}"; GroupDescription: "{cm:ShortcutGroup}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AITIC Desktop"; Filename: "{app}\AITIC Desktop.exe"; WorkingDir: "{app}"; IconFilename: "{app}\AITIC Desktop.exe"
Name: "{group}\{cm:ModelGuide}"; Filename: "{app}\MODEL_SETUP_GUIDE.zh-CN.md"; WorkingDir: "{app}"
Name: "{autodesktop}\AITIC Desktop"; Filename: "{app}\AITIC Desktop.exe"; WorkingDir: "{app}"; IconFilename: "{app}\AITIC Desktop.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AITIC Desktop.exe"; Description: "{cm:LaunchAITIC}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKA; Subkey: "Software\AITIC\Desktop\SetupV1"; ValueType: string; ValueName: "InstallLocation"; ValueData: "{app}"; Flags: uninsdeletekey
