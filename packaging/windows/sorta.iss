; F211 - the Windows installer: the base tier whole, everything heavier offered once.
;
; Every variable part of this script arrives as a /D define from
; scripts/build_installer.py (Version, PayloadDir, OutputDir), so this file holds no
; path of the machine that built it and no version number to keep in sync. The
; defaults below exist only so that opening the script in the Inno IDE does not fail;
; a real build always passes all three (the pairing is pinned by the suite).
;
; What it does, and the three things it deliberately does NOT do:
;
;   * copies the payload to a per-user directory - no administrator, no UAC prompt;
;   * puts `config.example.yaml` in place as `config.yaml`, and never over one that
;     is already there;
;   * one shortcut that starts the tray (pythonw, so no console window) and one that
;     re-runs the setup wizard;
;   * NO autostart - a program that starts with the machine is the owner's decision;
;   * NO signing here - it is a separate, opt-in step of the build script (the owner's
;     decision of 2026-08-06: this release ships unsigned, and the download page warns
;     about SmartScreen instead);
;   * NO deleting of anybody's data on uninstall - the run log and the preview cache
;     live beside each other and carry this machine's measurements.

#ifndef Version
  #define Version "0.0.0"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\..\dist\windows\payload"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist\windows"
#endif

#define AppName "Sorta"
#define AppPublisher "shinKatana0"
#define AppUrl "https://github.com/shinKatana0/sorta"

[Setup]
AppId={{7A3B6C21-4E58-4C7E-9E3B-0F9D2B4A1C55}
AppName={#AppName}
AppVersion={#Version}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
VersionInfoVersion={#Version}
; Per-user by default: {autopf} under `lowest` is %LOCALAPPDATA%\Programs, which is
; what the user guide documents. An installer that asked for administrator rights to
; sort somebody's photographs would be asking for something it does not need.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=sorta-{#Version}-setup
; The payload is already mostly compressed wheels; lzma2/max buys little and costs
; minutes on every build.
Compression=lzma2/fast
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; F207 again: three pictures of one program read as three programs, so the setup, the
; entry in Programs and Features and the tray icon are one file.
SetupIconFile=..\..\sorta\web\favicon.ico
UninstallDisplayIcon={app}\favicon.ico
LicenseFile=..\..\LICENSE
WizardStyle=modern

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"

[Dirs]
; `{userappdata}\sorta` is where config.yaml and the index live, and it is made here
; rather than by the first run: both shortcuts name it as their working directory, and
; a shortcut whose working directory does not exist starts in C:\Windows\system32 and
; writes the index there.
Name: "{userappdata}\sorta"

[Files]
; The payload as it was staged: the interpreter, `lib`, uv.exe, exiftool, the icon,
; the licence and the install manifest the wizard reads.
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; The default profile is PUT IN PLACE rather than asked about - and it is
; `config.example.yaml` itself, not a second copy of its defaults that would drift
; away from it by the next release. `onlyifdoesntexist` + `uninsneveruninstall`:
; a reinstall must not overwrite an edited config, and uninstalling a program is not
; a request to delete somebody's settings.
Source: "{#PayloadDir}\config.example.yaml"; DestDir: "{userappdata}\sorta"; DestName: "config.yaml"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
; The one a person clicks: pythonw.exe, so the web app opens with a tray icon and no
; console window anywhere behind it (which is what F207 was for).
Name: "{group}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m sorta.tray"; WorkingDir: "{userappdata}\sorta"; IconFilename: "{app}\favicon.ico"
; ...and the way a refused tier gets added later. A console this time: the wizard is a
; conversation. `-X utf8` because its catalog is Russian, English and Japanese and a
; Windows console still runs on a legacy code page unless it is told otherwise.
Name: "{group}\{#AppName} setup"; Filename: "{app}\python\python.exe"; Parameters: "-X utf8 -m sorta.wizard"; WorkingDir: "{userappdata}\sorta"; IconFilename: "{app}\favicon.ico"

[Run]
; The first-run wizard, once, at the end of the installation. `skipifsilent` because a
; silent install has nobody to answer its questions: it would block on `input()` with
; no keyboard behind the console, which is exactly what the unattended install of the
; installer workflow does.
Filename: "{app}\python\python.exe"; Parameters: "-X utf8 -m sorta.wizard"; WorkingDir: "{userappdata}\sorta"; Description: "Set up Sorta (adds optional tiers)"; Flags: postinstall skipifsilent
