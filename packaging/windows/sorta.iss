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
;   * NO deleting of anybody's data on uninstall WITHOUT BEING ASKED - and, since F224,
;     it does ask. Uninstalling used to leave %APPDATA%\sorta, %LOCALAPPDATA%\sorta and
;     gigabytes of model weights in caches named after nobody; now the uninstaller
;     states what each of those weighs and offers them as two separate ticks, both
;     unticked. A silent uninstall (/VERYSILENT) asks nothing and therefore deletes
;     nothing: silence is not consent, and the installer workflow runs exactly that.
;
; This file is UTF-8 WITH a BOM, and it has to stay that way: Inno 6 reads a script
; without one as ANSI, and the uninstall page below is written in the same three
; languages as the rest of the product.

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

; F224 - the words of the uninstall page, in the three languages of the product.
[CustomMessages]
en.CleanupCaption=Uninstalling Sorta
ru.CleanupCaption=Удаление Sorta
ja.CleanupCaption=Sorta のアンインストール
en.CleanupIntro=Sorta is about to be removed. Its data and the models it downloaded stay on this disk unless you say otherwise. Nothing is deleted unless you tick it, and your photographs are never touched.
ru.CleanupIntro=Sorta сейчас будет удалена. Её данные и скачанные модели останутся на диске, если не сказать иначе. Ничего не удаляется без вашей отметки, а ваши фотографии не трогаются никогда.
ja.CleanupIntro=これから Sorta を削除します。データとダウンロード済みモデルは、指定しないかぎりディスクに残ります。チェックしないかぎり何も削除されず、あなたの写真に手を触れることはありません。
en.CleanupData=Data and settings - %1
ru.CleanupData=Данные и настройки - %1
ja.CleanupData=データと設定 - %1
en.CleanupDataNote=AppData\Roaming\sorta and AppData\Local\sorta: the index, config.yaml, the logs and the preview cache. The undo journal of the moves goes with them; the files that were moved stay where they are.
ru.CleanupDataNote=AppData\Roaming\sorta и AppData\Local\sorta: индекс, config.yaml, логи и кэш превью. Вместе с ними уйдёт журнал перемещений для отката; сами перемещённые файлы останутся на своих местах.
ja.CleanupDataNote=AppData\Roaming\sorta と AppData\Local\sorta: インデックス、config.yaml、ログ、プレビューキャッシュ。取り消し用の移動履歴も一緒に消えますが、移動したファイル自体はそのまま残ります。
en.CleanupModels=Downloaded models - %1
ru.CleanupModels=Скачанные модели - %1
ja.CleanupModels=ダウンロード済みモデル - %1
en.CleanupModelsNote=The weights live in caches shared with other programs, so only the models Sorta itself names are removed. Anything else in those caches, and whatever a link points at, is left alone.
ru.CleanupModelsNote=Веса лежат в кэшах, общих с другими программами, поэтому удаляются только те модели, которые называет сама Sorta. Всё остальное в этих кэшах и то, на что указывают ссылки, не трогается.
ja.CleanupModelsNote=重みは他のプログラムと共有するキャッシュにあるため、Sorta 自身が名前を挙げるモデルだけを削除します。そのキャッシュ内の他のものや、リンクの指す先には手を触れません。

[Code]
// F224 - uninstalling can leave nothing behind, but it asks first.
//
// Two ticks, both off, and the sizes are stated as numbers before anything happens:
// "frees 1.9 GB" is an answer, "clear the cache" is a riddle. The model half CALLS
// `sorta cache --clear-models --yes` instead of repeating its logic here - that rule is
// the dangerous one (shared caches, junctions that must not be followed) and it belongs
// where ordinary tests can reach it. The precedent is F211: the wizard calls
// `sorta doctor` rather than growing a check screen of its own.
//
// The data half is this script's own work: those two directories are the ones its own
// [Dirs] section created, they are ours by name, and no shared cache is inside them.

// FILE_ATTRIBUTE_REPARSE_POINT is NOT declared here: Inno 6 predefines the Windows file
// attribute constants, and re-declaring one aborts the compile with "Duplicate
// identifier" rather than shadowing it.

function GetFileAttributesW(lpFileName: String): DWORD;
  external 'GetFileAttributesW@kernel32.dll stdcall';

var
  RemoveData, RemoveModels: Boolean;

function SizeText(Bytes: Int64): String;
begin
  if Bytes >= 1000000000 then
    Result := Format('%.1f GB', [Bytes / 1000000000.0])
  else if Bytes >= 1000000 then
    Result := Format('%.0f MB', [Bytes / 1000000.0])
  else
    Result := Format('%.0f KB', [Bytes / 1000.0]);
end;

// What the two ticks would free, asked of the program itself rather than measured here:
// the model weights sit in caches this script has no business knowing the layout of.
function ReadReport(var ModelBytes: Int64; var DataBytes: Int64): Boolean;
var
  Python, OutFile, Line: String;
  Lines: TArrayOfString;
  Code, I: Integer;
begin
  Result := False;
  ModelBytes := 0;
  DataBytes := 0;
  Python := ExpandConstant('{app}\python\python.exe');
  if not FileExists(Python) then
    Exit;
  OutFile := ExpandConstant('{tmp}\sorta-uninstall-report.txt');
  if not Exec(ExpandConstant('{cmd}'), '/C ""' + Python +
      '" -X utf8 -c "from sorta import weights; print(weights.report())" > "' +
      OutFile + '""', '', SW_HIDE, ewWaitUntilTerminated, Code) then
    Exit;
  if Code <> 0 then
    Exit;
  if not LoadStringsFromFile(OutFile, Lines) then
    Exit;
  for I := 0 to GetArrayLength(Lines) - 1 do
  begin
    Line := Trim(Lines[I]);
    if Pos('models ', Line) = 1 then
      ModelBytes := StrToInt64Def(Copy(Line, 8, MaxInt), 0)
    else if Pos('data ', Line) = 1 then
      DataBytes := StrToInt64Def(Copy(Line, 6, MaxInt), 0);
  end;
  Result := True;
end;

function MakeNote(Form: TSetupForm; const Text: String; Top: Integer): Integer;
var
  Note: TNewStaticText;
begin
  Note := TNewStaticText.Create(Form);
  Note.Parent := Form;
  Note.Left := ScaleX(38);
  Note.Top := Top;
  Note.Width := Form.ClientWidth - ScaleX(56);
  Note.AutoSize := False;
  Note.WordWrap := True;
  Note.Height := ScaleY(32);
  Note.Caption := Text;
  Result := Note.Top + Note.Height;
end;

// The page. Both boxes start unticked, and Cancel is the same answer as leaving them
// that way: uninstalling a program is not a request to delete somebody's data.
function AskWhatToRemove(ModelBytes: Int64; DataBytes: Int64): Boolean;
var
  Form: TSetupForm;
  Intro: TNewStaticText;
  DataBox, ModelsBox: TNewCheckBox;
  OkButton, CancelButton: TNewButton;
  Y: Integer;
begin
  // Size and sizing flags go in the CALL: in Inno 6 `CreateCustomForm` takes four
  // parameters, and the shape here is the one CodeClasses.iss of the Inno distribution
  // itself uses. Neither axis may be resized -- every control on this page is laid out
  // at a fixed offset, so a dragged edge would only move the buttons out of view.
  Form := CreateCustomForm(ScaleX(470), ScaleY(285), False, False);
  try
    Form.Caption := CustomMessage('CleanupCaption');

    Intro := TNewStaticText.Create(Form);
    Intro.Parent := Form;
    Intro.Left := ScaleX(18);
    Intro.Top := ScaleY(16);
    Intro.Width := Form.ClientWidth - ScaleX(36);
    Intro.AutoSize := False;
    Intro.WordWrap := True;
    Intro.Height := ScaleY(48);
    Intro.Caption := CustomMessage('CleanupIntro');

    DataBox := TNewCheckBox.Create(Form);
    DataBox.Parent := Form;
    DataBox.Left := ScaleX(18);
    DataBox.Top := Intro.Top + Intro.Height + ScaleY(12);
    DataBox.Width := Form.ClientWidth - ScaleX(36);
    DataBox.Height := ScaleY(18);
    DataBox.Checked := False;
    DataBox.Caption := FmtMessage(CustomMessage('CleanupData'), [SizeText(DataBytes)]);
    Y := MakeNote(Form, CustomMessage('CleanupDataNote'),
                  DataBox.Top + DataBox.Height + ScaleY(2));

    ModelsBox := TNewCheckBox.Create(Form);
    ModelsBox.Parent := Form;
    ModelsBox.Left := ScaleX(18);
    ModelsBox.Top := Y + ScaleY(10);
    ModelsBox.Width := Form.ClientWidth - ScaleX(36);
    ModelsBox.Height := ScaleY(18);
    ModelsBox.Checked := False;
    // The argument array stays on THIS line on purpose. Inno decides what a section
    // header is by the first non-space character of a line, so a continuation that
    // begins with `[` is read as a section tag and the compile dies with "Invalid
    // section tag" pointing at Pascal that is perfectly valid.
    ModelsBox.Caption := FmtMessage(CustomMessage('CleanupModels'), [SizeText(ModelBytes)]);
    MakeNote(Form, CustomMessage('CleanupModelsNote'),
             ModelsBox.Top + ModelsBox.Height + ScaleY(2));

    OkButton := TNewButton.Create(Form);
    OkButton.Parent := Form;
    OkButton.Width := ScaleX(95);
    OkButton.Height := ScaleY(26);
    OkButton.Left := Form.ClientWidth - ScaleX(212);
    OkButton.Top := Form.ClientHeight - ScaleY(40);
    OkButton.Caption := SetupMessage(msgButtonOK);
    OkButton.ModalResult := mrOk;

    CancelButton := TNewButton.Create(Form);
    CancelButton.Parent := Form;
    CancelButton.Width := ScaleX(95);
    CancelButton.Height := ScaleY(26);
    CancelButton.Left := Form.ClientWidth - ScaleX(110);
    CancelButton.Top := OkButton.Top;
    CancelButton.Caption := SetupMessage(msgButtonCancel);
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;

    // No centering call. Inno centres a custom form relative to WizardForm through
    // `FlipAndCenterIfNeeded`, and an UNINSTALL has no WizardForm to centre on; without
    // the call the form centres itself on the screen, which is what this page wants.
    // (`Form.Center` is not a method at all -- it aborts the compile.)
    Result := Form.ShowModal() = mrOk;
    if Result then
    begin
      RemoveData := DataBox.Checked;
      RemoveModels := ModelsBox.Checked;
    end;
  finally
    Form.Free();
  end;
end;

function InitializeUninstall: Boolean;
var
  ModelBytes, DataBytes: Int64;
begin
  Result := True;
  RemoveData := False;
  RemoveModels := False;
  // /VERYSILENT has nobody to ask, and silence is not consent. The installer workflow
  // uninstalls exactly that way and must not quietly clean out the runner's disk.
  if UninstallSilent then
    Exit;
  // No numbers, no question: the whole point of the page is to say what disappears and
  // how much that is, and an unmeasured tick would be the "delete the cache" riddle
  // again. Nothing is removed in that case - the default answer everywhere here.
  if not ReadReport(ModelBytes, DataBytes) then
    Exit;
  AskWhatToRemove(ModelBytes, DataBytes);
end;

procedure RemoveOurDir(const Path: String);
var
  Attributes: DWORD;
begin
  if not DirExists(Path) then
    Exit;
  // Never through a link: a junction here points at a directory that is not ours, and
  // DelTree walks into one. The same rule the model half follows in `sorta.weights`.
  // A failed call answers $FFFFFFFF, which has that bit set too - so a directory this
  // cannot read is left alone, which is the right way round for a delete.
  Attributes := GetFileAttributesW(Path);
  if (Attributes and FILE_ATTRIBUTE_REPARSE_POINT) <> 0 then
    Exit;
  DelTree(Path, True, True, True);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Code: Integer;
begin
  // usUninstall and not usPostUninstall: the command below is run by {app}\python,
  // which does not exist any more once the files have been removed.
  if CurUninstallStep <> usUninstall then
    Exit;
  if RemoveModels then
    Exec(ExpandConstant('{app}\python\python.exe'),
         '-X utf8 -m sorta.cli cache --clear-models --yes',
         ExpandConstant('{userappdata}\sorta'), SW_HIDE, ewWaitUntilTerminated, Code);
  if RemoveData then
  begin
    RemoveOurDir(ExpandConstant('{localappdata}\sorta'));
    RemoveOurDir(ExpandConstant('{userappdata}\sorta'));
  end;
end;
