"""F182: the chrome string catalog of the web app — and nothing else.

Every feature that changes a caption edits this table: F158, F171 and F175 all did,
in one day, while it sat in the middle of `ui.py` between the payload builders and
the request handler. Alone in a file it still collides — but only caption against
caption, and those merges are additive and small.

`_t` resolves a key against it; that lives in `page.py`, next to the template the
strings are substituted into.
"""
from __future__ import annotations


# F133: the tabs are named after what a person DOES, not after the code that computed
# the numbers. Three things can be done to a collection and they differ by what they do
# to the file system: one canon (a physical move), any number of slices (hardlinks, free
# to make and to drop) and the junk (a subtraction, and the dangerous one). "Overview"
# holds the state of the collection and the run that produces it — one question asked at
# two moments in time.
_UI_STRINGS: dict[str, dict[str, str]] = {
    # F126: the tab is the workspace, not one of its slices — duplicates are the first
    # of four things a person opens it to go through.
    "tab_review": {"ru": "Разбор", "en": "Review", "ja": "仕分け"},
    "tab_layout": {"ru": "Раскладка", "en": "Layout", "ja": "振り分け"},
    "tab_slices": {"ru": "Срезы", "en": "Slices", "ja": "スライス"},
    "tab_person": {"ru": "Люди", "en": "People", "ja": "人物"},
    "tab_event": {"ru": "События", "en": "Events", "ja": "イベント"},
    "tab_animal": {"ru": "Животные", "en": "Animals", "ja": "動物"},
    "tab_moves": {"ru": "Перемещения", "en": "Moves", "ja": "移動"},
    # F175: the slice used to be called "Not personal photos", and that name was wrong
    # twice over. A photograph of a receipt, a screenshot of a conversation with your
    # wife and a passport are all personal — they are simply not photographs taken FOR
    # MEMORY, which is a different thing; and read as "not personal" the slice invites
    # deleting it, while a thousand of the frames in it are documents that must not be
    # deleted. The old name also sat one letter away from `files.not_personal`, the flag
    # for downloaded films (three files of 38 485), which is about where a file came
    # from and not about what is in the frame — see the note in i18n._FOLDERS.
    "tab_junk": {"ru": "Служебные кадры", "en": "Utility frames",
                 "ja": "実用目的のコマ"},
    "process_intro": {
        "ru": "Укажите папку с фото и нажмите «Обработать» — индекс наполнится "
              "(гео, лица, события, мусор, почти-дубликаты). Файлы не перемещаются.",
        "en": "Enter a photo folder and click Process — the index fills in "
              "(geo, faces, events, junk, near-duplicates). Files are not moved.",
        "ja": "写真フォルダを指定して「処理する」を押すと、インデックスが作成されます"
              "（位置情報、顔、イベント、不要写真、類似写真）。ファイルは移動されません。",
    },
    "process_path_placeholder": {
        "ru": "Путь к папке с фото", "en": "Path to photo folder",
        "ja": "写真フォルダのパス",
    },
    "process_start_button": {"ru": "Обработать", "en": "Process", "ja": "処理する"},
    "process_browse_button": {"ru": "Обзор…", "en": "Browse…", "ja": "参照…"},
    "process_deep_label": {
        "ru": "Глубокий анализ (VLM)", "en": "Deep analysis (VLM)",
        "ja": "詳細分析（VLM）",
    },
    # F161: the hint used to open with "Slower", and that stopped being true here. The
    # checkbox is permission and nothing else since the deep tier became a line of its
    # own below it: what is slow are the lines it unlocks, each of which now says its own
    # price. Leaving "slower" on the master would price the same hours twice and leave
    # the one thing this switch really decides — whether a model may be raised at all —
    # unsaid.
    "process_deep_hint": {
        "ru": "Разрешает поднимать модель. Сам по себе не считает ничего: время "
              "показано у строк под ним. Нужен `uv sync --extra vlm` (иначе "
              "автоматический откат на быстрый анализ).",
        "en": "Permission to load the model. It computes nothing by itself — the time "
              "is on the lines below it. Requires `uv sync --extra vlm` (otherwise "
              "falls back to the fast tier automatically).",
        "ja": "モデルの読み込みを許可します。これ自体は何も計算しません（所要時間は"
              "下の各項目に表示されます）。`uv sync --extra vlm` が必要です"
              "（なければ自動的に高速分析にフォールバックします）。",
    },
    # F161: the effect that used to be the master switch's own, given its name back. It
    # is deliberately named after what it PRODUCES and not after how: "deep analysis" is
    # a technology, and 85% of what that technology did on the live run of 2026-07-28 was
    # find products (2 202 verdicts of 2 592).
    "process_products_label": {
        "ru": "Распознавание товаров", "en": "Product recognition", "ja": "商品の認識",
    },
    # And the hint says what a person GETS, in the two places they will look for it. The
    # last sentence is the one that matters: the fast tier does not produce the class at
    # all, so without this line the products slice is not thin — it is empty.
    "process_products_hint": {
        "ru": "Отсюда берутся папка «_Товары» в раскладке и одноимённый срез: снимки "
              "вещей на продажу отделяются от снимков на память. Без этой строки "
              "товаров не мало — их ноль.",
        "en": "The “_Products” folder of the layout and the slice of the same name come "
              "from here: pictures of things for sale are told apart from pictures kept "
              "for memory. Without this line products are not few — there are none.",
        "ja": "振り分けの「_商品」フォルダーと同名のスライスはここから作られます: "
              "売るために撮った写真を、思い出の写真と切り分けます。この項目がなければ"
              "商品は少ないのではなく、ゼロです。",
    },
    "process_deep_vlm_missing": {
        "ru": "VLM не установлен — будет использован быстрый ярус (CLIP). "
              "Доустановите: `uv sync --extra vlm`.",
        "en": "VLM is not installed — the fast tier (CLIP) will be used instead. "
              "Install it: `uv sync --extra vlm`.",
        "ja": "VLM がインストールされていません。代わりに高速ティア（CLIP）が"
              "使用されます。インストール: `uv sync --extra vlm`。",
    },
    "process_geo_online_label": {
        "ru": "Онлайн-гео (точнее заграница)", "en": "Online geo (more accurate abroad)",
        "ja": "オンライン位置情報（海外でより正確）",
    },
    "process_geo_online_hint": {
        "ru": "Точнее определяет места за границей, но отправляет GPS-координаты "
              "фото на сервер геокодирования (сами фото никуда не отправляются).",
        "en": "More accurate place names abroad, but sends photo GPS coordinates "
              "to a geocoding server (the photos themselves are never sent).",
        "ja": "海外の地名をより正確に特定しますが、写真のGPS座標をジオコーディング"
              "サーバーに送信します（写真自体は送信されません）。",
    },
    "process_faces_label": {
        "ru": "Разбор по лицам", "en": "Detect faces",
        "ja": "顔の検出",
    },
    "process_faces_hint": {
        "ru": "Самый долгий шаг (детекция + кластеризация); включай, если "
              "нужна раскладка/альбомы по людям.",
        "en": "The slowest step (detection + clustering); enable it if you "
              "need sorting or albums by person.",
        "ja": "最も時間のかかるステップです（検出とクラスタリング）。人物ごとの"
              "整理やアルバムが必要な場合に有効にしてください。",
    },
    "process_events_label": {
        "ru": "Разбор по событиям", "en": "Detect events",
        "ja": "イベントの検出",
    },
    # F123: the hint has one job — to say that this checkbox is NOT another long step.
    # It stands next to faces (17 minutes) and deep analysis (hours), and read as one of
    # them it simply never gets ticked; the animals ride on the CLIP pass the junk stage
    # makes anyway.
    "process_pets_label": {
        "ru": "Искать животных", "en": "Detect animals",
        "ja": "動物の検出",
    },
    "process_pets_hint": {
        "ru": "Почти бесплатно: едет на уже идущем проходе CLIP (не отдельный "
              "долгий шаг), добавляет вкладку «Животные» и альбом.",
        "en": "Almost free: it rides on the CLIP pass that already runs (not another "
              "long step), and adds the “Animals” tab and album.",
        "ja": "ほぼ無料です。すでに実行中の CLIP パスに相乗りするため（別の長い"
              "ステップではありません）、「動物」タブとアルバムが追加されます。",
    },
    "process_events_hint": {
        "ru": "Группировка в поездки/события по времени и месту (нужен geo); "
              "для раскладки/альбомов по событиям.",
        "en": "Groups photos into trips/events by time and place (needs "
              "geo); for sorting or albums by event.",
        "ja": "時間と場所に基づいて旅行やイベントにグループ化します"
              "（位置情報が必要）。イベントごとの整理やアルバムに使います。",
    },
    # --- F138: the run budget — the block that turns six switches into an estimate ---
    # The list has to MEAN something, or moving four expensive knobs here is just the
    # console of toggles F133 removed. The meaning is the price on every line and the
    # sum under them, right where the eye is already going to the button.
    "costs_title": {
        "ru": "Что посчитать", "en": "What to compute", "ja": "何を計算するか",
    },
    # The estimate says it is an estimate, in the block itself. A wrong exact number is
    # worse than an honest approximate one: promise twenty minutes, take two hours, and
    # no figure on this screen is believed again.
    "costs_estimate_note": {
        "ru": "Время — оценка по этой коллекции: замеренная скорость на число кадров "
              "в индексе. Не обещание; прочерк значит «по этой базе не сосчитать».",
        "en": "The times are an estimate for this collection: a measured rate times the "
              "frames this index holds. Not a promise; a dash means this index cannot "
              "tell.",
        "ja": "所要時間はこのコレクションに対する目安です（実測の速度 × インデックス内"
              "のコマ数）。約束ではありません。ダッシュは「このインデックスでは算出でき"
              "ない」という意味です。",
    },
    # F159: where the numbers came from, said next to them. A person deciding whether to
    # wait four hours needs to tell "this is how it went for YOU last time" from "this is
    # how it went for the developer" — the second is an honest guess, and calling it one
    # is what keeps the first believable.
    "costs_source_measured": {
        "ru": "Числа — по вашему прошлому прогону ({date}).",
        "en": "The numbers come from your own last run ({date}).",
        "ja": "数値は前回のご自身の実行（{date}）に基づいています。",
    },
    "costs_source_default": {
        "ru": "Оценка по умолчанию: своих замеров на этой машине ещё нет.",
        "en": "A default estimate: this machine has no measurements of its own yet.",
        "ja": "既定の見積もりです。この端末での実測値はまだありません。",
    },
    "costs_source_mixed": {
        "ru": "Часть чисел — по вашему прошлому прогону ({date}), остальные — оценка "
              "по умолчанию.",
        "en": "Some numbers come from your own last run ({date}), the rest are default "
              "estimates.",
        "ja": "一部の数値は前回のご自身の実行（{date}）に基づき、残りは既定の見積もりです。",
    },
    "costs_base_label": {
        "ru": "Города, места и дубли", "en": "Cities, places and duplicates",
        "ja": "都市・場所・重複",
    },
    "costs_always": {"ru": "всегда", "en": "always", "ja": "常に"},
    "costs_total_label": {
        "ru": "Примерно за прогон:", "en": "This run, roughly:", "ja": "実行の目安:",
    },
    # A sum with an unknown line in it is not a sum. It is still worth showing as a
    # floor — "at least this much" is a decision a person can make.
    "costs_total_at_least": {
        "ru": "не меньше {time}", "en": "at least {time}", "ja": "{time} 以上",
    },
    "costs_unknown": {"ru": "—", "en": "—", "ja": "—"},
    "costs_free": {
        "ru": "почти бесплатно", "en": "almost free", "ja": "ほぼ無料",
    },
    # F145: what a line under a cleared master switch costs. Not "almost free" — that is
    # said about a stage that RUNS and is cheap; this one does not run at all, and the
    # number says so plainly because the sum below has to add up with it.
    "costs_off": {
        "ru": "0 — не выполняется", "en": "0 — does not run",
        "ja": "0 — 実行されません",
    },
    # F161: and what the MASTER switch costs, which is also nothing — for the opposite
    # reason. A line under a cleared master does not run; this one has nothing to run.
    # Both numbers are zero and saying so with one string would hide the difference the
    # feature is about: permission is not work.
    "costs_permission_only": {
        "ru": "0 — только разрешение", "en": "0 — permission only",
        "ja": "0 — 許可のみ",
    },
    "costs_under_minute": {
        "ru": "меньше минуты", "en": "under a minute", "ja": "1 分未満",
    },
    "costs_minutes": {"ru": "~{minutes} мин", "en": "~{minutes} min", "ja": "約 {minutes} 分"},
    "costs_hours": {
        "ru": "~{hours} ч {minutes} мин", "en": "~{hours} h {minutes} min",
        "ja": "約 {hours} 時間 {minutes} 分",
    },
    # F130 moved out of config.yaml onto this screen: it costs ~13 minutes, and a knob
    # that costs a quarter of an hour belongs where the run is started.
    "process_pets_verify_label": {
        "ru": "Проверять животных моделью", "en": "Verify the animals with the model",
        "ja": "動物をモデルで確認",
    },
    "process_pets_verify_hint": {
        "ru": "Каждого кандидата показать модели: живое животное, изображение или его "
              "нет. Точнее, но это отдельные вопросы к модели по каждому кадру.",
        "en": "Every candidate is shown to the model: a live animal, a picture of one, "
              "or none. More accurate, but it is one model question per frame.",
        "ja": "候補を 1 枚ずつモデルに見せます（実際の動物か、その画像か、いないか）。"
              "精度は上がりますが、コマごとにモデルへ問い合わせます。",
    },
    # F145: said next to every option that asks the SAME model the "Deep analysis"
    # checkbox loads. With the checkbox clear each of them costs nothing and does
    # nothing — the line says which switch turns it back on, so a dead option does not
    # read as a missing feature.
    "process_needs_deep_hint": {
        "ru": "Работает только с «Глубоким анализом (VLM)» — без него модель не "
              "поднимается и этот пункт ничего не делает.",
        "en": "Works only with Deep analysis (VLM) — without it no model is loaded and "
              "this option does nothing.",
        "ja": "「詳細解析（VLM）」がオンのときのみ動作します。オフの場合モデルは読み込ま"
              "れず、この項目は何もしません。",
    },
    # --- F81/F82: the three blocks of the first tab + the exclusion tree ------
    # F82: the two mechanisms are now side by side in one tree, so the wording carries
    # the whole difference between them — "do not SCAN" (the files never enter the index
    # at all) and "do not LAY OUT" (`sort.exclude_dirs`: indexed, searched, deduplicated,
    # simply left where they are). Each gets a one-line explanation, because this is
    # exactly the distinction a live user got wrong. The F77 per-file corrections
    # ("leave alone") are a third thing and live on the "Cities" tab.
    "step_source_title": {"ru": "Источник", "en": "Source", "ja": "ソース"},
    "step_options_title": {
        "ru": "Параметры запуска", "en": "Run options", "ja": "実行オプション",
    },
    "step_actions_title": {"ru": "Действия", "en": "Actions", "ja": "アクション"},
    "step_change_button": {"ru": "изменить", "en": "change", "ja": "変更"},
    # The same button folds the step back: opening one and finding nothing to change
    # is the common case, and it used to leave the block open with no way back.
    "step_collapse_button": {"ru": "свернуть", "en": "collapse", "ja": "折りたたむ"},
    "step_needs_source_hint": {
        "ru": "Сначала укажите папку с фото.",
        "en": "Choose a photo folder first.",
        "ja": "先に写真フォルダを指定してください。",
    },
    "step_options_summary_prefix": {
        "ru": "Параметры: ", "en": "Options: ", "ja": "オプション: ",
    },
    "step_options_summary_default": {
        "ru": "по умолчанию", "en": "defaults", "ja": "既定",
    },
    "excludes_button": {
        "ru": "Исключить папки…", "en": "Leave folders out…", "ja": "フォルダを除外…",
    },
    "excludes_title": {
        "ru": "Какие папки исключить", "en": "Folders to leave out",
        "ja": "除外するフォルダ",
    },
    "excludes_hint": {
        "ru": "Нажимайте на значок слева от папки, чтобы переключить её состояние. "
              "Состояние родителя действует на всё поддерево.",
        "en": "Click the mark to the left of a folder to switch its state. A folder's "
              "state applies to its whole subtree.",
        "ja": "フォルダ左のマークをクリックして状態を切り替えます。親の状態は"
              "サブツリー全体に適用されます。",
    },
    # The three states, each with the one line that says what it actually does.
    "tri_none_label": {
        "ru": "обрабатывать", "en": "process", "ja": "処理する",
    },
    "tri_none_hint": {
        "ru": "как обычно: сканируется и раскладывается",
        "en": "as usual: scanned and laid out",
        "ja": "通常どおり: スキャンして振り分けます",
    },
    "tri_layout_label": {
        "ru": "не раскладывать", "en": "don't sort", "ja": "振り分けない",
    },
    "tri_layout_hint": {
        "ru": "уже разобрано руками: файлы остаются в индексе и на месте, "
              "дубликаты по ним ищутся, но раскладка их не трогает",
        "en": "already sorted by hand: the files stay in the index and where they "
              "are, duplicates still find them, the layout leaves them alone",
        "ja": "手作業で整理済み: ファイルはインデックスに残り、その場に置かれます。"
              "重複検索の対象にはなりますが、振り分けは行いません",
    },
    "tri_scan_label": {
        "ru": "не сканировать", "en": "don't scan", "ja": "スキャンしない",
    },
    "tri_scan_hint": {
        "ru": "не нужно совсем: папка не читается, её файлов не будет в индексе, "
              "они не попадут ни в поиск дубликатов, ни в статистику",
        "en": "not needed at all: the folder is not read, its files never enter the "
              "index and take part in neither duplicate search nor statistics",
        "ja": "まったく不要: フォルダは読み込まれず、ファイルはインデックスに"
              "入らないため、重複検索にも統計にも含まれません",
    },
    "excludes_save_button": {"ru": "Сохранить", "en": "Save", "ja": "保存"},
    "excludes_saved": {
        "ru": "Сохранено. «Не сканировать» исчезнет из индекса при следующей "
              "обработке, «не раскладывать» подействует на следующей раскладке.",
        "en": "Saved. «Don't scan» leaves the index on the next run, «don't sort» "
              "applies on the next layout.",
        "ja": "保存しました。「スキャンしない」は次回の処理でインデックスから消え、"
              "「振り分けない」は次回の振り分けから適用されます。",
    },
    "excludes_error_prefix": {
        "ru": "Не удалось получить дерево папок: ",
        "en": "Could not load the folder tree: ",
        "ja": "フォルダツリーを取得できません: ",
    },
    "excludes_save_error_prefix": {
        "ru": "Не удалось сохранить: ", "en": "Could not save: ", "ja": "保存できません: ",
    },
    "excludes_empty": {
        "ru": "Вложенных папок нет.", "en": "No subfolders here.",
        "ja": "サブフォルダはありません。",
    },
    "excludes_truncated": {
        "ru": "Дерево очень большое — показаны первые {limit} папок.",
        "en": "The tree is very large — the first {limit} folders are shown.",
        "ja": "ツリーが大きいため、最初の {limit} 件のフォルダのみ表示しています。",
    },
    "excludes_summary_none": {
        "ru": "обрабатывается целиком", "en": "processed in full", "ja": "全体を処理",
    },
    # Two numbers, never merged into one (§3): they mean different things, and one
    # total would hide which mechanism a folder ended up in.
    "excludes_summary": {
        "ru": "не сканируется папок: {count} ({size})",
        "en": "not scanned: {count} folder(s), {size}",
        "ja": "スキャンしないフォルダ: {count} 件 ({size})",
    },
    "excludes_summary_layout": {
        "ru": "не раскладывается папок: {count}",
        "en": "not sorted: {count} folder(s)",
        "ja": "振り分けないフォルダ: {count} 件",
    },
    "excludes_folder_meta": {
        "ru": "{count} файлов · {size}", "en": "{count} files · {size}",
        "ja": "{count} 件 · {size}",
    },
    "size_units": {
        "ru": "Б КБ МБ ГБ ТБ", "en": "B KB MB GB TB", "ja": "B KB MB GB TB",
    },
    # F135: one button, so the run has to say what it skipped. "Nothing happened" and
    # "everything was already done" look identical without these two lines.
    "process_summary_title": {
        "ru": "Что сделал прогон:",
        "en": "What the run did:",
        "ja": "この実行の内容:",
    },
    "process_summary_stage": {
        "ru": "{stage} — обработано: {processed}, пропущено как уже обработанные: {skipped}",
        "en": "{stage} — processed: {processed}, skipped as already processed: {skipped}",
        "ja": "{stage} — 処理: {processed} 件、処理済みのためスキップ: {skipped} 件",
    },
    "env_cpu_warning": {
        "ru": "Установлен CPU-профиль: обработка идёт на процессоре — распознавание "
              "людей, VLM и большие коллекции заметно медленнее. Для скорости "
              "поставьте GPU-профиль: uv tool install --force \".[gpu]\".",
        "en": "CPU profile installed: processing runs on the CPU — face recognition, "
              "VLM and large collections are noticeably slower. For speed, install "
              "the GPU profile: uv tool install --force \".[gpu]\".",
        "ja": "CPU プロファイルがインストールされています: 処理は CPU で実行され、"
              "顔認識・VLM・大規模なコレクションは著しく遅くなります。高速化するには "
              "GPU プロファイルをインストールしてください: uv tool install --force \".[gpu]\"。",
    },
    "process_cancel_button": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "process_enter_path": {
        "ru": "Введите путь к папке.", "en": "Enter a folder path.",
        "ja": "フォルダのパスを入力してください。",
    },
    "process_stage_progress": {
        "ru": "Этап {stage} ({index}/{total}): {done} из {all}",
        "en": "Stage {stage} ({index}/{total}): {done} of {all}",
        "ja": "ステージ {stage}（{index}/{total}）: {done}/{all}",
    },
    "process_stage_progress_indeterminate": {  # #37: total not yet known (e.g. indexing)
        "ru": "Этап {stage} ({index}/{total}): обработано {done}",
        "en": "Stage {stage} ({index}/{total}): {done} processed",
        "ja": "ステージ {stage}（{index}/{total}）: {done} 件処理済み",
    },
    "process_done": {
        "ru": "Обработка завершена.", "en": "Processing complete.",
        "ja": "処理が完了しました。",
    },
    "process_cancelled": {
        "ru": "Обработка остановлена.", "en": "Processing stopped.",
        "ja": "処理が中止されました。",
    },
    "process_cancel_requested": {
        "ru": "Отмена запрошена — остановка после текущего шага…",
        "en": "Cancel requested — stopping after the current step…",
        "ja": "キャンセルを要求しました — 現在のステップ後に停止します…",
    },
    "process_error_prefix": {
        "ru": "Ошибка обработки: ", "en": "Processing error: ", "ja": "処理エラー: ",
    },
    "process_start_error_prefix": {
        "ru": "Не удалось запустить: ", "en": "Failed to start: ", "ja": "開始できません: ",
    },
    # F84: the sub-phases of clustering inside the `faces` stage. The keys mirror
    # faces.CLUSTER_PHASE_* ("process_phase_" + the key from /api/process/status).
    "process_phase_cluster_read": {
        "ru": "кластеры: чтение эмбеддингов", "en": "clusters: reading embeddings",
        "ja": "クラスタ: 埋め込みを読み込み中",
    },
    "process_phase_cluster_hdbscan": {
        "ru": "кластеры: группировка лиц", "en": "clusters: grouping faces",
        "ja": "クラスタ: 顔をグループ化中",
    },
    "process_phase_cluster_inherit": {
        "ru": "кластеры: перенос имён", "en": "clusters: carrying names over",
        "ja": "クラスタ: 名前を引き継ぎ中",
    },
    "process_phase_cluster_write": {
        "ru": "кластеры: запись", "en": "clusters: saving",
        "ja": "クラスタ: 保存中",
    },
    # F100: the sub-phases of the `junk` stage. Keys: junk.CLASSIFY_PHASE_*. All four
    # are measurable, the deep one included (the VLM gate knows its candidates before
    # the loop starts), so the caption is shown next to the real N / M — the
    # process_phase_elapsed form below is for phases that have no percent at all.
    # The stage line right above already says "классификация", so these captions name
    # only the phase — the same reason the clustering ones say "кластеры" and not "лица".
    "process_phase_junk_clip": {
        "ru": "быстрый разбор (CLIP)", "en": "fast pass (CLIP)",
        "ja": "高速判定 (CLIP)",
    },
    "process_phase_junk_ocr": {
        "ru": "поиск текста (OCR)", "en": "text detection (OCR)",
        "ja": "テキスト検出 (OCR)",
    },
    "process_phase_junk_vlm": {
        "ru": "глубокий анализ (VLM)", "en": "deep analysis (VLM)",
        "ja": "詳細解析 (VLM)",
    },
    "process_phase_junk_write": {
        "ru": "запись вердиктов", "en": "saving verdicts",
        "ja": "判定を保存中",
    },
    # F141: the second CLIP pass — the search index. Named apart from the fast pass above
    # because it is what `features.search_index` costs and nothing else, and a caption
    # saying "fast pass" over ten minutes of a second encode would be the wrong sentence.
    "process_phase_junk_search": {
        "ru": "поисковый индекс (CLIP)", "en": "search index (CLIP)",
        "ja": "検索インデックス (CLIP)",
    },
    # F154: the object detector over the candidates of the animal query. A caption of its
    # own for the reason the search index has one: it is a second model over a short list,
    # neither the fast CLIP pass nor the VLM tier, and its minutes are what
    # `features.detector` costs. (The only line this feature adds to this file — a phase
    # without a string surfaces as a raw identifier, which tests/test_ui_junk_phase.py
    # requires it not to.)
    "process_phase_junk_detect": {
        "ru": "детектор объектов", "en": "object detector",
        "ja": "物体検出",
    },
    "process_phase_elapsed": {  # a phase with no percent — the clock is the sign of life
        "ru": "{phase} — идёт {seconds} с",
        "en": "{phase} — {seconds}s so far",
        "ja": "{phase} — 経過 {seconds} 秒",
    },
    "process_stage_index": {"ru": "индексация", "en": "indexing", "ja": "インデックス作成"},
    "process_stage_geo": {"ru": "гео", "en": "geo", "ja": "位置情報"},
    "process_stage_landmarks": {"ru": "места", "en": "landmarks", "ja": "ランドマーク"},
    # F165: the two halves of the classification, and the chips have to tell them apart —
    # the first one decides WHAT a frame is (and lets the faces stage skip what is not a
    # photograph), the second one measures the photographs it left.
    "process_stage_classify": {"ru": "вердикты", "en": "verdicts", "ja": "判定"},
    "process_stage_faces": {"ru": "лица", "en": "faces", "ja": "顔"},
    "process_stage_events": {"ru": "события", "en": "events", "ja": "イベント"},
    "process_stage_junk": {"ru": "классификация", "en": "classification", "ja": "分類"},
    "process_stage_phash": {"ru": "почти-дубликаты", "en": "near-duplicates", "ja": "類似写真"},
    "process_reset_button": {
        "ru": "Начать заново", "en": "Start over", "ja": "最初からやり直す",
    },
    "process_reset_confirm": {
        "ru": "Сотрёт индекс, включая имена людей/событий и решения по дублям. "
              "Фото и уже разложенные папки НЕ тронет. Продолжить?",
        "en": "This will erase the index, including people/event names and "
              "duplicate decisions. Photos and already-sorted folders are NOT "
              "touched. Continue?",
        "ja": "人物名・イベント名・重複の判定を含むインデックスを消去します。"
              "写真や既に整理済みのフォルダには触れません。続行しますか?",
    },
    # F93: the geo cache survives "Start over" — the name of a point on the map does
    # not depend on which files the user keeps, and re-asking the provider costs ~10
    # minutes of network. But an invisible unresettable thing must not exist, so the
    # way out lives exactly where the user already decided to erase something. Default
    # UNCHECKED: the cache is normally what makes the next run fast.
    "process_reset_clear_geo_label": {
        "ru": "Также очистить кэш геоданных",
        "en": "Also clear the geo cache",
        "ja": "位置情報のキャッシュも消去する",
    },
    "process_reset_clear_geo_hint": {
        "ru": "Ответы онлайн-геокодера переживают сброс, поэтому повторный прогон не "
              "стоит сети. Ставьте галочку, если провайдер ответил неверно и город "
              "нужно переспросить (при provider: online это снова минуты сети).",
        "en": "The online geocoder's answers survive a reset, so the next run costs no "
              "network. Tick this if the provider got a city wrong and has to be asked "
              "again (with provider: online that is minutes of network once more).",
        "ja": "オンライン地理コーダーの応答はリセット後も残るため、次回の実行に通信は不要です。"
              "プロバイダーが誤った都市を返した場合のみチェックしてください"
              "(provider: online では再び数分の通信が必要になります)。",
    },
    "process_reset_confirm_ok": {
        "ru": "Стереть индекс", "en": "Erase the index", "ja": "インデックスを消去",
    },
    "process_reset_confirm_cancel": {
        "ru": "Отмена", "en": "Cancel", "ja": "キャンセル",
    },
    "process_reset_done": {
        "ru": "Индекс сброшен.", "en": "Index reset.", "ja": "インデックスをリセットしました。",
    },
    "process_reset_done_geo": {
        "ru": "Индекс сброшен, кэш геоданных очищен.",
        "en": "Index reset, geo cache cleared.",
        "ja": "インデックスをリセットし、位置情報のキャッシュを消去しました。",
    },
    "process_reset_error_prefix": {
        "ru": "Не удалось сбросить: ", "en": "Failed to reset: ", "ja": "リセットできません: ",
    },
    # F94: the caches were reachable only from `sorta cache`, while the web app is
    # advertised as a full entry point — so on a live collection 12 GB of previews had
    # no way out for anyone who does not use the terminal. Sizes are shown and both
    # clears are offered. F117 added a ceiling, and it does not change that stance: it
    # is 0 by default, so nothing is ever deleted unless a person sets a number — the
    # ceiling answers "my disk filled up", it is not a policy applied on their behalf.
    "cache_title": {"ru": "Кэши", "en": "Caches", "ja": "キャッシュ"},
    # F117: shown next to the size, because a size without its bound says nothing.
    "cache_limit": {
        "ru": "Потолок: {limit} ГБ — занято {percent}%",
        "en": "Ceiling: {limit} GB — {percent}% used",
        "ja": "上限: {limit} GB — {percent}% 使用",
    },
    "cache_no_limit": {
        "ru": "Потолок не задан — кэш растёт, пока есть место на диске",
        "en": "No ceiling — the cache grows for as long as the disk allows",
        "ja": "上限なし — ディスクの空きがある限り増えます",
    },
    "cache_sizes": {
        "ru": "Кэш превью: {preview} ({files} файлов) · Кэш геоданных: {geo} записей",
        "en": "Preview cache: {preview} ({files} files) · Geo cache: {geo} entries",
        "ja": "プレビューキャッシュ: {preview} ({files} 件) · 位置情報キャッシュ: {geo} 件",
    },
    "cache_hint": {
        "ru": "Кэш превью — уменьшенные копии кадров, он ускоряет прогон и "
              "пересобирается сам. Кэш геоданных — ответы онлайн-геокодера, они "
              "избавляют повторный прогон от сети. Сами по себе они не уменьшаются; "
              "если задать потолок, кэш превью удаляет самые давно не читанные копии, "
              "пока не уложится в него.",
        "en": "The preview cache holds downscaled copies of the frames: it speeds the "
              "run up and rebuilds itself. The geo cache holds the online geocoder's "
              "answers, which spare a repeat run the network. Neither shrinks on its "
              "own; with a ceiling set, the preview cache drops its least recently read "
              "copies until it fits.",
        "ja": "プレビューキャッシュは縮小したコマの控えで、処理を速くし、自動的に作り直されます。"
              "位置情報キャッシュはオンライン地理コーダーの応答で、再実行時の通信を省きます。"
              "自動では減りませんが、上限を設定すると、収まるまで最も長く読まれていない"
              "プレビューから削除されます。",
    },
    "cache_clear_preview_button": {
        "ru": "Очистить кэш превью", "en": "Clear the preview cache",
        "ja": "プレビューキャッシュを消去",
    },
    "cache_clear_geo_button": {
        "ru": "Очистить кэш геоданных", "en": "Clear the geo cache",
        "ja": "位置情報キャッシュを消去",
    },
    "cache_clear_preview_confirm": {
        "ru": "Удалить кэш превью ({preview})? Место освободится сразу, а кэш "
              "соберётся заново сам — но первый прогон после этого будет медленнее: "
              "336 мс на кадр против 73 мс на готовом кэше. Фото и индекс не тронет.",
        "en": "Delete the preview cache ({preview})? The space is freed at once and the "
              "cache rebuilds itself — but the first run after that is slower: 336 ms "
              "per frame against 73 ms on a warm cache. Photos and the index are NOT "
              "touched.",
        "ja": "プレビューキャッシュ ({preview}) を削除しますか? 容量はすぐに解放され、"
              "キャッシュは自動的に作り直されますが、次の処理は遅くなります"
              "(1 コマあたり 336 ミリ秒、キャッシュありなら 73 ミリ秒)。"
              "写真とインデックスには触れません。",
    },
    "cache_clear_geo_confirm": {
        "ru": "Удалить ответы онлайн-геокодера ({geo} записей)? У уже обработанных "
              "фото города останутся, но при provider: online следующий прогон "
              "снова сходит в сеть — это минуты. Делайте это, если провайдер "
              "ответил неверно.",
        "en": "Delete the online geocoder's answers ({geo} entries)? The photos already "
              "processed keep their cities, but with provider: online the next run goes "
              "to the network again — that is minutes. Do this if the provider got an "
              "answer wrong.",
        "ja": "オンライン地理コーダーの応答 ({geo} 件) を削除しますか? "
              "処理済みの写真の都市は残りますが、provider: online では次回の実行で"
              "再び通信が発生します(数分)。応答が誤っていた場合に実行してください。",
    },
    "cache_clear_preview_done": {
        "ru": "Кэш превью очищен: удалено файлов {n}.",
        "en": "Preview cache cleared: {n} files removed.",
        "ja": "プレビューキャッシュを消去しました: {n} 件を削除。",
    },
    "cache_clear_geo_done": {
        "ru": "Кэш геоданных очищен: удалено записей {n}.",
        "en": "Geo cache cleared: {n} entries removed.",
        "ja": "位置情報キャッシュを消去しました: {n} 件を削除。",
    },
    "cache_clear_error_prefix": {
        "ru": "Не удалось очистить кэш: ", "en": "Failed to clear the cache: ",
        "ja": "キャッシュを消去できません: ",
    },
    "lightbox_close": {"ru": "Закрыть", "en": "Close", "ja": "閉じる"},
    "lightbox_open": {"ru": "Открыть превью", "en": "Open preview", "ja": "プレビューを開く"},
    # F80: the filmstrip of a clip — the tile marker and the frame pager.
    "video_badge": {"ru": "Видео", "en": "Video", "ja": "動画"},
    "video_open": {
        "ru": "Открыть кадры видео", "en": "Open video frames", "ja": "動画のフレームを開く",
    },
    "frame_prev": {"ru": "Предыдущий кадр", "en": "Previous frame", "ja": "前のフレーム"},
    "frame_next": {"ru": "Следующий кадр", "en": "Next frame", "ja": "次のフレーム"},
    "frame_of": {
        "ru": "Кадр {n} из {all}", "en": "Frame {n} of {all}", "ja": "フレーム {all} 中 {n}",
    },
    "delete_remember_label": {
        "ru": "Не спрашивать подтверждение удаления в этой сессии",
        "en": "Don't ask for delete confirmation this session",
        "ja": "このセッション中は削除の確認をしない",
    },
    "expand_all": {"ru": "Развернуть всё", "en": "Expand all", "ja": "すべて展開"},
    "collapse_all": {"ru": "Свернуть всё", "en": "Collapse all", "ja": "すべて折りたたむ"},
    "back_to_top": {"ru": "Наверх", "en": "Top", "ja": "上へ"},
    "loading": {"ru": "Загрузка...", "en": "Loading...", "ja": "読み込み中..."},
    "save_all_choices": {
        "ru": "Сохранить весь выбор", "en": "Save all choices", "ja": "すべての選択を保存",
    },
    "merge_selected": {"ru": "Слить выбранные", "en": "Merge selected", "ja": "選択を統合"},
    "theme_light": {"ru": "Светлая", "en": "Light", "ja": "ライト"},
    "theme_dark": {"ru": "Тёмная", "en": "Dark", "ja": "ダーク"},
    "error_loading_plan": {
        "ru": "Ошибка загрузки плана: ", "en": "Error loading plan: ",
        "ja": "プラン読み込みエラー: ",
    },
    # F70: the plan tab loads a folder page by page — the counter and the button
    # that asks for the next page.
    "plan_shown_of": {
        "ru": "показано {n} из {all}", "en": "showing {n} of {all}",
        "ja": "{all} 件中 {n} 件を表示",
    },
    "plan_load_more": {
        "ru": "Загрузить ещё", "en": "Load more", "ja": "さらに読み込む",
    },
    "plan_empty": {
        "ru": "План пуст — нечего раскладывать.",
        "en": "The plan is empty — nothing to lay out.",
        "ja": "プランは空です — 整理する対象がありません。",
    },
    # --- F173: the three captions of the shared pager (`makePager`) --------------------
    # One button, one counter, one warning, for every ordered slice there is and every one
    # there will be. They are not per-slice keys because the fifth copy of "Показать ещё"
    # is how a new slice ships without the button at all: a slice that has to add a string
    # to get one has a reason to skip it, and search shipped without one for that reason.
    "slice_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    # THE fix of the counter. "Показано 200" is indistinguishable from "нашлось ровно 200",
    # and for a ranking the second is almost never true — so the denominator is the length
    # of the list, always, and the numerator only says how far down it the reader is.
    "slice_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    # The price of depth, in one line and only where something is actually ranked. Measured
    # on 2026-08-02/03: doubling the list adds ~25 points of completeness on average, and
    # the query «дети» goes from 61% to 89% — while the frames that arrive with the second
    # page are exactly the ones the model was least sure about. Pressing the button buys
    # coverage with precision, and a person choosing that has to know it is a trade.
    "slice_depth_hint": {
        "ru": "Дальше по списку — больше находок и больше промахов: вторая половина "
              "заметно полнее, но модель в ней уверена меньше.",
        "en": "Further down the list means more found and more missed: the second half is "
              "noticeably more complete, and the model is less sure of it.",
        "ja": "リストを下るほど、見つかる数は増え、外れも増えます。後半は網羅性が高い"
              "一方で、モデルの確信度は低くなります。",
    },
    "error_loading_moves": {
        "ru": "Ошибка загрузки перемещений: ", "en": "Error loading moves: ",
        "ja": "移動読み込みエラー: ",
    },
    "error_loading_dupes": {
        "ru": "Ошибка загрузки дублей: ", "en": "Error loading duplicates: ",
        "ja": "重複読み込みエラー: ",
    },
    "error_loading_clusters": {
        "ru": "Ошибка загрузки кластеров: ", "en": "Error loading clusters: ",
        "ja": "クラスター読み込みエラー: ",
    },
    "confirm_delete_photo": {
        "ru": "Удалить этот файл в корзину?", "en": "Move this file to trash?",
        "ja": "このファイルをごみ箱に移動しますか?",
    },
    "delete": {"ru": "Удалить", "en": "Delete", "ja": "削除"},
    "delete_selected": {
        "ru": "Удалить выбранное", "en": "Delete selected", "ja": "選択を削除",
    },
    "select_for_delete": {
        "ru": "Выбрать для удаления", "en": "Select for deletion", "ja": "削除対象に選択",
    },
    "confirm_delete_selected": {
        "ru": "Удалить {n} файлов в корзину?", "en": "Move {n} files to trash?",
        "ja": "{n} 件のファイルをごみ箱に移動しますか?",
    },
    "status_planned": {"ru": "запланировано", "en": "planned", "ja": "予定"},
    "status_done": {"ru": "выполнено", "en": "done", "ja": "完了"},
    "status_undone": {"ru": "отменено", "en": "undone", "ja": "取消"},
    "status_failed": {"ru": "ошибка", "en": "failed", "ja": "失敗"},
    "status_deleted": {"ru": "удалено", "en": "deleted", "ja": "削除済み"},
    "batch_label": {"ru": "Батч", "en": "Batch", "ja": "バッチ"},
    "started_label": {"ru": "начат", "en": "started", "ja": "開始"},
    "finished_label": {"ru": "завершён", "en": "finished", "ja": "終了"},
    "in_progress_label": {"ru": "в процессе", "en": "in progress", "ja": "進行中"},
    "files_count_label": {"ru": "файлов", "en": "files", "ja": "ファイル数"},
    "no_moves_yet": {
        "ru": "Перемещений ещё не выполнялось.", "en": "No moves have been made yet.",
        "ja": "まだ移動は実行されていません。",
    },
    "unnamed": {"ru": "без имени", "en": "unnamed", "ja": "名前なし"},
    "faces_unit": {"ru": "лиц", "en": "faces", "ja": "顔"},
    "person_name_placeholder": {"ru": "Имя человека", "en": "Person's name", "ja": "人物名"},
    "name_button": {"ru": "Назвать", "en": "Name", "ja": "名前を設定"},
    "alert_enter_name": {
        "ru": "Введите имя.", "en": "Enter a name.", "ja": "名前を入力してください。",
    },
    "select_for_merge": {
        "ru": "выбрать для слияния", "en": "select for merge", "ja": "統合対象として選択",
    },
    "no_clusters": {
        "ru": "Кластеры лиц не найдены.", "en": "No face clusters found.",
        "ja": "顔クラスターが見つかりません。",
    },
    "recommended_badge": {
        "ru": "★ рекомендовано", "en": "★ recommended", "ja": "★ おすすめ",
    },
    # F148: what the group's STORED recommendation says under the frame it names. The
    # source is part of the caption and not a detail: how much an advice is worth
    # depends on who gives it, and these two are given by different judges.
    "keeper_badge_model": {
        "ru": "рекомендуем оставить · по модели",
        "en": "recommended to keep · by the model",
        "ja": "残すのがおすすめ · モデルの判断",
    },
    "keeper_badge_sharpness": {
        "ru": "рекомендуем оставить · по резкости",
        "en": "recommended to keep · by sharpness",
        "ja": "残すのがおすすめ · 鮮明さで判定",
    },
    # What the recommendation does NOT say, in the one place it can be read: there is
    # always exactly one per group, and a burst of six can hold two moments both worth
    # keeping. Keeping several frames is allowed and normal — advising several is what
    # the program cannot do.
    "keeper_badge_hint": {
        "ru": "Рекомендация одна на группу. В серии может быть несколько удачных "
              "кадров — оставить можно любой из них и не один.",
        "en": "One recommendation per group. A burst can hold more than one frame worth "
              "keeping — you may keep any of them, and more than one.",
        "ja": "推奨はグループにつき1件です。連写には残す価値のあるコマが複数ある"
              "こともあり、どれでも、また複数でも残せます。",
    },
    "action_keep": {"ru": "оставить", "en": "keep", "ja": "保持"},
    "action_to_delete": {"ru": "к удалению", "en": "to delete", "ja": "削除予定"},
    "skip_group_label": {
        "ru": "не удалять эту группу", "en": "don't delete this group",
        "ja": "このグループを削除しない",
    },
    "delete_dupes_button": {
        "ru": "Удалить дубли", "en": "Delete duplicates", "ja": "重複を削除",
    },
    "confirm_trash_group": {
        "ru": "Удалить в корзину все кадры группы {n}, кроме выбранного?",
        "en": "Move all frames in group {n} to trash, except the selected one?",
        "ja": "選択したもの以外、グループ{n}のすべてのフレームをごみ箱に移動しますか?",
    },
    "alert_choose_keeper": {
        "ru": "Выберите кадр, который нужно оставить.", "en": "Select the frame to keep.",
        "ja": "残すフレームを選択してください。",
    },
    "no_dupes": {
        "ru": "Почти-дубликаты не найдены.", "en": "No near-duplicates found.",
        "ja": "ほぼ重複が見つかりません。",
    },
    "select_group_to_save": {
        "ru": "Отметьте хотя бы одну группу для сохранения.",
        "en": "Mark at least one group to save.",
        "ja": "保存するグループを少なくとも1つ選択してください。",
    },
    "saved_groups": {
        "ru": "Сохранено групп: {n}", "en": "Groups saved: {n}", "ja": "保存したグループ数: {n}",
    },
    "group_title": {
        "ru": "Группа {n} ({count} кадра)", "en": "Group {n} ({count} frames)",
        "ja": "グループ{n}（{count}枚）",
    },
    "album_button": {
        "ru": "Собрать в папку", "en": "Gather into folder", "ja": "フォルダにまとめる",
    },
    "album_mode_link": {"ru": "Ссылка (link)", "en": "Link", "ja": "リンク"},
    "album_mode_copy": {"ru": "Копия", "en": "Copy", "ja": "コピー"},
    "album_mode_move": {"ru": "Перемещение", "en": "Move", "ja": "移動"},
    "album_where_placeholder": {
        "ru": "Фильтр, напр. city=Барселона", "en": "Filter, e.g. city=Barcelona",
        "ja": "フィルター（例: city=Barcelona）",
    },
    "album_name_placeholder": {
        "ru": "Имя папки альбома", "en": "Album folder name", "ja": "アルバムフォルダ名",
    },
    "album_dest_placeholder": {
        "ru": "Путь назначения альбома", "en": "Album destination path",
        "ja": "アルバムの保存先パス",
    },
    "album_name_first_hint": {
        "ru": "Сначала назовите кластер", "en": "Name the cluster first",
        "ja": "先にクラスターに名前を付けてください",
    },
    "album_preview_text": {
        "ru": "{n} файлов → {dest}", "en": "{n} files → {dest}", "ja": "{n} ファイル → {dest}",
    },
    "album_blocked_text": {
        "ru": "; move заблокирует {k} мульти-кадров",
        "en": "; move will block {k} multi-person frames",
        "ja": "；moveは{k}件のマルチ人物フレームをブロックします",
    },
    "album_confirm_move": {
        "ru": "Внимание: перемещение изымет файлы из общего пула сортировки. Продолжить?",
        "en": "Warning: moving will remove files from the common sorting pool. Continue?",
        "ja": "警告: 移動するとファイルは共通の振り分けプールから除外されます。続行しますか?",
    },
    "album_confirm_generic": {
        "ru": "Собрать альбом?", "en": "Gather the album?", "ja": "アルバムをまとめますか?",
    },
    "album_result_text": {
        "ru": "Собрано {n}, ошибок {f}", "en": "Gathered {n}, errors {f}",
        "ja": "収集済み{n}、エラー{f}",
    },
    "album_in_progress": {
        "ru": "Идёт сбор альбома...", "en": "Gathering album...", "ja": "アルバムを収集中...",
    },
    "no_events": {
        "ru": "События не найдены.", "en": "No events found.", "ja": "イベントが見つかりません。",
    },
    "error_loading_events": {
        "ru": "Ошибка загрузки событий: ", "en": "Error loading events: ",
        "ja": "イベント読み込みエラー: ",
    },
    # --- F43: apply the city layout (the "Cities" tab) -----------------
    "sort_dest_placeholder": {
        "ru": "Папка назначения (пусто = в исходной папке)",
        "en": "Destination folder (empty = in the source folder)",
        "ja": "移動先フォルダ（空欄 = 元のフォルダ内）",
    },
    "sort_dest_hint": {
        "ru": "Пусто — коллекция раскладывается внутри исходной папки (in-place).",
        "en": "Empty — the collection is sorted inside the source folder (in-place).",
        "ja": "空欄の場合、コレクションは元のフォルダ内で振り分けられます（in-place）。",
    },
    "sort_dest_inplace_label": {
        "ru": "исходная папка (in-place)", "en": "source folder (in-place)",
        "ja": "元のフォルダ（in-place）",
    },
    "sort_mode_move": {"ru": "Переместить", "en": "Move", "ja": "移動"},
    "sort_mode_copy": {"ru": "Копировать", "en": "Copy", "ja": "コピー"},
    "sort_apply_button": {"ru": "Разложить", "en": "Apply", "ja": "振り分ける"},
    "folder_lang_label": {
        "ru": "Язык папок", "en": "Folder language", "ja": "フォルダの言語",
    },
    "folder_lang_saved": {
        "ru": "Язык папок сохранён — план пересчитан.",
        "en": "Folder language saved — the plan was recomputed.",
        "ja": "フォルダの言語を保存しました — プランを再計算しました。",
    },
    # F104: `sort_confirm_summary` (F43) lived here — the single line of a window.confirm
    # that said only "N files, M folders". It is gone with that dialog: the summary is
    # built from `sort_summary_*` below, off /api/sort/summary, and names the volume, the
    # review folders and what is already in the destination as well.
    # F97: the text used to send the user to the terminal (`sorta undo`) — there is a
    # button on the "Moves" tab now, so it points at the button.
    "sort_confirm_move": {
        "ru": "ВНИМАНИЕ: оригиналы будут ПЕРЕМЕЩЕНЫ. "
              "Откатить можно кнопкой на вкладке «Перемещения».",
        "en": "WARNING: originals will be MOVED. "
              "You can roll this back with the button on the Moves tab.",
        "ja": "警告: オリジナルファイルが移動されます。"
              "「移動」タブのボタンで元に戻せます。",
    },
    "sort_confirm_inplace": {
        "ru": "ВНИМАНИЕ: реструктурируется ИСХОДНОЕ дерево коллекции, а не копия "
              "в отдельной папке.",
        "en": "WARNING: this restructures the SOURCE tree of the collection, "
              "not a copy in a separate folder.",
        "ja": "警告: これは別フォルダのコピーではなく、コレクションの元のツリー"
              "構造そのものを再編成します。",
    },
    "sort_confirm_copy": {
        "ru": "Оригиналы останутся на месте — будут созданы копии.",
        "en": "Originals stay in place — copies will be created.",
        "ja": "オリジナルはそのまま残り、コピーが作成されます。",
    },
    "sort_progress_line": {
        "ru": "Готово {done} из {all}", "en": "Done {done} of {all}",
        "ja": "完了 {done}/{all}",
    },
    "sort_done_text": {
        "ru": "Разложено {n}, ошибок {f} (+ пропущено {p} на месте)",
        "en": "Sorted {n}, errors {f} (+ {p} skipped in place)",
        "ja": "振り分け済み {n}、エラー {f}（+ その場でスキップ {p}）",
    },
    "sort_error_prefix": {
        "ru": "Ошибка раскладки: ", "en": "Sort error: ", "ja": "振り分けエラー: ",
    },
    "sort_preview_stale_warning": {
        "ru": "Превью плана не обновилось — обновите вкладку.",
        "en": "Plan preview did not refresh — reload the tab.",
        "ja": "プレビューが更新されませんでした — タブを再読み込みしてください。",
    },
    "sort_start_error_prefix": {
        "ru": "Не удалось запустить: ", "en": "Failed to start: ", "ja": "開始できません: ",
    },
    # --- F97: cancelling a layout + rolling back from the "Moves" tab ---------
    "sort_cancel_button": {"ru": "Отменить", "en": "Cancel", "ja": "中止"},
    "sort_cancel_requested": {
        "ru": "Отмена запрошена — текущий файл будет дописан…",
        "en": "Cancellation requested — the current file will be finished…",
        "ja": "中止をリクエストしました — 現在のファイルは書き終えます…",
    },
    "sort_cancelled_text": {
        "ru": "Отменено: разложено {n} из {all}, ошибок {f}.",
        "en": "Cancelled: sorted {n} of {all}, errors {f}.",
        "ja": "中止しました: {all} 件中 {n} 件を振り分け、エラー {f} 件。",
    },
    "sort_already_copied_note": {
        "ru": " Уже было на месте: {c}.", "en": " Already there: {c}.",
        "ja": " すでに配置済み: {c} 件。",
    },
    "sort_undo_hint": {
        "ru": "Разложенное можно откатить — вкладка «Перемещения».",
        "en": "What was sorted can be rolled back on the Moves tab.",
        "ja": "振り分けた結果は「移動」タブで元に戻せます。",
    },
    "undo_button": {"ru": "Откатить", "en": "Roll back", "ja": "元に戻す"},
    "undo_cancel_button": {"ru": "Отменить откат", "en": "Cancel rollback", "ja": "中止"},
    "undo_confirm_copy": {
        "ru": "Будет удалено {n} копий в {dest}. Оригиналы не тронутся.",
        "en": "{n} copies in {dest} will be deleted. The originals stay untouched.",
        "ja": "{dest} 内のコピー {n} 件を削除します。オリジナルはそのまま残ります。",
    },
    "undo_confirm_move": {
        "ru": "{n} файлов вернутся в исходные папки.",
        "en": "{n} files will go back to their original folders.",
        "ja": "{n} 件のファイルが元のフォルダに戻ります。",
    },
    "undo_confirm_ok": {"ru": "Откатить", "en": "Roll back", "ja": "元に戻す"},
    "undo_confirm_cancel": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "undo_progress_line": {
        "ru": "Откачено {done} из {all}", "en": "Rolled back {done} of {all}",
        "ja": "元に戻した件数 {done}/{all}",
    },
    "undo_done_text": {
        "ru": "Откачено {n}, отсутствовало {m}, ошибок {f}",
        "en": "Rolled back {n}, missing {m}, errors {f}",
        "ja": "元に戻した {n} 件、見つからない {m} 件、エラー {f} 件",
    },
    "undo_cancelled_text": {
        "ru": "Отменено: откачено {n}. Нажмите «Откатить» ещё раз, чтобы доделать.",
        "en": "Cancelled: {n} rolled back. Press Roll back again to finish.",
        "ja": "中止しました: {n} 件を元に戻しました。続けるにはもう一度「元に戻す」を押してください。",
    },
    "undo_stray_title": {
        "ru": "Битые копии прерванного переноса — не удалены, проверьте вручную:",
        "en": "Broken copies from an interrupted transfer — not deleted, check by hand:",
        "ja": "中断された転送による壊れたコピー — 削除していません。手動で確認してください:",
    },
    "undo_nothing_to_undo": {
        "ru": "Откатывать нечего.", "en": "Nothing to roll back.",
        "ja": "元に戻すものがありません。",
    },
    "undo_error_prefix": {
        "ru": "Ошибка отката: ", "en": "Rollback error: ", "ja": "元に戻す処理のエラー: ",
    },
    "undo_start_error_prefix": {
        "ru": "Не удалось запустить откат: ", "en": "Failed to start the rollback: ",
        "ja": "元に戻す処理を開始できません: ",
    },
    "undo_cancel_requested": {
        "ru": "Отмена отката запрошена…", "en": "Rollback cancellation requested…",
        "ja": "元に戻す処理の中止をリクエストしました…",
    },
    # --- F104: the settings column + the summary before a layout ------------
    "settings_title": {"ru": "Настройки", "en": "Settings", "ja": "設定"},
    "settings_hint": {
        "ru": "Меняются прямо здесь и сохраняются в config.yaml. Перезапускать "
              "«sorta ui» не нужно — новые значения берёт следующий прогон.",
        "en": "Changed right here and saved into config.yaml. No need to restart "
              "`sorta ui` — the next run picks the new values up.",
        "ja": "ここで変更すると config.yaml に保存されます。`sorta ui` の再起動は不要 — "
              "次の処理から新しい値が使われます。",
    },
    # F138: the column says out loud that the expensive knobs are not missing but
    # elsewhere — a person who used to switch the deep tier on from here has to be told
    # where it went, not left looking for it.
    "settings_costs_moved_hint": {
        "ru": "Здесь то, что не стоит времени прогона: пороги, модель, потоки. Что "
              "стоит часов — глубокий разбор, качество кадров, животные, лучший кадр "
              "в группе — живёт на экране запуска, рядом со своей ценой.",
        "en": "What is here costs a run nothing: thresholds, the model, the pools. What "
              "costs hours — the deep tier, frame quality, animals, the best frame of a "
              "group — lives on the run screen, next to its price.",
        "ja": "ここにあるのは実行時間を増やさない項目です（しきい値・モデル・スレッド）。"
              "時間のかかる項目 — 詳細解析、コマの品質、動物、グループ内のベストショット "
              "— は実行画面にあり、そこで所要時間が示されます。",
    },
    "settings_vlm_model_label": {"ru": "Модель", "en": "Model", "ja": "モデル"},
    "settings_vlm_workers_label": {
        "ru": "Потоки подготовки", "en": "Preparation threads", "ja": "前処理スレッド数",
    },
    "settings_vlm_workers_hint": {
        "ru": "Сколько кадров готовится к отправке в модель параллельно. Каждый поток "
              "держит кадр в памяти — больше не значит быстрее.",
        "en": "How many frames are prepared for the model in parallel. Every thread "
              "holds a frame in RAM — more is not automatically faster.",
        "ja": "モデルに渡すフレームを同時に何枚準備するか。各スレッドがフレームを"
              "メモリに保持するため、増やせば速くなるとは限りません。",
    },
    "settings_vlm_max_edge_label": {
        "ru": "Разрешение кадра, px", "en": "Frame resolution, px", "ja": "フレーム解像度 (px)",
    },
    "settings_vlm_max_edge_hint": {
        "ru": "Длинная сторона кадра, который видит модель. Меньше — быстрее и "
              "экономнее по видеопамяти, но мелкий текст на снимке различим хуже.",
        "en": "The long edge of the frame the model sees. Smaller is faster and easier "
              "on VRAM, but fine text in a shot becomes harder to make out.",
        "ja": "モデルが見るフレームの長辺。小さいほど高速で VRAM も節約できますが、"
              "写真内の細かい文字は読み取りにくくなります。",
    },
    # F119: the F113 quality cascade. Each signal is taken by the cheapest instrument
    # that answers it, and the hints say which — a person deciding whether to switch
    # something on needs to know what it will cost, not only what it does.
    "settings_quality_title": {
        "ru": "Качество кадра", "en": "Frame quality", "ja": "コマの品質",
    },
    "settings_quality_hint": {
        "ru": "Необязательные признаки: помогают выбрать лучший кадр из серии и найти "
              "случайные снимки. Всё выключено по умолчанию и считается только на "
              "прогоне. Сгруппировано по тому, ЧЕМ признак считается, — это и есть "
              "разница в цене.",
        "en": "Optional signals: they help pick the best frame of a burst and spot the "
              "shots nobody meant to take. All off by default and computed during a "
              "run. Grouped by WHAT answers each one, because that is where the cost "
              "difference is.",
        "ja": "任意のシグナル: 連写から最良のコマを選び、意図しない撮影を見つけるのに"
              "役立ちます。既定はすべて無効で、実行中にのみ計算されます。**何が**答える"
              "かで分けてあります — 費用の差はそこにあるからです。",
    },
    "settings_quality_cheap_title": {
        "ru": "Без VLM", "en": "No VLM needed", "ja": "VLM 不要",
    },
    "settings_quality_cheap_hint": {
        "ru": "Считается на проходе, который и так идёт: CLIP и обычная арифметика по "
              "превью. Включать можно, даже если глубокий анализ выключен.",
        "en": "Computed on a pass that runs anyway: CLIP and plain arithmetic over the "
              "preview. Safe to switch on even with deep analysis off.",
        "ja": "どのみち走るパスで計算されます: CLIP と、プレビューに対する単純な演算。"
              "詳細解析が無効でも有効にできます。",
    },
    "settings_quality_gate_title": {
        "ru": "Кого спрашивать у модели",
        "en": "Who reaches the model",
        "ja": "モデルに届くコマ",
    },
    "settings_quality_gate_hint": {
        "ru": "Пороги, которые решают, у каких кадров вообще спрашивать. Считаются "
              "дёшево, а экономят дорогое: чем уже полоса, тем меньше кадров уйдёт в "
              "модель.",
        "en": "The thresholds that decide which frames are worth asking about at all. "
              "Cheap to compute and what saves the expensive part: the narrower the "
              "band, the fewer frames reach the model.",
        "ja": "そもそもどのコマについて尋ねるかを決めるしきい値です。計算は安価で、"
              "高価な部分を節約します — 帯が狭いほど、モデルに届くコマは減ります。",
    },
    "settings_features_pet_threshold_label": {
        "ru": "Порог уверенности для животных",
        "en": "Animal confidence threshold",
        "ja": "動物の信頼度しきい値",
    },
    "settings_features_subject_score_min_label": {
        "ru": "Порог «это вообще фотография»",
        "en": "“This is a photograph at all” threshold",
        "ja": "「そもそも写真か」のしきい値",
    },
    "settings_features_subject_score_min_hint": {
        "ru": "Второй вход в модель. Это вероятность от CLIP: если он оценивает кадр "
              "как фотографию ниже этого порога — значит, сам не понял, на что смотрит, "
              "и такой кадр стоит показать модели.",
        "en": "The second way into the model. This is CLIP's own probability: scoring a "
              "frame as “a photograph” below this threshold is CLIP saying it does not "
              "know what it is looking at, and such a frame is worth showing to the "
              "model.",
        "ja": "モデルへの 2 つ目の入口です。これは CLIP 自身の確率で、「写真である」の"
              "スコアがこのしきい値を下回るのは、CLIP が何を見ているか分からないという"
              "ことであり、そのコマはモデルに見せる価値があります。",
    },
    "settings_features_sharpness_max_edge_hint": {
        "ru": "Сама резкость считается всегда и бесплатно — это дисперсия лапласиана по "
              "превью, которое другие стадии уже построили. Модель здесь ни при чём.",
        "en": "Sharpness itself is always computed and costs nothing — the variance of "
              "a Laplacian over the preview other stages have already built. No model "
              "is involved.",
        "ja": "鮮鋭度そのものは常に計算され、費用はかかりません — 他の段階がすでに作った"
              "プレビューに対するラプラシアンの分散です。モデルは関係ありません。",
    },
    "settings_features_sharpness_band_min_label": {
        "ru": "Резкость: нижняя граница",
        "en": "Sharpness: lower bound",
        "ja": "鮮鋭度: 下限",
    },
    "settings_features_sharpness_band_max_label": {
        "ru": "Резкость: верхняя граница",
        "en": "Sharpness: upper bound",
        "ja": "鮮鋭度: 上限",
    },
    "settings_features_sharpness_band_hint": {
        "ru": "Ниже нижней кадр однозначно смазан, выше верхней — однозначно резкий; "
              "спрашивать модель незачем ни там, ни там. К модели уходит только полоса "
              "между ними.",
        "en": "Below the lower bound a frame is plainly blurred, above the upper one it "
              "is plainly sharp, and neither is worth asking a model about. Only the "
              "band between them reaches the model.",
        "ja": "下限より下は明らかにぶれており、上限より上は明らかに鮮明で、どちらもモデル"
              "に尋ねる価値はありません。モデルに届くのは、その間の帯だけです。",
    },
    "settings_features_sharpness_max_edge_label": {
        "ru": "Размер кадра для оценки резкости, px",
        "en": "Frame size for the sharpness measure, px",
        "ja": "鮮鋭度を測るコマのサイズ (px)",
    },
    # F117: the ceiling belongs in the settings column rather than next to the cache
    # sizes, because it is a stored preference and the numbers next to the buttons are
    # a measurement. 0 is spelled out in the hint: an empty-looking limit is the one
    # value a person is most likely to misread.
    "settings_preview_max_gb_label": {
        "ru": "Потолок кэша превью, ГБ",
        "en": "Preview cache ceiling, GB",
        "ja": "プレビューキャッシュ上限 (GB)",
    },
    "settings_preview_max_gb_hint": {
        "ru": "0 — без потолка: кэш растёт, пока есть место (около 150 КБ на снимок, "
              "то есть ~45 ГБ на 300 тысячах). С потолком удаляются самые давно не "
              "читанные превью, пока кэш не уложится; выключать кэш ради места не "
              "стоит — холодный кадр стоит 336 мс против 73.",
        "en": "0 means no ceiling: the cache grows while there is room (about 150 KB a "
              "shot, so ~45 GB at 300 000). With a ceiling the least recently read "
              "previews are dropped until it fits. Do not switch the cache off to save "
              "space — a cold frame costs 336 ms against 73.",
        "ja": "0 は上限なし: 空きがある限り増えます (1 枚およそ 150 KB、30 万枚で ~45 GB)。"
              "上限を設けると、収まるまで最も長く読まれていないプレビューから削除されます。"
              "容量のためにキャッシュを切るのは得策ではありません — 未キャッシュのコマは "
              "73 ms に対し 336 ms かかります。",
    },
    "settings_folders_title": {"ru": "Папки", "en": "Folders", "ja": "フォルダ"},
    "settings_folder_lang_hint": {
        "ru": "Язык названий папок раскладки. План ниже пересчитывается сразу.",
        "en": "The language of the layout's folder names. The plan below is recomputed "
              "immediately.",
        "ja": "振り分けフォルダ名の言語。下のプランはすぐに再計算されます。",
    },
    "settings_saved": {"ru": "Сохранено.", "en": "Saved.", "ja": "保存しました。"},
    "settings_error_prefix": {
        "ru": "Не удалось сохранить настройку: ", "en": "Could not save the setting: ",
        "ja": "設定を保存できませんでした: ",
    },
    "settings_busy": {
        "ru": "Идёт прогон — настройки не меняются на ходу. Дождитесь окончания.",
        "en": "A run is in progress — settings do not change mid-run. Wait for it to end.",
        "ja": "処理の実行中です — 途中で設定は変更できません。終了までお待ちください。",
    },
    # F145: the same statement for everything else that writes — marks, the trash, an
    # album, a layout. The server has always answered 409; this is the sentence that
    # says so BEFORE the click instead of after it.
    "actions_busy": {
        "ru": "Идёт прогон — действия, меняющие данные, недоступны. "
              "Вернутся сами по окончании.",
        "en": "A run is in progress — actions that change data are unavailable. "
              "They come back on their own when it ends.",
        "ja": "処理の実行中です — データを変更する操作は利用できません。"
              "終了すると自動的に戻ります。",
    },
    "selection_delete_hint": {
        "ru": "Файлы уедут в корзину системы — не мимо неё.",
        "en": "The files go to the system trash, not past it.",
        "ja": "ファイルはシステムのゴミ箱に移動します（完全削除ではありません）。",
    },
    "sort_confirm_title": {
        "ru": "Разложить коллекцию?", "en": "Lay the collection out?",
        "ja": "コレクションを振り分けますか?",
    },
    "sort_confirm_ok": {"ru": "Разложить", "en": "Apply", "ja": "振り分ける"},
    "sort_confirm_cancel": {"ru": "Отмена", "en": "Cancel", "ja": "キャンセル"},
    "sort_summary_dest": {
        "ru": "Куда: {dest}", "en": "Where to: {dest}", "ja": "移動先: {dest}",
    },
    "sort_summary_mode_move": {
        "ru": "Перемещение — оригиналы будут перенесены",
        "en": "Move — the originals will be transferred",
        "ja": "移動 — オリジナルが移されます",
    },
    "sort_summary_mode_copy": {
        "ru": "Копирование — оригиналы останутся на месте",
        "en": "Copy — the originals stay where they are",
        "ja": "コピー — オリジナルはその場に残ります",
    },
    "sort_summary_files": {
        "ru": "{n} файлов в {dirs} папок, {size}",
        "en": "{n} files into {dirs} folders, {size}",
        "ja": "{n} 件のファイルを {dirs} 個のフォルダへ、{size}",
    },
    "sort_summary_existing": {
        "ru": "В назначении уже лежит {n} из них; {same} совпадут и будут пропущены",
        "en": "{n} of them are already in the destination; {same} match and will be skipped",
        "ja": "そのうち {n} 件はすでに移動先にあります。{same} 件は一致するためスキップされます",
    },
    "sort_summary_existing_none": {
        "ru": "В назначении ничего из этого ещё нет",
        "en": "None of this is in the destination yet",
        "ja": "これらはまだ移動先にありません",
    },
    "sort_summary_existing_unknown": {
        "ru": "Что уже лежит в назначении — неизвестно: папка не задана, а источник не один",
        "en": "What is already in the destination is unknown: no folder given and more "
              "than one source",
        "ja": "移動先に何があるかは不明です: フォルダ未指定でソースが複数あります",
    },
    "sort_summary_service": {
        "ru": "В служебные папки: товары — {products}, документы — {documents}",
        "en": "Into the review folders: products — {products}, documents — {documents}",
        "ja": "確認用フォルダへ: 商品 — {products} 件、書類 — {documents} 件",
    },
    "sort_summary_empty": {
        "ru": "Раскладывать нечего: план пуст. Обработайте коллекцию на вкладке "
              "«Обработка» — или снимите пометки «не трогать».",
        "en": "There is nothing to lay out: the plan is empty. Process the collection on "
              "the Process tab — or unmark the frames left alone.",
        "ja": "振り分ける対象がありません: プランが空です。「処理」タブでコレクションを"
              "処理するか、「そのままにする」の指定を解除してください。",
    },
    "sort_summary_error": {
        "ru": "Не удалось посчитать сводку: ", "en": "Could not compute the summary: ",
        "ja": "サマリーを計算できませんでした: ",
    },
    # --- F77: manual corrections to the layout (the "Cities" tab) ----------
    "override_exclude_button": {
        "ru": "Не трогать", "en": "Leave alone", "ja": "そのままにする",
    },
    "override_clear_button": {
        "ru": "Снять правку", "en": "Clear correction", "ja": "修正を解除",
    },
    "override_move_button": {
        "ru": "Перенести в…", "en": "Move to…", "ja": "移動先…",
    },
    "override_target_placeholder": {
        "ru": "папка раскладки…", "en": "layout folder…", "ja": "振り分け先フォルダ…",
    },
    "override_exclude_folder_button": {
        "ru": "Не трогать папку", "en": "Leave folder alone", "ja": "フォルダをそのままに",
    },
    "override_exclude_folder_confirm": {
        "ru": "Исключить из раскладки все файлы этой папки ({n})? Они останутся там, "
              "где лежат.",
        "en": "Exclude all {n} files of this folder from the layout? They stay exactly "
              "where they are.",
        "ja": "このフォルダの {n} 件すべてを振り分けから除外しますか? "
              "ファイルは現在の場所に残ります。",
    },
    "override_excluded_mark": {
        "ru": "не трогать", "en": "left alone", "ja": "移動しない",
    },
    "override_reassigned_mark": {
        "ru": "перенос → {target}", "en": "moved → {target}", "ja": "移動先 → {target}",
    },
    "override_hint": {
        "ru": "Ручные правки сильнее автоматики: помеченные «не трогать» остаются на "
              "месте, перенесённые уходят в выбранную папку при раскладке.",
        "en": "Manual corrections outrank the automatic rules: files marked «leave "
              "alone» stay put, moved ones go to the chosen folder when you apply.",
        "ja": "手動の修正は自動判定より優先されます。「そのままにする」を付けた"
              "ファイルは移動せず、移動先を指定したものは振り分け時にそのフォルダへ入ります。",
    },
    "override_alert_choose_target": {
        "ru": "Выберите папку для переноса.", "en": "Choose a destination folder.",
        "ja": "移動先のフォルダを選択してください。",
    },
    "override_error_prefix": {
        "ru": "Не удалось сохранить правку: ", "en": "Could not save the correction: ",
        "ja": "修正を保存できません: ",
    },
    # F85c: assigning a place to a whole group by hand
    "place_search_placeholder": {
        "ru": "Город или страна", "en": "City or country", "ja": "都市または国",
    },
    "place_assign_button": {
        "ru": "Назначить место", "en": "Assign place", "ja": "場所を指定",
    },
    "place_clear_button": {
        "ru": "Отменить назначение", "en": "Undo assignment", "ja": "指定を取り消す",
    },
    "place_folder_button": {
        "ru": "Место для исходной папки", "en": "Place for the source folder",
        "ja": "元フォルダの場所",
    },
    "place_not_found": {
        "ru": "Такого места нет в базе — проверьте написание.",
        "en": "No such place in the bundled data — check the spelling.",
        "ja": "その場所は同梱データにありません。綴りを確認してください。",
    },
    "place_alert_choose": {
        "ru": "Сначала выберите место из списка.",
        "en": "Pick a place from the list first.",
        "ja": "先に一覧から場所を選んでください。",
    },
    "place_assign_confirm": {
        "ru": "Назначить место «{place}» файлам этой группы ({n})?",
        "en": "Assign the place «{place}» to the files of this group ({n})?",
        "ja": "このグループのファイル（{n}）に場所「{place}」を指定しますか？",
    },
    "place_folder_confirm": {
        "ru": "Назначить место «{place}» всем файлам исходной папки «{dir}»?",
        "en": "Assign the place «{place}» to every file of the source folder «{dir}»?",
        "ja": "元フォルダ「{dir}」のすべてのファイルに場所「{place}」を指定しますか？",
    },
    "place_event_clear_confirm": {
        "ru": "Снять назначенное место с файлов этого события ({n})?",
        "en": "Remove the assigned place from the files of this event ({n})?",
        "ja": "このイベントのファイル（{n}）から指定した場所を解除しますか？",
    },
    "place_folder_clear_confirm": {
        "ru": "Снять назначенное место с файлов исходной папки «{dir}»?",
        "en": "Remove the assigned place from the files of the source folder «{dir}»?",
        "ja": "元フォルダ「{dir}」のファイルから指定した場所を解除しますか？",
    },
    "place_assigned_status": {
        "ru": "Назначено: {n}", "en": "Assigned: {n}", "ja": "指定しました: {n}",
    },
    "place_cleared_status": {
        "ru": "Назначение снято: {n}", "en": "Assignment removed: {n}",
        "ja": "指定を解除しました: {n}",
    },
    "place_skipped_gps": {
        "ru": " · с точным GPS пропущено: {n}",
        "en": " · skipped, they have exact GPS: {n}",
        "ja": " · GPS があるためスキップ: {n}",
    },
    "place_include_gps_confirm": {
        "ru": "{n} файлов уже имеют координаты из камеры — они не тронуты. "
              "Перезаписать место и у них?",
        "en": "{n} files already carry camera coordinates and were left alone. "
              "Overwrite their place too?",
        "ja": "{n} 件はカメラの座標を持つためそのままです。これらの場所も上書きしますか？",
    },
    "place_manual_mark": {
        "ru": "место назначено вручную", "en": "place assigned by hand",
        "ja": "場所は手動指定",
    },
    "place_hint": {
        "ru": "Место назначается группе целиком — событию или исходной папке. Оно "
              "переживает пересчёт гео и видно в плане как «вручную».",
        "en": "A place is assigned to a whole group — an event or a source folder. It "
              "survives a geo recompute and shows up in the plan as «manual».",
        "ja": "場所はグループ単位（イベントまたは元フォルダ）で指定します。位置情報の"
              "再計算後も残り、プランには「手動」と表示されます。",
    },
    "place_error_prefix": {
        "ru": "Не удалось назначить место: ", "en": "Could not assign the place: ",
        "ja": "場所を指定できません: ",
    },
    # F103: the "Utility frames" view — the buckets the classifier carries out of
    # the collection, and the bulk way back for the frames it got wrong.
    # F175: the caption of the WHOLE slice names no percentage, and deliberately. Behind
    # one name lie four buckets measured separately (products 78%, screenshots 59%,
    # documents and memes not measured at all), and a single number over them would be
    # honest about none of them. What it does say is the thing a person has to know
    # before ticking everything: one of the four is not to be deleted.
    "junk_intro": {
        "ru": "Кадры, снятые не ради памяти: товары, скриншоты, документы, мемы. Это "
              "четыре разные корзины с разной надёжностью — откройте любую, и подпись "
              "назовёт её точность. Документы удалять нельзя: там паспорта, справки и "
              "чеки. Отметьте кадры, попавшие сюда зря, и верните их — они снова "
              "разложатся по городам. Вердикт модели при этом не переписывается.",
        "en": "Frames that were not taken for memory: products, screenshots, documents, "
              "memes. These are four different buckets of different reliability — open "
              "any one of them and the caption names its precision. The documents are "
              "not to be deleted: passports, certificates and receipts live there. Tick "
              "the frames that landed here by mistake and return them — they go back "
              "into the city layout. The model's verdict itself is not rewritten.",
        "ja": "思い出のためではなく実用のために撮られたコマです: 商品、"
              "スクリーンショット、書類、ミーム。信頼度の異なる 4 つの別々のバケットで、"
              "いずれかを開くとその精度が説明に出ます。書類は削除できません — "
              "パスポート、証明書、レシートが入っています。誤って入ったコマに"
              "チェックを入れて戻すと、再び都市ごとに振り分けられます。モデルの"
              "判定自体は書き換えません。",
    },
    # F175: precision belongs to a CLASS, not to the slice. Each line below is one
    # measurement with its date and its sample size, shown when that bucket is the one
    # open. A class nobody has measured gets `junk_accuracy_unmeasured` — the lookup in
    # the client falls back to it, so a class added later says "not measured" instead of
    # quietly inheriting somebody else's number.
    "junk_accuracy_product": {
        "ru": "Точность 78% при полноте 81% (замер 2026-08-03, 999 кадров): примерно "
              "каждый пятый кадр здесь — не товар.",
        "en": "Precision 78% at 81% recall (measured 2026-08-03 on 999 frames): about "
              "one frame in five here is not a product.",
        "ja": "精度 78%、再現率 81%（2026-08-03、999 コマで測定）: ここにあるコマの"
              "およそ 5 枚に 1 枚は商品ではありません。",
    },
    # F171: this bucket states an OPINION and has to be read as one. The rescue of
    # 2026-08-04 added 441 frames to it (1 782 against 1 341) and 41% of what it adds is
    # an ordinary photograph — about 181 personal pictures leaving the city layout for a
    # bucket a person reads as "these are your screenshots" and does not look through.
    # So the caption names the model as the author of the verdict, and names returning a
    # frame as the ordinary next step rather than as the repair of a rare mistake.
    "junk_accuracy_screenshot": {
        "ru": "Модель считает эти кадры экранными — это её оценка, а не факт. Точность "
              "59% при полноте 83% (замер 2026-08-03, 350 кадров): каждый "
              "третий кадр здесь — обычная фотография. Просмотрите список перед "
              "удалением и верните такие кадры в раскладку — здесь это обычный шаг "
              "работы, а не исправление редкой ошибки.",
        "en": "The model considers these frames screen captures — that is its estimate "
              "and not a fact. Precision 59% at 83% recall (measured 2026-08-03 on 350 "
              "frames): every third frame here is an ordinary photograph. Look the list "
              "over before deleting anything and return such frames to the layout — "
              "here that is an ordinary step of the work, not the repair of a rare "
              "mistake.",
        "ja": "モデルはこれらのコマを画面のコマだと考えています — 事実ではなく推定です。"
              "精度 59%、再現率 83%（2026-08-03、350 コマで測定）: ここにあるコマの"
              "3 枚に 1 枚は普通の写真です。削除する前にリストを見て、そうしたコマは"
              "振り分けに戻してください — ここではそれが通常の作業であり、まれな誤りの"
              "修正ではありません。",
    },
    "junk_accuracy_unmeasured": {
        "ru": "Точность этой корзины не измерена — сколько здесь ошибок, неизвестно.",
        "en": "The precision of this bucket has not been measured — how many frames "
              "here are wrong is not known.",
        "ja": "このバケットの精度は測定されていません — 誤りがどれだけあるかは"
              "分かりません。",
    },
    # F171: appended to the caption of the bucket that is open, and ONLY where the server
    # says the page was actually ordered by the model's own estimate (`ordered_by_score`).
    # A promise about the order that is true on one collection and silent on another is
    # the F157 rule: the sentence appears exactly where the ordering it describes does.
    "junk_order_hint": {
        "ru": " Список идёт от кадров, в которых модель уверена больше, к сомнительным: "
              "читайте сверху и остановитесь, где сходство кончилось.",
        "en": " The list runs from the frames the model is most sure of down to the "
              "doubtful ones: read from the top and stop where the resemblance ends.",
        "ja": " リストはモデルの確信が強いコマから弱いコマへ並びます。上から読み、"
              "似ていると思えなくなった所で止めてください。",
    },
    "junk_bucket_product": {"ru": "Товары", "en": "Products", "ja": "商品"},
    "junk_bucket_document": {"ru": "Документы", "en": "Documents", "ja": "書類"},
    "junk_bucket_screenshot": {"ru": "Скриншоты", "en": "Screenshots",
                               "ja": "スクリーンショット"},
    "junk_bucket_meme": {"ru": "Мемы", "en": "Memes", "ja": "ミーム"},
    "junk_empty": {
        "ru": "Здесь пусто — таких кадров нет.",
        "en": "Nothing here — there are no such frames.",
        "ja": "ここは空です。該当するフレームはありません。",
    },
    "junk_restore_confirm": {
        "ru": "{n} кадров вернутся в раскладку: {breakdown}. Продолжить?",
        "en": "{n} frames will return to the layout: {breakdown}. Continue?",
        "ja": "{n} 件が振り分けに戻ります: {breakdown}。続けますか？",
    },
    "junk_undo_restore_button": {
        "ru": "Отменить возврат", "en": "Undo the return", "ja": "戻すのを取り消す",
    },
    # F174: nothing has moved yet — the mark applies on the next `sort --apply`, and a
    # past tense here would promise a transfer that has not happened.
    "junk_restored_mark": {
        "ru": "вернётся в раскладку", "en": "will return to the layout",
        "ja": "振り分けに戻ります",
    },
    "junk_select_all": {"ru": "Выбрать всё на странице",
                        "en": "Select everything on this page",
                        "ja": "このページをすべて選択"},
    "junk_select_none": {"ru": "Снять выделение", "en": "Clear the selection",
                         "ja": "選択を解除"},
    "junk_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    "junk_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    "junk_document_no_preview": {
        "ru": "без превью", "en": "no preview", "ja": "プレビューなし",
    },
    # F175: the hint says what a document IS before it says how it is shown. This slice
    # reads as "junk, select all, delete", and the frames the sentence is about are the
    # ones a person needs most — so the warning has to arrive above the grid, before the
    # selection, and not as an explanation of a missing thumbnail.
    "junk_document_hint": {
        "ru": "Документы здесь — не на удаление: это паспорта, справки, чеки и "
              "медицинские бланки, и они помечены отдельно. Sorta их не открывает и не "
              "показывает; видно имя файла и дату — этого хватает, чтобы решить.",
        "en": "The documents here are not for deletion: they are passports, "
              "certificates, receipts and medical forms, and they are marked out "
              "separately. Sorta neither opens nor renders them; the file name and the "
              "date are shown — enough to decide.",
        "ja": "ここにある書類は削除の対象ではありません: パスポート、証明書、"
              "レシート、診断書であり、別に印が付いています。Sorta はそれらを開かず"
              "表示もしません。判断にはファイル名と日付で十分です。",
    },
    # The same warning ON the card, because the hint above the grid is read once and the
    # selection is made card by card.
    "junk_document_mark": {
        "ru": "не удалять", "en": "not for deletion", "ja": "削除しない",
    },
    "junk_error_prefix": {
        "ru": "Не удалось вернуть кадры: ", "en": "Could not return the frames: ",
        "ja": "フレームを戻せません: ",
    },
    "error_loading_junk": {
        "ru": "Не удалось загрузить корзины: ", "en": "Could not load the buckets: ",
        "ja": "バケットを読み込めません: ",
    },
    # --- F174: the action names its destination ---------------------------------------
    # ONE name for one intention. "This is not an animal" and "return to the photos" are
    # the same movement to the person making it — the frame does not belong in this slice
    # — so the button reads the same in both, and what differs (a real transfer versus a
    # membership) is said UNDER it, in `dest_goes_to` / `dest_stays_in`. Two buttons for
    # one intention was the whole complaint.
    "slice_return_button": {
        "ru": "Вернуть в раскладку", "en": "Return to the layout", "ja": "振り分けに戻す",
    },
    "dest_goes_to": {
        "ru": "попадёт в {folder}", "en": "goes into {folder}",
        "ja": "{folder} に入ります",
    },
    "dest_stays_in": {
        "ru": "уберём из среза; кадр и так лежит в {folder}, файл не двинется",
        "en": "we take it out of the slice; the frame already lies in {folder}, "
              "the file will not move",
        "ja": "区分から外すだけです。コマはすでに {folder} にあり、ファイルは動きません",
    },
    "dest_unknown": {
        "ru": "папку назначения посчитать не удалось",
        "en": "the destination folder could not be computed",
        "ja": "保存先フォルダーを計算できませんでした",
    },
    # Looked up as `dest_why_<reason>` — the plan's own reason codes. A reason without a
    # key simply gets no explanation, the way an unknown bucket gets no chip label.
    "dest_why_no_place": {
        "ru": "у кадра нет геоданных", "en": "the frame carries no geodata",
        "ja": "このコマに位置情報がありません",
    },
    "dest_why_country_only": {
        "ru": "город не определился — известна только страна",
        "en": "no city resolved — only the country is known",
        "ja": "都市は不明で、国だけが分かっています",
    },
    "dest_why_low_date": {
        "ru": "у кадра нет надёжной даты съёмки",
        "en": "the frame carries no reliable capture date",
        "ja": "このコマに信頼できる撮影日がありません",
    },
    "dest_why_downloaded": {
        "ru": "ни даты съёмки, ни следов камеры — это скачанный кадр",
        "en": "no capture date and no camera trace — a downloaded frame",
        "ja": "撮影日もカメラの痕跡もありません。ダウンロードされたコマです",
    },
    # The bulk caption groups by destination instead of naming one folder: a person
    # selects dozens at a time, and one folder name out of twelve deceives them.
    "dest_bulk_summary": {
        "ru": "{n} кадров вернутся: {breakdown}",
        "en": "{n} frames will return: {breakdown}",
        "ja": "{n} 件が戻ります: {breakdown}",
    },
    "dest_bulk_item": {"ru": "{n} {group}", "en": "{n} {group}", "ja": "{n} 件 {group}"},
    "dest_group_city": {"ru": "в города", "en": "into cities", "ja": "都市へ"},
    "dest_group_country": {
        "ru": "на уровень страны", "en": "to the country level", "ja": "国のレベルへ",
    },
    "dest_group_no_place": {
        "ru": "в «без места»", "en": "into “no place”", "ja": "「場所不明」へ",
    },
    "dest_group_undated": {
        "ru": "в «без даты»", "en": "into “no date”", "ja": "「日付不明」へ",
    },
    "dest_group_other": {
        "ru": "в другие папки", "en": "into other folders", "ja": "その他のフォルダーへ",
    },
    # --- F123: the "Animals" tab -----------------------------------------------------
    # F160: the caption states BOTH measurements, because the slice is two different
    # promises and a config switch chooses between them. The cascade alone is 82%
    # precision at 64% recall; with the object detector on (`features.detector` +
    # `detect.enabled`) it is 62% at 87% — a quarter more animals found and a fifth of the
    # confidence given up for them. A caption naming one number while the other rule is in
    # force buys recall with the reader's trust, which is the mistake F158 measured on the
    # very same line.
    "animals_intro": {
        "ru": "Кадры с животными, сверху — те, в которых модель уверена больше. "
              "Точность около 82% при полноте 64%; с включённым детектором объектов "
              "(features.detector) размен другой — точность около 62% при полноте 87%: "
              "животных находится заметно больше, а шуб и игрушек среди них тоже. "
              "Ниже по списку видно, где проходит граница.",
        "en": "Frames with animals, the ones the model is most confident about first. "
              "Precision is about 82% at 64% recall; with the object detector on "
              "(features.detector) the trade is a different one — about 62% precision at "
              "87% recall: noticeably more animals found, and more fur coats and plush "
              "toys among them. Further down the list is where the border of confidence "
              "runs.",
        "ja": "動物が写ったコマです。モデルの確信度が高い順に並びます。精度は約 82%、"
              "再現率は 64% です。物体検出を有効にすると (features.detector) 精度は約 "
              "62%、再現率は 87% になり、見つかる動物は増えますが毛皮のコートや"
              "ぬいぐるみも増えます。下に行くほど確信度の境目が見えてきます。",
    },
    "animals_empty": {
        "ru": "Здесь пусто — животные не найдены.",
        "en": "Nothing here — no animals were found.",
        "ja": "ここは空です。動物は見つかりませんでした。",
    },
    "animals_score_label": {
        "ru": "уверенность {score}", "en": "confidence {score}", "ja": "確信度 {score}",
    },
    # F173: the button and the counter of this slice are `slice_load_more` /
    # `slice_shown_label` now — the shared pager's, like every other ordered list.
    "error_loading_animals": {
        "ru": "Не удалось загрузить животных: ", "en": "Could not load the animals: ",
        "ja": "動物を読み込めません: ",
    },
    # --- F124: taking a false mark off a frame (and putting a missing one back) --------
    # The two buttons are one toggle: the card offers the answer the frame does NOT have
    # right now. The third string is the way back to the automatic verdict, which is a
    # different thing from "not an animal" and therefore says so in words.
    # F174: the "take it off this frame" half is `slice_return_button` now — the same
    # words the junk view uses, because it is the same intention. What the two do differ
    # in is stated under the button (`dest_stays_in` here, `dest_goes_to` there).
    "animals_mark_animal": {
        "ru": "Это животное", "en": "This is an animal", "ja": "これは動物",
    },
    "animals_mark_clear": {
        "ru": "Вернуть автоматически", "en": "Back to automatic", "ja": "自動判定に戻す",
    },
    "animals_manual_excluded": {
        "ru": "снято вручную", "en": "unmarked by hand", "ja": "手動で解除",
    },
    "animals_manual_included": {
        "ru": "отмечено вручную", "en": "marked by hand", "ja": "手動で設定",
    },
    "animals_counted_label": {
        "ru": "Животных: {n}", "en": "Animals: {n}", "ja": "動物: {n} 件",
    },
    "animals_error_prefix": {
        "ru": "Не удалось сохранить отметку: ", "en": "Could not save the mark: ",
        "ja": "マークを保存できません: ",
    },
    # --- F126: the "Review" workspace -------------------------------------------------
    # The switcher labels are the slices; the duplicates one keeps the wording the tab
    # had, because that is what the user has been calling it since U3.
    "review_slice_dupes": {"ru": "Дубли", "en": "Duplicates", "ja": "重複"},
    "review_slice_blurred": {"ru": "Размытые", "en": "Blurred", "ja": "ぼやけ"},
    "review_slice_eyes": {"ru": "Закрытые глаза", "en": "Closed eyes", "ja": "目を閉じた"},
    # F150: named after the FACT and never after a judgement. "Bad" or "junk" would be a
    # verdict the program has no business passing on a picture somebody was sent once and
    # never got again.
    "review_slice_low_resolution": {
        "ru": "Низкое разрешение", "en": "Low resolution", "ja": "低解像度",
    },
    "review_intro": {
        "ru": "Одно место для всего, что надо просмотреть глазами и частью удалить. "
              "Отметка «удалить» — это пометка, а не удаление: файлы уедут в папку "
              "«_удалить» на следующей раскладке. Отметка «оставить» переживает "
              "пересчёт и больше не спросится.",
        "en": "One place for everything that has to be looked at by eye and partly "
              "deleted. Marking “delete” is a mark, not a deletion: those files go to "
              "the “_delete” folder on the next layout. A “keep” survives a recompute "
              "and is not asked about again.",
        "ja": "目で確認して一部を削除する作業を、ここ一か所にまとめています。「削除」は"
              "印であって削除ではありません。対象は次回の振り分けで「_削除」フォルダへ"
              "移ります。「残す」は再計算後も保持され、再び尋ねられません。",
    },
    # F157: the caption of a RANKING. It used to describe a window ("the list opens down
    # to 90"), which read as a verdict about the frames inside it — and the number behind
    # that reading catches 12% of what a person calls blurred. The list is now ordered from
    # the softest frame, `{max}` is only how far the first page reaches, and the sentence
    # says the two things a reader has to know: read from the top and stop where the
    # resemblance ends, and this number cannot tell a detailed sharp street from a smooth
    # blurred face. The "delete everything below the threshold" line stays: a button this
    # feature deliberately does not have has to be named, or somebody adds it.
    "review_hint_blurred": {
        "ru": "Это порядок, а не приговор. Сверху кадры, которые почти наверняка "
              "смазаны; ниже резкость растёт, и где-то начинаются нормальные "
              "фотографии — читайте сверху вниз и остановитесь, где сходство кончилось. "
              "Первая страница открыта до резкости {max}, «показать ещё» идёт дальше по "
              "списку. Признак грубый: детализированная резкая улица и гладкое размытое "
              "лицо дают близкие числа, поэтому кнопки «удалить всё ниже порога» здесь "
              "нет и по умолчанию не удаляется ничего.",
        "en": "This is an order, not a verdict. At the top are frames that are almost "
              "certainly smeared; further down the sharpness grows and at some point "
              "ordinary photographs begin — read from the top and stop where the "
              "resemblance ends. The first page opens down to a sharpness of {max}, and "
              "“show more” simply continues down the list. The signal is coarse: a "
              "detailed sharp street and a smooth blurred face score alike, so there is "
              "no “delete everything below the threshold” button here and nothing is "
              "marked by default.",
        "ja": "これは判定ではなく並び順です。上にあるのはほぼ確実にぶれているコマで、"
              "下にいくほど鮮鋭度は上がり、どこかで普通の写真が始まります。上から読み、"
              "似ていると思えなくなった所で止めてください。最初のページは鮮鋭度 {max} "
              "まで開き、「さらに表示」はその先へ続きます。この指標は粗いものです。"
              "細部の多い鮮明な街並みと、なめらかにぼけた顔は近い値になるため、"
              "「しきい値以下をすべて削除」というボタンはなく、既定では何も削除しません。",
    },
    # F155 + F157: shown only where `frame_quality.face_sharpness` exists, because only
    # there is it true. It is the answer to "why is this sharp-looking street above that
    # soft portrait": the frames with a face are ordered by a different number, measured
    # inside the face, which finds 62% of the blurred ones against 15% for the whole frame.
    "review_hint_blurred_faces": {
        "ru": " Кадры с лицами идут первыми и упорядочены по резкости самого лица — "
              "по кадру целиком этот признак их почти не находит.",
        "en": " Frames with a face come first and are ordered by the sharpness measured "
              "inside the face — over the whole frame this signal barely finds them.",
        "ja": " 顔のあるコマが先に並び、顔の内側で測った鮮鋭度で順序づけられます。"
              "コマ全体で測ると、この指標はそれらをほとんど拾えません。",
    },
    # F179: the caption states the MEASURED PRECISION and not a count. "Found 730 frames"
    # reads as a verdict about 730 photographs; on 249 hand-labelled frames this list is
    # right about 62% of what it points at, which is the one thing a person needs to know
    # before opening it. The list is ordered from the most closed, so the top is where the
    # 62% lives and "show more" walks into the doubtful part on purpose.
    "review_hint_eyes": {
        "ru": "Кадры, на которых у людей, скорее всего, закрыты глаза: посчитано по "
              "геометрии век самого крупного лица, а не спрошено у модели. На 249 "
              "размеченных кадрах такой список верен в 62% случаев — каждый третий кадр "
              "здесь на самом деле с открытыми глазами, поэтому ничего не удаляется само. "
              "Сверху самые закрытые; «показать ещё» продолжает список за порог {max}, в "
              "сомнительную часть. Мерится только там, где найдено лицо.",
        "en": "Frames where the people most likely have their eyes closed — computed from "
              "the eyelid geometry of the largest face, not asked of a model. On 249 "
              "hand-labelled frames a list like this is right 62% of the time: one frame "
              "in three here actually has its eyes open, so nothing is ever deleted "
              "automatically. The most closed come first, and “show more” continues past "
              "the {max} mark into the doubtful part. Measured only where a face was found.",
        "ja": "最も大きい顔のまぶたの形状から算出した、目を閉じている可能性が高いコマです"
              "（モデルへの質問ではありません）。手作業でラベル付けした 249 コマでは、この"
              "一覧の正解率は 62% です。3 コマに 1 コマは実際には目が開いているため、自動的"
              "な削除は行いません。閉じている度合いの高い順に並び、「もっと見る」はしきい値 "
              "{max} を越えて確度の低い部分へ進みます。顔が検出されたコマのみで計測します。",
    },
    "review_eyes_no_faces": {
        "ru": "Данных нет: стадия «лица» не запускалась, а глаза мерятся только там, где "
              "найдено лицо. Прогоните лица и повторите разбор.",
        "en": "No data: the faces stage never ran, and the eyes are only measured where a "
              "face was found. Run faces and come back to this slice.",
        "ja": "データがありません。顔ステージが実行されておらず、目の計測は顔が検出された"
              "コマにのみ行われます。顔ステージを実行してから、この区分を開いてください。",
    },
    # F150: the whole boundary of the slice, said out loud. Three things a person has to
    # know before pressing anything here: it is a fact and not a verdict (the frame may be
    # the only copy of something), megapixels say nothing about a big frame ruined by
    # compression, and videos are not in this list at all.
    "review_hint_low_resolution": {
        "ru": "Кадры меньше {mp} мегапикселя, сначала самые мелкие. Это факт из индекса, "
              "а не оценка: ширина и высота записаны при индексации, ничего не "
              "измерялось. Малое разрешение — не признак брака: это может быть "
              "единственная сохранившаяся фотография, присланная десять лет назад, "
              "поэтому по умолчанию не удаляется ничего. Пережатое сюда не попадает: "
              "кадр 4000×3000, убитый JPEG-артефактами, формально большой — это другой "
              "сигнал и другой разговор. Видео не считаем.",
        "en": "Frames smaller than {mp} megapixels, the smallest first. This is a fact "
              "out of the index rather than an estimate: width and height were written "
              "down when the file was indexed and nothing was measured. A small frame is "
              "not a faulty one — it can be the only surviving photograph, sent ten "
              "years ago — so nothing is marked for deletion by default. Over-compressed "
              "frames are not here: a 4000×3000 picture ruined by JPEG artefacts is "
              "formally large, and that is a different signal and a different "
              "conversation. Videos are not counted.",
        "ja": "{mp} メガピクセル未満のコマを、小さい順に並べています。これは推定ではなく"
              "索引に記録された事実です。幅と高さは登録時に書き込まれたもので、何も"
              "測定していません。解像度が低いことは欠陥ではありません — 十年前に送られて"
              "きた唯一の一枚かもしれないので、既定では何も削除の印を付けません。"
              "圧縮で潰れたコマはここには入りません。JPEG のノイズで壊れた 4000×3000 の"
              "画像は形式上は大きく、それは別の指標であり別の話です。動画は数えません。",
    },
    # The size of the picture, as a person reads it off a camera: the two sides and the
    # megapixels they come to.
    "review_resolution_label": {
        "ru": "{w}×{h} ({mp} Мп)", "en": "{w}×{h} ({mp} MP)", "ja": "{w}×{h}（{mp} MP）",
    },
    "review_empty": {
        "ru": "Здесь пусто — таких кадров нет.",
        "en": "Nothing here — there are no such frames.",
        "ja": "ここは空です。該当するフレームはありません。",
    },
    "review_sharpness_label": {
        "ru": "резкость {value}", "en": "sharpness {value}", "ja": "鮮鋭度 {value}",
    },
    "review_mark_delete": {
        "ru": "Пометить на удаление", "en": "Mark for deletion", "ja": "削除の印を付ける",
    },
    "review_mark_keep": {"ru": "Оставить", "en": "Keep", "ja": "残す"},
    "review_mark_clear": {
        "ru": "Снять отметку", "en": "Clear the mark", "ja": "印を外す",
    },
    # --- F149: "try to improve" — the third action, on one frame ----------------------
    # Every string here says PROCESSED, never "improved". The model draws something
    # plausible instead of bringing back what was lost, and an interface that calls that
    # an improvement is the one thing this feature must not do: a person has to know at
    # every moment which of the two pictures in front of them is the photograph.
    "review_restore": {
        "ru": "Попробовать улучшить", "en": "Try to improve", "ja": "補正を試す",
    },
    "review_restore_hint": {
        "ru": "Один кадр за раз: выберите ровно один. Модель НЕ возвращает утраченное — "
              "она дорисовывает правдоподобное, поэтому рядом появится помеченная "
              "копия, а оригинал останется как есть. Первое нажатие качает веса "
              "(~400 МБ), дальше — около секунды на кадр.",
        "en": "One frame at a time: select exactly one. The model does NOT bring back "
              "what was lost — it draws something plausible — so what appears beside the "
              "original is a marked copy, and the original stays as it is. The first "
              "press downloads the weights (~400 MB), after that it is about a second "
              "per frame.",
        "ja": "一度に 1 枚だけです。ちょうど 1 枚を選んでください。モデルは失われた情報を"
              "復元するのではなく、それらしく描き足します。そのため元の写真の隣には印の"
              "付いた複製が現れ、元の写真はそのまま残ります。初回は重み (約 400 MB) を"
              "取得し、その後は 1 枚あたり約 1 秒です。",
    },
    "review_restore_running": {
        "ru": "Обрабатываем кадр…", "en": "Processing the frame…", "ja": "処理中…",
    },
    "review_restore_badge": {
        "ru": "обработано моделью", "en": "processed by a model", "ja": "モデルによる処理",
    },
    "review_restore_badge_hint": {
        "ru": "Это НЕ фотография, а копия, дорисованная моделью: детали на ней "
              "правдоподобные, но выдуманные. Оригинал не изменён и лежит рядом. "
              "Оставить можно любую, обе или ни одной — выбор копии сам по себе ничего "
              "не помечает на удаление.",
        "en": "This is NOT a photograph but a copy a model drew over: its detail is "
              "plausible and invented. The original is unchanged and lies beside it. Keep "
              "either, both or neither — choosing the copy marks nothing for deletion by "
              "itself.",
        "ja": "これは写真ではなく、モデルが描き足した複製です。細部はそれらしく見えますが"
              "作られたものです。元の写真は変更されず隣にあります。どちらを残しても、"
              "両方でも、どちらも残さなくても構いません。複製を選んでも、それだけでは"
              "何も削除対象になりません。",
    },
    "review_restore_done": {
        "ru": "Готово: обработанная копия рядом с оригиналом. Оригинал не изменён.",
        "en": "Done: the processed copy is beside the original. The original is unchanged.",
        "ja": "完了しました。処理済みの複製が元の写真の隣にあります。元の写真は変更されて"
              "いません。",
    },
    "review_restore_reused": {
        "ru": "Такая копия уже была — показываем её, второй не делаем.",
        "en": "That copy already existed — here it is; a second one is not made.",
        "ja": "その複製はすでに存在します。既存のものを表示し、二つ目は作りません。",
    },
    # F169: the sentence a full-sized frame is owed. The model is x4 and cannot be shown
    # the whole frame, so a big one is REDUCED first and blown back up to about its own
    # size — the copy comes out the same size and holds less of what was really there.
    # Said next to "done", every time it happens, because it is the one outcome a person
    # cannot see by looking: the copy usually looks sharper, and sharper is not truer.
    "review_restore_rebuilt": {
        "ru": "Внимание: кадр больше предела ({max_edge} px по длинной стороне, здесь "
              "{source_edge}). Копия пересобрана из уменьшенной: настоящая детализация "
              "оригинала не попала в модель, и на её месте дорисована правдоподобная. "
              "Это не улучшение оригинала — предел меняется ключом "
              "features.restore_max_edge.",
        "en": "Note: this frame is larger than the limit ({max_edge} px on the longer "
              "side, this one is {source_edge}). The copy was rebuilt from a reduced "
              "frame: the real detail of the original never reached the model, and "
              "plausible detail was drawn in its place. This is not an improved original "
              "— the limit is the features.restore_max_edge key.",
        "ja": "注意: このコマは上限 (長辺 {max_edge} px、このコマは {source_edge} px) を"
              "超えています。複製は縮小した画像から作り直されました。元の写真の本当の"
              "細部はモデルに渡らず、代わりにそれらしい細部が描き足されています。"
              "元の写真が良くなったわけではありません。上限は "
              "features.restore_max_edge で変えられます。",
    },
    # --- F168: the same action, reached from the expanded frame in any slice ----------
    # The hint says the same things as `review_restore_hint` minus the one sentence that
    # belongs to the Review grid ("select exactly one"): here the frame IS the one being
    # looked at, and there is nothing to select.
    "review_restore_expanded_hint": {
        "ru": "Модель НЕ возвращает утраченное — она дорисовывает правдоподобное. Рядом "
              "с оригиналом появится помеченная копия, а оригинал останется как есть. "
              "Первое нажатие качает веса (~400 МБ), дальше — около секунды на кадр.",
        "en": "The model does NOT bring back what was lost — it draws something "
              "plausible — so what appears beside the original is a marked copy, and the "
              "original stays as it is. The first press downloads the weights (~400 MB), "
              "after that it is about a second per frame.",
        "ja": "モデルは失われた情報を復元するのではなく、それらしく描き足します。その"
              "ため元の写真の隣には印の付いた複製が現れ、元の写真はそのまま残ります。"
              "初回は重み (約 400 MB) を取得し、その後は 1 枚あたり約 1 秒です。",
    },
    # F168/F169: why the action is NOT offered on a big frame. The gain the measurement
    # found belongs to small frames (66% under 640 px, a coin toss by 1280), and above the
    # ceiling the copy would be rebuilt from a quarter of the original. Withdrawing the
    # button without a word would be the silent half of the same promise.
    "review_restore_too_large": {
        "ru": "Кадр крупнее предела ({max_edge} px по длинной стороне, здесь "
              "{source_edge}): копию пришлось бы пересобирать из уменьшенной, а на таких "
              "кадрах замер пользы не показал. Поэтому здесь действие не предлагается — "
              "предел меняется ключом features.restore_max_edge.",
        "en": "This frame is larger than the limit ({max_edge} px on the longer side, "
              "this one is {source_edge}): the copy would be rebuilt from a reduced "
              "frame, and on frames this size the measurement found no gain. So the "
              "action is not offered here — the limit is the features.restore_max_edge "
              "key.",
        "ja": "このコマは上限 (長辺 {max_edge} px、このコマは {source_edge} px) を超えて"
              "います。複製は縮小した画像から作り直すことになり、この大きさのコマでは"
              "効果が確認できませんでした。そのためここでは操作を提供しません。上限は "
              "features.restore_max_edge で変えられます。",
    },
    # The copy is a canonical file: it lies in the city folder beside its source and turns
    # up in every slice the source does. Wherever it is opened it says what it is and
    # which frame it was made from — otherwise it reads as a second similar photograph
    # that came from nowhere.
    "review_restore_source_badge": {
        "ru": "обработано моделью из {name}",
        "en": "processed by a model from {name}",
        "ja": "{name} をモデルで処理した複製",
    },
    "review_restore_error_sensitive_class": {
        "ru": "Кадр отнесён к личным документам (vlm.exclude_classes): такие кадры "
              "продукт не разворачивает и не обрабатывает. Ничего не создано.",
        "en": "This frame is classed as a personal document (vlm.exclude_classes): the "
              "product neither enlarges nor processes those. Nothing was created.",
        "ja": "このコマは個人的な書類 (vlm.exclude_classes) に分類されています。"
              "拡大も処理も行いません。何も作成されていません。",
    },
    "review_restore_error_video": {
        "ru": "Это видео, а модель работает с изображениями. Ничего не создано.",
        "en": "This is a video and the model works on images. Nothing was created.",
        "ja": "これは動画で、モデルは画像を扱います。何も作成されていません。",
    },
    "review_restore_error_model_unavailable": {
        "ru": "Модель не загрузилась. Веса качаются из сети и нужен дополнительный "
              "набор пакетов ([vlm]); офлайн и без скачанных весов эта кнопка работать "
              "не будет. Ничего не создано.",
        "en": "The model did not load. The weights come from the network and need the "
              "extra package set ([vlm]); offline and without cached weights this button "
              "cannot work. Nothing was created.",
        "ja": "モデルを読み込めませんでした。重みはネットワークから取得され、追加の"
              "パッケージ ([vlm]) が必要です。オフラインで重みが未取得の場合、この"
              "ボタンは動作しません。何も作成されていません。",
    },
    "review_restore_error_decode_failed": {
        "ru": "Кадр не читается — обрабатывать нечего. Ничего не создано.",
        "en": "The frame will not read — there is nothing to process. Nothing was created.",
        "ja": "このコマを読み込めないため、処理できません。何も作成されていません。",
    },
    "review_restore_error_write_failed": {
        "ru": "Копию не удалось записать рядом с оригиналом. Оригинал не изменён.",
        "en": "The copy could not be written beside the original. The original is unchanged.",
        "ja": "元の写真の隣に複製を書き込めませんでした。元の写真は変更されていません。",
    },
    "review_select_label": {"ru": "выбрать", "en": "select", "ja": "選択"},
    "review_select_all": {"ru": "Выбрать всё на странице",
                          "en": "Select everything on this page",
                          "ja": "このページをすべて選択"},
    "review_select_none": {"ru": "Снять выделение", "en": "Clear the selection",
                           "ja": "選択を解除"},
    "review_marked_status": {
        "ru": "Отмечено кадров: {n}", "en": "Frames marked: {n}", "ja": "印を付けたコマ: {n}",
    },
    "review_load_more": {"ru": "Показать ещё", "en": "Show more", "ja": "さらに表示"},
    "review_load_more_beyond": {
        "ru": "Показать за пределами окна", "en": "Show past the window",
        "ja": "表示範囲の先も表示",
    },
    "review_shown_label": {
        "ru": "Показано {shown} из {total}", "en": "Showing {shown} of {total}",
        "ja": "{total} 件中 {shown} 件を表示",
    },
    # F157: the counter of a ranking says how long the LIST is and never how many blurred
    # frames there are. "Showing 2 210" read as "you have 2 210 blurred photographs" —
    # a claim the signal cannot make (four of five frames on that page are not blurred),
    # and one that grows or shrinks the moment somebody edits a number in the config.
    "review_shown_ranked": {
        "ru": "Показано {shown}; дальше по списку резкость растёт",
        "en": "Showing {shown}; further down the list the sharpness grows",
        "ja": "{shown} 件を表示中。リストの先へ進むほど鮮鋭度は上がります",
    },
    "review_error_prefix": {
        "ru": "Не удалось сохранить отметку: ", "en": "Could not save the mark: ",
        "ja": "印を保存できません: ",
    },
    "error_loading_review": {
        "ru": "Не удалось загрузить разбор: ", "en": "Could not load the review: ",
        "ja": "仕分けを読み込めません: ",
    },
    # --- F108: the "Overview" tab ---------------------------------------------------
    "tab_overview": {"ru": "Обзор", "en": "Overview", "ja": "概要"},
    # F145: the caption over the SAME rows of counters, drawn with dashes. It replaced an
    # invitation with a button, which was a block of a different height: it was swapped
    # for the full one in the middle of a run, right after the `index` stage, and
    # everything below — the run options among them — jumped down the page.
    "overview_empty": {
        "ru": "Данных пока нет: ниже — то, что появится после прогона. "
              "Укажите папку с фото и нажмите «Обработать».",
        "en": "No data yet: below is what shows up after a run. Enter a photo folder "
              "and click Process.",
        "ja": "まだデータがありません。以下は処理後に表示される項目です。"
              "写真フォルダを指定して「処理する」を押してください。",
    },
    "overview_group_collection": {"ru": "Коллекция", "en": "Collection", "ja": "コレクション"},
    "overview_group_place": {"ru": "Место", "en": "Place", "ja": "場所"},
    "overview_group_classes": {"ru": "Разбор", "en": "Classification", "ja": "分類"},
    "overview_group_layout": {"ru": "Раскладка", "en": "Layout", "ja": "振り分け"},
    "overview_files": {"ru": "Файлов в индексе", "en": "Files in the index",
                       "ja": "インデックス内のファイル"},
    "overview_photos": {"ru": "Фото", "en": "Photos", "ja": "写真"},
    "overview_videos": {"ru": "Видео", "en": "Videos", "ja": "動画"},
    "overview_duplicates": {"ru": "Дубликатов", "en": "Duplicates", "ja": "重複"},
    "overview_errors": {"ru": "Ошибок чтения", "en": "Read errors", "ja": "読み込みエラー"},
    "overview_events": {"ru": "Событий", "en": "Events", "ja": "イベント"},
    "overview_animals": {"ru": "С животными", "en": "With animals", "ja": "動物あり"},
    # F152: the three face slices. They are the only rows of this card that can show a
    # dash instead of a number — without a faces run they are unmeasured, not empty.
    "overview_with_people": {"ru": "С людьми", "en": "With people", "ja": "人物あり"},
    "overview_group_photos": {"ru": "Групповых", "en": "Group photos", "ja": "集合写真"},
    "overview_portraits": {"ru": "Портретов", "en": "Portraits", "ja": "ポートレート"},
    # F126: the review slices that have a number of their own. Blurred is counted
    # inside the window the list opens to, so the row and the list agree.
    "overview_blurred": {"ru": "Размытых", "en": "Blurred", "ja": "ぼやけ"},
    "overview_eyes_closed": {"ru": "С закрытыми глазами", "en": "With closed eyes",
                             "ja": "目を閉じた"},
    # F150: counted under `features.low_resolution_mp`, the same ceiling the slice lists.
    "overview_low_resolution": {"ru": "Низкого разрешения", "en": "Low resolution",
                                "ja": "低解像度"},
    "overview_place_exact_gps": {"ru": "Точный GPS", "en": "Exact GPS", "ja": "正確なGPS"},
    "overview_place_manual": {"ru": "Указано вручную", "en": "Set by hand", "ja": "手動指定"},
    "overview_place_session_inferred": {
        "ru": "Унаследовано от съёмки", "en": "Inherited from the session",
        "ja": "撮影セッションから継承",
    },
    "overview_place_trip_inferred": {
        "ru": "Унаследовано от поездки", "en": "Inherited from the trip",
        "ja": "旅行から継承",
    },
    "overview_place_path_inferred": {
        "ru": "Унаследовано от имени папки", "en": "Inherited from the folder name",
        "ja": "フォルダ名から継承",
    },
    "overview_place_visual": {
        "ru": "Определено по кадру", "en": "Recognised from the frame", "ja": "画像から判定",
    },
    "overview_no_place": {
        "ru": "Без места вообще", "en": "No place at all", "ja": "場所が全く不明",
    },
    "overview_no_place_hint": {
        "ru": "Эти кадры уедут в «_Без места».",
        "en": "These frames end up in the “no place” folder.",
        "ja": "これらは「場所なし」フォルダーに入ります。",
    },
    "overview_classified": {
        "ru": "Разобрано кадров", "en": "Frames classified", "ja": "分類済みフレーム",
    },
    "overview_verdict_photo": {
        "ru": "Личные фото", "en": "Personal photos", "ja": "個人写真",
    },
    "overview_by_source": {"ru": "Чем решено", "en": "Decided by", "ja": "判定の根拠"},
    "overview_by_tier": {"ru": "Каким ярусом", "en": "Tier that handled it",
                         "ja": "処理したティア"},
    "overview_source_heuristic": {"ru": "Эвристика", "en": "Heuristics", "ja": "ヒューリスティック"},
    "overview_source_clip": {"ru": "CLIP", "en": "CLIP", "ja": "CLIP"},
    "overview_source_ocr": {"ru": "OCR", "en": "OCR", "ja": "OCR"},
    "overview_source_vlm": {"ru": "VLM", "en": "VLM", "ja": "VLM"},
    "overview_tier_heuristic": {"ru": "Быстрый (эвристика)", "en": "Fast (heuristics)",
                                "ja": "高速（ヒューリスティック）"},
    "overview_tier_clip": {"ru": "Быстрый (CLIP)", "en": "Fast (CLIP)", "ja": "高速（CLIP）"},
    "overview_tier_vlm": {"ru": "Глубокий (VLM)", "en": "Deep (VLM)", "ja": "詳細（VLM）"},
    "overview_tier_none": {"ru": "Ярус не записан", "en": "Tier not recorded",
                           "ja": "ティア未記録"},
    "overview_vlm_ran": {
        "ru": "Глубокий ярус (VLM) прогонялся.",
        "en": "The deep tier (VLM) has run.",
        "ja": "詳細ティア（VLM）は実行済みです。",
    },
    "overview_vlm_not_ran": {
        "ru": "Глубокий ярус (VLM) не прогонялся.",
        "en": "The deep tier (VLM) has not run.",
        "ja": "詳細ティア（VLM）は未実行です。",
    },
    "overview_updated_at": {
        "ru": "Последнее изменение разбора: {at}",
        "en": "Classification last changed: {at}",
        "ja": "分類の最終更新: {at}",
    },
    "overview_not_classified": {
        "ru": "Разбор ещё не запускался.", "en": "The classifier has not run yet.",
        "ja": "分類はまだ実行されていません。",
    },
    "overview_layout_none": {
        "ru": "Раскладка ещё не запускалась — файлы лежат там же, где лежали.",
        "en": "No layout has run yet — the files are still where they were.",
        "ja": "まだ振り分けは実行されていません。ファイルは元の場所のままです。",
    },
    "overview_layout_batches": {"ru": "Раскладок было", "en": "Layout runs",
                                "ja": "振り分けの回数"},
    "overview_layout_started": {"ru": "Начата", "en": "Started", "ja": "開始"},
    "overview_layout_finished": {"ru": "Завершена", "en": "Finished", "ja": "完了"},
    "overview_layout_dest": {"ru": "Куда", "en": "Destination", "ja": "振り分け先"},
    "overview_layout_mode": {"ru": "Режим", "en": "Mode", "ja": "モード"},
    "overview_layout_files": {"ru": "Файлов в раскладке", "en": "Files in the batch",
                              "ja": "バッチ内のファイル"},
    "overview_layout_done": {"ru": "Из них перенесено", "en": "Of them moved",
                             "ja": "うち移動済み"},
    "overview_layout_unfinished": {
        "ru": "Батч не закрыт — прогон был прерван.",
        "en": "The batch is not closed — the run was interrupted.",
        "ja": "バッチが閉じられていません。実行が中断されました。",
    },
    "overview_op_move": {"ru": "перенос", "en": "move", "ja": "移動"},
    "overview_op_copy": {"ru": "копия", "en": "copy", "ja": "コピー"},
    "overview_goto_hint": {
        "ru": "Открыть вкладку «{tab}»", "en": "Open the {tab} tab", "ja": "「{tab}」タブを開く",
    },
    "error_loading_overview": {
        "ru": "Не удалось загрузить обзор: ", "en": "Could not load the overview: ",
        "ja": "概要を読み込めません: ",
    },
    # --- F133: the "Slices" tab, the layout warning and the settings drawer -----------
    "slices_intro": {
        "ru": "Срез — это подборка поверх канона: кадры с людьми, групповые, портреты, "
              "имена, события, животные, товары, скриншоты, документы. Альбом среза — "
              "жёсткие ссылки, их можно собрать и удалить сколько угодно раз.",
        "en": "A slice is a selection on top of the canon: frames with people, group "
              "photos, portraits, names, events, animals, products, screenshots, "
              "documents. An album of a slice is hardlinks — gather it and drop it as "
              "often as you like.",
        "ja": "スライスは正本の上に重ねる抽出です（人物あり・集合写真・ポートレート・"
              "名前・イベント・動物・商品・スクリーンショット・書類）。スライスの"
              "アルバムはハードリンクなので、何度でも作成・削除できます。",
    },
    # --- F134: the search line itself. The place F133 reserved is wired now, so the
    # placeholder names what actually goes in it — words, not the name of a slice.
    "search_placeholder": {
        "ru": "Найти словами: торт, снег, море…",
        "en": "Search by words: cake, snow, the sea…",
        "ja": "言葉で検索: ケーキ、雪、海…",
    },
    "search_button": {"ru": "Найти", "en": "Search", "ja": "検索"},
    # Shown until the first answer about the index arrives. Not "search is unavailable":
    # the state is not known yet, and guessing it in either direction is a lie that lasts
    # exactly as long as the request.
    "search_state_checking": {
        "ru": "Проверяем индекс поиска…", "en": "Checking the search index…",
        "ja": "検索インデックスを確認しています…",
    },
    # THE state of this feature: nothing was ever encoded. An empty result list would read
    # as "you have no photographs like that", which is a conclusion about somebody's own
    # archive drawn from a table that was never filled.
    # F141 corrected this sentence: the index is no longer a by-product of an ordinary
    # run. It is a second CLIP pass with a multilingual model, ~10.5 minutes per 20 000
    # frames, behind `features.search_index` — so the setting has to be named, or the
    # reader follows an instruction that will not fill the table.
    "search_state_empty": {
        "ru": "Искать пока не по чему: индекс поиска пуст. Включите "
              "features.search_index: true и запустите обработку коллекции — это "
              "отдельный проход CLIP многоязычной моделью (~10,5 минут на 20 000 кадров).",
        "en": "There is nothing to search yet: the search index is empty. Switch on "
              "features.search_index: true and process the collection — it is a separate "
              "CLIP pass with a multilingual model (~10.5 minutes per 20 000 frames).",
        "ja": "検索できる対象がまだありません。検索インデックスが空です。"
              "features.search_index: true を有効にしてコレクションを処理してください — "
              "多言語モデルによる別途の CLIP パスです（2 万コマあたり約 10.5 分）。",
    },
    # The other unavailable state, and deliberately a different sentence: the fix is the
    # same run, but the reason is that the stored vectors belong to another model and are
    # not comparable with this query. Mixing them silently would produce a plausible
    # ranking that nothing on screen marks as wrong.
    "search_state_other_model": {
        "ru": "Индекс поиска посчитан другой моделью ({model}): её векторы несравнимы с "
              "текущей, поэтому выдача была бы правдоподобной чушью. Нужен повторный "
              "прогон коллекции.",
        "en": "The search index was computed by another model ({model}): its vectors are "
              "not comparable with the current one, so the ranking would be plausible "
              "nonsense. The collection has to be processed again.",
        "ja": "検索インデックスは別のモデル（{model}）で作成されています。ベクトルに"
              "互換性がなく、もっともらしい誤った結果になります。コレクションを再度"
              "処理してください。",
    },
    # Available, and honest about the denominator: an incremental run is the normal way to
    # live with a growing archive, and a person must be able to tell "it is not in the
    # collection" from "it is not in the index yet".
    "search_state_partial": {
        "ru": "Ищем по {n} из {all} фотографий: остальные попадут в индекс на следующем "
              "прогоне.",
        "en": "Searching {n} of {all} photographs: the rest join the index on the next "
              "run.",
        "ja": "{all} 枚中 {n} 枚を検索対象にしています。残りは次回の処理でインデックスに"
              "追加されます。",
    },
    "search_state_ready": {
        "ru": "Ищем по всем {all} фотографиям коллекции.",
        "en": "Searching all {all} photographs of the collection.",
        "ja": "コレクションの {all} 枚すべてを検索します。",
    },
    "search_goto_overview": {
        "ru": "К «Обзору»", "en": "Go to Overview", "ja": "「概要」へ",
    },
    # No threshold exists and none will (search.py): the score orders frames against each
    # other and says nothing in absolute terms. The line says so instead of promising an
    # accuracy nobody has measured.
    "search_ranking_hint": {
        "ru": "Это ранжирование, а не фильтр: список отсортирован по близости к запросу, "
              "порога «точно оно» нет. Смотрите сверху вниз и остановитесь, где кончится "
              "похожее.",
        "en": "This is a ranking, not a filter: the list is sorted by closeness to the "
              "query and there is no “this really is it” threshold. Read top-down and "
              "stop where the resemblance runs out.",
        "ja": "これはフィルタではなくランキングです。クエリとの近さで並んでおり、"
              "「確実に該当」というしきい値はありません。上から順に見て、似ていないと"
              "感じたところで止めてください。",
    },
    "search_score_label": {
        "ru": "близость {score}", "en": "closeness {score}", "ja": "近さ {score}",
    },
    # F173: the numerator AND the denominator. The old wording ("{n} frames") was true of
    # the page and read as a fact about the collection — «200 кадров» for a query whose
    # ranking is four thousand long, with the half that matters below the fold.
    "search_shown_label": {
        "ru": "Запрос «{q}»: показано {shown} из {total}, от самого близкого",
        "en": "Query “{q}”: showing {shown} of {total}, closest first",
        "ja": "クエリ「{q}」: {total} 件中 {shown} 件を表示（近い順）",
    },
    # An available index always ranks everything it holds, so an empty list means the
    # index itself is empty of frames a search may return — never "there are no such
    # photographs".
    "search_no_frames": {
        "ru": "Ранжировать нечего: в индексе поиска нет ни одного кадра, который можно "
              "показать.",
        "en": "There is nothing to rank: the search index holds no frame that could be "
              "shown.",
        "ja": "並べ替える対象がありません。検索インデックスに表示できるコマがありません。",
    },
    "error_loading_search": {
        "ru": "Не удалось выполнить поиск: ", "en": "Could not run the search: ",
        "ja": "検索を実行できません: ",
    },
    # --- F189: the same line, answering with a person ----------------------------------
    # Said in front of the index's own reason rather than instead of it: the ranking still
    # cannot run and the way to fix that is still on screen — what changes is that the
    # field is not dead while there is somebody to find in it.
    "search_state_names_only": {
        "ru": "Имя названного человека здесь найдётся и без индекса — наберите имя.",
        "en": "The name of a person you have labelled is found here without the index — "
              "type a name.",
        "ja": "名前を付けた人物は、インデックスがなくてもここで見つかります — "
              "名前を入力してください。",
    },
    # The caption is the feature as much as the selection is. A reader who cannot tell an
    # exact answer from the top of a ranking has been handed one thing and shown another,
    # so this sentence says what it is and the ranking's sentence stays where it was.
    "search_person_shown_label": {
        "ru": "Кадры человека: {name} — показано {shown} из {total}",
        "en": "Frames of a person: {name} — showing {shown} of {total}",
        "ja": "人物のコマ: {name} — {total} 件中 {shown} 件を表示",
    },
    "search_person_hint": {
        "ru": "Это точный отбор по кластеру лиц, а не ранжирование: кадр либо в кластере "
              "этого человека, либо нет. Порога и «похожести» здесь нет, список полный — "
              "он лишь показывается по частям.",
        "en": "This is an exact selection by face cluster, not a ranking: a frame is "
              "either in this person's cluster or it is not. There is no threshold and no "
              "“closeness” here — the list is complete and merely shown in portions.",
        "ja": "これはランキングではなく、顔クラスタによる正確な抽出です。コマがこの人物の"
              "クラスタに入っているかどうかだけで決まります。しきい値も「近さ」もなく、"
              "一覧は完全で、分割して表示しているだけです。",
    },
    # The depth warning of a ranking does not apply to a list: the next page is more of the
    # same fact, not a worse guess.
    "search_person_more_hint": {
        "ru": "Дальше — продолжение того же списка: кадры не становятся менее «точными».",
        "en": "Further on is the same list continued: the frames do not get less certain.",
        "ja": "この先も同じ一覧の続きです。コマの確かさが下がることはありません。",
    },
    # Requirement 4 on screen: a name can be an ordinary word («Роза», «Марк»), and the
    # other answer is one click away instead of gone.
    "search_person_words_link": {
        "ru": "Искать «{q}» по картинке",
        "en": "Search for “{q}” as an image",
        "ja": "「{q}」を画像として検索",
    },
    "search_words_person_link": {
        "ru": "Показать кадры человека: {name}",
        "en": "Show the frames of a person: {name}",
        "ja": "人物のコマを表示: {name}",
    },
    # A named cluster all of whose frames are duplicates or unreadable. Rare, and still not
    # "nothing was found": the person exists, the frames a search may show do not.
    "search_person_no_frames": {
        "ru": "У этого человека нет кадров, которые можно показать: все они дубли или "
              "нечитаемые файлы.",
        "en": "This person has no frame that can be shown: all of them are duplicates or "
              "unreadable files.",
        "ja": "この人物には表示できるコマがありません。すべて重複か読み取れない"
              "ファイルです。",
    },
    # --- F152: the three face slices ---------------------------------------------------
    # The labels are deliberately not the label of the cluster slice next to them: "Люди"
    # there answers "who is this", these answer "is anybody in the frame".
    "face_slice_people": {"ru": "С людьми", "en": "With people", "ja": "人物あり"},
    "face_slice_group": {"ru": "Групповые", "en": "Group photos", "ja": "集合写真"},
    "face_slice_portrait": {"ru": "Портреты", "en": "Portraits", "ja": "ポートレート"},
    # THE line that has to differ from the caption of an approximate slice. A query slice
    # is a ranking and says so; this one is a fact of the detector, and the sentence says
    # what the fact is and where its errors come from, without a percentage nobody
    # measured.
    "face_slices_intro": {
        "ru": "Эти срезы — не оценка: кадр в них потому, что детектор нашёл на нём лицо. "
              "Порога «похоже на человека» здесь нет, ошибки бывают только у самого "
              "детектора. Служебная отметка «файл обработан, лиц нет» исключена везде.",
        "en": "These slices are not an estimate: a frame is here because the detector "
              "found a face on it. There is no “looks like a person” threshold — the only "
              "errors are the detector's own. The “processed, no faces” marker row is "
              "excluded everywhere.",
        "ja": "これらのスライスは推定ではありません。検出器がその写真で顔を見つけたから"
              "入っています。「人物らしさ」のしきい値はなく、誤りは検出器そのものの誤り"
              "だけです。「処理済み・顔なし」の内部記録はすべて除外されます。",
    },
    "face_hint_people": {
        "ru": "Хотя бы одно лицо в кадре.",
        "en": "At least one face in the frame.",
        "ja": "写真に顔が 1 つ以上あります。",
    },
    "face_hint_group": {
        "ru": "Лиц в кадре — {n} и больше (features.group_photo_faces).",
        "en": "{n} faces or more in the frame (features.group_photo_faces).",
        "ja": "顔が {n} 個以上（features.group_photo_faces）。",
    },
    "face_hint_portrait": {
        "ru": "Ровно одно лицо, и оно занимает не меньше {share}% кадра "
              "(features.portrait_face_share).",
        "en": "Exactly one face, covering at least {share}% of the frame "
              "(features.portrait_face_share).",
        "ja": "顔がちょうど 1 つで、写真の {share}% 以上を占めます"
              "（features.portrait_face_share）。",
    },
    # F125's rule: the reason, never a zero. Without a faces run nothing was measured, and
    # "0 photographs with people" is a statement about somebody's archive that no table
    # in this index supports.
    "face_no_faces_run": {
        "ru": "Стадия «лица» не запускалась — считать нечего. Запустите обработку с "
              "галочкой «Разбор по лицам», и срезы наполнятся сами.",
        "en": "The faces stage has not run — there is nothing to count yet. Process the "
              "collection with “Detect faces” ticked and these slices fill in by "
              "themselves.",
        "ja": "顔の処理がまだ実行されていないため、集計できません。「顔の検出」を"
              "有効にして処理すると、これらのスライスが表示されます。",
    },
    "face_empty": {
        "ru": "В этом срезе пусто: таких кадров не нашлось.",
        "en": "This slice is empty — no such frames were found.",
        "ja": "このスライスは空です。該当するコマは見つかりませんでした。",
    },
    "face_count_label": {
        "ru": "лиц: {n}", "en": "{n} faces", "ja": "顔 {n}",
    },
    # F173: the shared pager's button and counter here too. What this slice does NOT take
    # from it is `slice_depth_hint` — nothing is ranked here (a frame is in the slice
    # because the detector found a face), so there is no precision to trade for depth and
    # a line saying otherwise would be a warning about a risk this list does not carry.
    "error_loading_face_slices": {
        "ru": "Не удалось загрузить срезы по лицам: ",
        "en": "Could not load the face slices: ",
        "ja": "顔のスライスを読み込めません: ",
    },
    # --- F151: the pinned queries ------------------------------------------------------
    # The labels of the three slices `features.saved_slices` ships with. A name that is not
    # in this catalog is shown as it stands in the config — the row must not refuse to draw
    # a slice somebody added, and a made-up translation would be worse than the key itself.
    "query_slice_children": {"ru": "Дети", "en": "Children", "ja": "子ども"},
    "query_slice_products": {"ru": "Товары", "en": "Products", "ja": "商品"},
    "query_slice_animals": {"ru": "Животные", "en": "Animals", "ja": "動物"},
    # THE caption rule of this feature. «Животные» the pin and «Животные» the pet label are
    # two different slices of one archive — 60% precision against 71%, a ranking against a
    # verdict a model checked — and with the same label a reader would take the estimate for
    # the fact. So every pinned query wears the mark, including the ones with no exact
    # counterpart: what is marked is the METHOD, not the collision.
    "query_slice_pin": {
        "ru": "{name} · по запросу", "en": "{name} · by query", "ja": "{name}・クエリ",
    },
    "query_slice_intro": {
        "ru": "Это оценка, а не метка: срез собран запросом к тем же векторам, и ни одна "
              "модель его не проверяла. Порога «точно оно» здесь нет — список идёт от "
              "самого близкого, и где он перестаёт быть про запрос, решаете вы. На "
              "размеченной выборке из 200 кадров такой срез находит около 60% нужного в "
              "первой порции и около 90% в удвоенной, поэтому «Показать ещё» здесь — "
              "главная кнопка, а не украшение.",
        "en": "This is an estimate, not a label: the slice is a query over the same "
              "vectors and no model has checked it. There is no “this really is it” "
              "threshold — the list runs from the closest down, and where it stops being "
              "about the query is yours to decide. On a hand-labelled sample of 200 "
              "frames a slice like this finds about 60% of what you are after in the "
              "first portion and about 90% in a doubled one, which is why “Show more” is "
              "the main button here rather than a decoration.",
        "ja": "これはラベルではなく推定です。同じベクトルへの問い合わせで集めた"
              "スライスであり、モデルによる確認は行われていません。「確実に該当」と"
              "いうしきい値はなく、近い順に並ぶだけなので、どこで終わりにするかは"
              "あなたが決めます。200 コマの人手ラベル付き標本では、最初の一覧で約 "
              "60%、倍の深さで約 90% を拾えます。だからこそ「さらに表示」が主役の"
              "ボタンです。",
    },
    # What the slice actually asked, on screen — the half that makes "editable without
    # code" real rather than stated. The phrases stay English whatever `language:` says:
    # they go to a CLIP text tower and not to a reader, and the measured numbers were
    # produced by this wording.
    "query_slice_phrases": {
        "ru": "Запрос среза: {phrases}. Правится в features.saved_slices; формулировки "
              "английские — язык интерфейса на выдачу не влияет.",
        "en": "The slice asks: {phrases}. Edit it in features.saved_slices; the phrases "
              "are English — the interface language does not change this list.",
        "ja": "このスライスの問い合わせ: {phrases}。features.saved_slices で編集でき"
              "ます。表現は英語です（表示言語はこの一覧に影響しません）。",
    },
    "query_slice_shown_label": {
        "ru": "Срез «{name}»: показано {shown} из {total}, от самого близкого",
        "en": "Slice “{name}”: showing {shown} of {total}, closest first",
        "ja": "スライス「{name}」: {total} 件中 {shown} 件を表示（近い順）",
    },
    "error_loading_saved_slices": {
        "ru": "Не удалось загрузить срез по запросу: ",
        "en": "Could not load the query slice: ",
        "ja": "クエリのスライスを読み込めません: ",
    },
    # --- F156: pinning a query of one's own --------------------------------------------
    # The product stops guessing which facets matter (the sample of 200 says there is no
    # such thing as "the" facets: ten candidate slices covered 26% of the unclassed frames
    # at best), so these strings are all about one act — a person saving THEIR query.
    "pin_slice_button": {
        "ru": "Закрепить как срез", "en": "Pin as a slice", "ja": "スライスとして固定",
    },
    # The name is asked for, with the query itself offered: the query is usually the best
    # name there is, and a dialog that demands a different one is a dialog that gets «мое1».
    "pin_slice_prompt": {
        "ru": "Название среза (запрос: {query})",
        "en": "Name of the slice (the query: {query})",
        "ja": "スライスの名前（クエリ: {query}）",
    },
    # THE warning of this feature, and it is said BEFORE the pin rather than afterwards.
    # The phrases go to the model as they stand and the search index is English until F141
    # reaches this collection, so a Russian or Japanese pin will rank badly — a person who
    # learns that a week later concludes the feature is broken.
    "pin_slice_language_warning": {
        "ru": "Запрос не на английском. Индекс пока английский, поэтому такой срез будет "
              "работать заметно хуже — формулировка уходит в модель как есть.",
        "en": "The query is not in English. The index is English for now, so a slice like "
              "this will work noticeably worse — the wording goes to the model as it is.",
        "ja": "クエリが英語ではありません。索引は現時点で英語なので、このスライスの精度は"
              "目に見えて落ちます（表現はそのままモデルに渡されます）。",
    },
    "pin_slice_done": {
        "ru": "Срез «{name}» закреплён.", "en": "The slice “{name}” is pinned.",
        "ja": "スライス「{name}」を固定しました。",
    },
    # Every refusal is a sentence, never a button that does nothing.
    "pin_error_empty": {
        "ru": "Пустой запрос закрепить нельзя.",
        "en": "An empty query cannot be pinned.",
        "ja": "空のクエリは固定できません。",
    },
    "pin_error_duplicate": {
        "ru": "Срез с таким названием уже закреплён.",
        "en": "A slice with that name is already pinned.",
        "ja": "その名前のスライスはすでに固定されています。",
    },
    # F133's reason and not a resource one — and the number is in the sentence, so the
    # person knows what to unpin and what to raise.
    "pin_error_limit": {
        "ru": "Закреплено {max} срезов — это предел. Открепите ненужный или поднимите "
              "features.max_pinned_slices.",
        "en": "{max} slices are pinned — that is the limit. Unpin one, or raise "
              "features.max_pinned_slices.",
        "ja": "固定できるスライスは {max} 件までです。不要なものを外すか、"
              "features.max_pinned_slices を増やしてください。",
    },
    "pin_error_generic": {
        "ru": "Не удалось закрепить срез: ", "en": "Could not pin the slice: ",
        "ja": "スライスを固定できません: ",
    },
    "pin_unpin_button": {
        "ru": "Открепить срез", "en": "Unpin the slice", "ja": "スライスを外す",
    },
    # The confirmation says what is removed AND what is not: "delete the slice" and
    # "delete the photographs" are one word apart, and only one of them is happening.
    "pin_unpin_confirm": {
        "ru": "Открепить срез «{name}»? Удалится только закрепление — файлы останутся "
              "на месте.",
        "en": "Unpin the slice “{name}”? Only the pin is removed — the files stay where "
              "they are.",
        "ja": "スライス「{name}」を外しますか？外れるのは固定だけで、ファイルはそのまま"
              "残ります。",
    },
    "pin_move_up": {"ru": "Выше", "en": "Move up", "ja": "上へ"},
    "pin_move_down": {"ru": "Ниже", "en": "Move down", "ja": "下へ"},
    # The album gathers what a single query ranks (`sorta album query`), and a slice asking
    # several phrases is ranked by their average — one selector cannot reproduce it, so the
    # button is not offered rather than gathering a different list under the same name.
    "pin_album_one_query": {
        "ru": "Альбом собирается по одной формулировке, а этот срез спрашивает несколько. "
              "Оставьте в features.saved_slices одну — и кнопка появится.",
        "en": "An album is gathered by a single wording, and this slice asks several. "
              "Leave one of them in features.saved_slices and the button appears.",
        "ja": "アルバムは 1 つの表現でまとめます。このスライスは複数を問い合わせている"
              "ため、features.saved_slices に 1 つだけ残すとボタンが表示されます。",
    },
    # --- F156: why a built-in slice is empty -------------------------------------------
    # The `frame_quality` rule of F125, said out loud on a whole slice: a zero with no
    # explanation reads as "there are none of these in your archive", and far more often
    # the truth is that nobody has looked yet. The counterpart answer — "it was computed
    # and there is nothing" — is the slice's own existing line ("События не найдены."),
    # which is why only this half needed writing.
    "slice_not_computed": {
        "ru": "Это не считалось: стадия, которая наполняет срез, не запускалась. "
              "Пусто здесь означает «не спрашивали», а не «в архиве нет».",
        "en": "This was not computed: the stage that fills this slice has not run. Empty "
              "here means “nobody asked”, not “there are none”.",
        "ja": "これは計算されていません。このスライスを埋める処理が実行されていません。"
              "ここでの空は「該当なし」ではなく「未確認」という意味です。",
    },
    "slice_goto_process": {
        "ru": "К экрану прогона", "en": "Go to the run screen", "ja": "実行画面へ",
    },
    "slices_pinned_label": {
        "ru": "Закреплённые срезы", "en": "Pinned slices", "ja": "固定スライス",
    },
    "slices_empty": {
        "ru": "Срезов пока нет: обработайте коллекцию — люди, события, животные и "
              "классы появятся здесь.",
        "en": "No slices yet: process the collection — people, events, animals and "
              "classes show up here.",
        "ja": "スライスはまだありません。コレクションを処理すると、人物・イベント・"
              "動物・分類がここに表示されます。",
    },
    "layout_review_warning": {
        "ru": "В «Разборе» осталось без решения: {n}. Отмеченные к удалению уезжают в "
              "«_delete» во время раскладки, а альбомы — ссылки из канона: собрав их "
              "раньше, вы получите ссылки на выброшенное. Раскладку это не запрещает.",
        "en": "The Review still holds {n} undecided. Frames marked for deletion leave "
              "for “_delete” during the layout, and albums are links out of the canon: "
              "gather them earlier and you get links to what you threw away. This does "
              "not block the layout.",
        "ja": "「仕分け」に未決定が {n} 件残っています。削除指定のコマは振り分けの際に"
              "「_delete」へ移動し、アルバムは正本からのリンクです。先にアルバムを作ると"
              "捨てたものへのリンクが残ります。振り分け自体は禁止されません。",
    },
    "layout_review_goto": {
        "ru": "К «Разбору»", "en": "Go to Review", "ja": "「仕分け」へ",
    },
    "settings_open_button": {
        "ru": "Настройки", "en": "Settings", "ja": "設定",
    },
    "settings_close_button": {
        "ru": "Закрыть", "en": "Close", "ja": "閉じる",
    },
}
