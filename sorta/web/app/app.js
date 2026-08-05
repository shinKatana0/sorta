(function () {
  var I18N = window.I18N;
  var THEME_KEY = "sorta-ui-theme";
  // F80: сколько кадров ленты может листать лайтбокс (SORTA_VIDEO_FRAMES). У
  // короткого ролика кадров реально меньше — это выясняется по первому 404.
  var VIDEO_FRAMES = window.VIDEO_FRAMES || 1;

  // --- инлайн-SVG иконки (U1: без иконочных шрифтов/эмодзи) --------------
  var ICONS = {
    folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 ' +
        '2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/><path d="M12 12v4M10 14h4"/></svg>',
    tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4 12 22l-9-9V4a1 1 ' +
        '0 0 1 1-1h9l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.2"/></svg>',
    merge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="4.5" r="1.6"/>' +
        '<circle cx="18" cy="4.5" r="1.6"/><circle cx="12" cy="19.5" r="1.6"/>' +
        '<path d="M6 6v3c0 2.5 2 4 4 4h1M18 6v3c0 2.5-2 4-4 4h-1M12 13v5"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 ' +
        '2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13"/><path d="M10 11v6M14 11v6"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7"/></svg>',
    spinner: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
        'stroke-linecap="round"><circle cx="12" cy="12" r="9" opacity="0.25"/>' +
        '<path d="M21 12a9 9 0 0 0-9-9"/></svg>',
    warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 22 20H2L12 3z"/>' +
        '<path d="M12 10v4M12 17h.01"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>' +
        '<path d="M12 8h.01M11 11.5h1v5.5h1"/></svg>',
    film: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" ' +
        'height="14" rx="2"/><path d="M7 5v14M17 5v14M3 12h18"/></svg>',
    pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s7-6.3 7-11a7 7 0 1 ' +
        '0-14 0c0 4.7 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/></svg>',
  };

  function icon(name) {
    var tmp = document.createElement("div");
    tmp.innerHTML = ICONS[name] || "";
    var el = tmp.firstElementChild;
    if (el) el.setAttribute("aria-hidden", "true");
    return el;
  }

  // Кнопка с опциональной иконкой: variant — "primary"/"ghost"/"danger"/null.
  function makeBtn(variant, iconName, label, extraClass) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn" + (variant ? " btn-" + variant : "") + (extraClass ? " " + extraClass : "");
    if (iconName) btn.appendChild(icon(iconName));
    btn.appendChild(document.createTextNode(label));
    return btn;
  }

  // Единый спокойный вид для пустых/загрузочных/ошибочных состояний вкладок.
  function stateEl(kind, text) {
    var div = document.createElement("div");
    div.className = "state-msg state-" + kind;
    var iconName = kind === "error" ? "warn" : kind === "loading" ? "spinner" : "info";
    var ic = icon(iconName);
    if (ic) div.appendChild(ic);
    div.appendChild(document.createTextNode(text));
    return div;
  }

  function wrapTable(table) {
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.appendChild(table);
    return wrap;
  }

  function fmt(template, vals) {
    return template.replace(/\{(\w+)\}/g, function (_, key) {
      return Object.prototype.hasOwnProperty.call(vals, key) ? vals[key] : "";
    });
  }

  // --- F173: "show more", once, for every ordered slice ----------------------
  // The measurements of 2026-08-02/03 left exactly one confirmed lever of completeness —
  // the DEPTH of the list. Doubling it adds ~25 points on average, and the query «дети»
  // goes from 61% to 89%: the second half of a ranking holds nearly a third of what the
  // reader is looking for. Four slices had a button for that and search, the one slice
  // built by a query rather than by a model's marks, did not — it stopped dead at
  // `features.search_limit` frames with a caption that read like an answer.
  //
  // So this is the organ itself, written once. A slice hands over its grid, the URL of a
  // page and the way to draw a card; it gets appending pages, a button that hides itself
  // at the end of the list, a counter that says how many there are IN TOTAL and — where
  // something is actually ranked — the one line about what depth costs. A slice added
  // tomorrow (a saved query, a low-resolution list) gets all of that by calling this, and
  // that is the point: the fifth copy of the same twenty lines is how a new list ships
  // without the button again.
  //
  // Deliberately NOT an infinite scroll. A page arrives when a person asks for one,
  // because depth is a trade against precision and the person making it has to be making
  // it on purpose.
  function makePager(opts) {
    var total = 0;
    var offset = 0;
    var hasMore = false;

    function grid() { return document.getElementById(opts.grid); }

    // The number on screen is counted off the DOM rather than accumulated, so a card that
    // leaves the grid (a mark that takes a frame out of the list) cannot desynchronize the
    // counter from what the reader can see — and the next page starts where the list ends.
    function shown() { return grid().querySelectorAll(opts.cardSelector).length; }

    function paint(data) {
      var n = shown();
      offset = n;
      var counter = document.getElementById(opts.shown);
      if (counter) {
        counter.textContent = n
            ? (opts.shownText ? opts.shownText(n, total, data)
                              : fmt(I18N.slice_shown_label, { shown: n, total: total }))
            : "";
      }
      var visible = hasMore && n > 0;
      var btn = document.getElementById(opts.moreBtn);
      if (btn) btn.style.display = visible ? "" : "none";
      // The warning belongs to the button: with nothing left to load there is no trade to
      // warn about, and a permanent line about precision is a line nobody reads.
      var hint = opts.hint ? document.getElementById(opts.hint) : null;
      if (hint) hint.style.display = visible ? "" : "none";
    }

    function renderPage(data, append) {
      var box = grid();
      if (!append) box.textContent = "";
      (data.items || []).forEach(function (it) { box.appendChild(opts.card(it)); });
      total = Number(data.total) || 0;
      // `has_more` is the server's, computed from the window it actually served: a client
      // guessing from its own running count is wrong the first time a page comes back
      // short, and the wrong direction of that mistake is a button that promises a page
      // which does not exist.
      hasMore = !!data.has_more;
      // `emptyEl` for a slice whose empty state carries an action (F156: "the stage has
      // not run" comes with a button to the run screen); `emptyText` for the rest.
      if (!shown()) {
        box.appendChild(opts.emptyEl ? opts.emptyEl(data)
                                     : stateEl("empty", opts.emptyText(data)));
      }
      paint(data);
      // Whatever else this slice prints beside the shared counter — the animal tab's
      // second number, for one. It runs after the page is in the DOM, because the number
      // it is about is usually a number of cards.
      if (opts.after) opts.after(data, shown());
    }

    function fetchPage(from, append) {
      var box = grid();
      if (!append) {
        box.textContent = "";
        box.appendChild(stateEl("loading", I18N.loading));
      }
      return fetch(opts.url(from, opts.pageSize))
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (opts.onData) opts.onData(data, append);
          renderPage(data, append);
          return data;
        })
        .catch(function (err) {
          box.textContent = "";
          box.appendChild(stateEl("error", opts.errorText() + err));
        });
    }

    var moreBtn = document.getElementById(opts.moreBtn);
    if (moreBtn) {
      moreBtn.addEventListener("click", function () { fetchPage(offset, true); });
    }

    return {
      load: function () { return fetchPage(0, false); },
      more: function () { return fetchPage(offset, true); },
      // The grid changed under the pager (a mark redrew or removed a card): recount and
      // restate, without asking the server for a page it already sent.
      sync: function (newTotal) {
        if (newTotal !== null && newTotal !== undefined) total = Number(newTotal) || 0;
        hasMore = shown() < total;
        paint(null);
        return shown();
      },
      shown: shown,
    };
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    document.getElementById("theme-toggle-label").textContent =
        theme === "dark" ? I18N.theme_light : I18N.theme_dark;
  }

  function initTheme() {
    var saved = null;
    try { saved = window.localStorage.getItem(THEME_KEY); } catch (e) { saved = null; }
    var theme = saved || ((window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light");
    applyTheme(theme);
  }

  document.getElementById("theme-toggle-btn").addEventListener("click", function () {
    var current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    var next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
  });

  initTheme();

  var LANG_KEY = "sorta_lang";
  var SUPPORTED_LANGS = ["ru", "en", "ja"];

  function urlWithLang(lang) {
    var url = new URL(window.location.href);
    url.searchParams.set("lang", lang);
    return url.toString();
  }

  function initLang() {
    var select = document.getElementById("lang-select");
    var currentLang = document.documentElement.lang;
    var saved = null;
    try { saved = window.localStorage.getItem(LANG_KEY); } catch (e) { saved = null; }
    if (saved && SUPPORTED_LANGS.indexOf(saved) !== -1 && saved !== currentLang) {
      window.location.replace(urlWithLang(saved));
      return;
    }
    if (select) {
      select.addEventListener("change", function () {
        var next = select.value;
        try { window.localStorage.setItem(LANG_KEY, next); } catch (e) { /* ignore */ }
        window.location.href = urlWithLang(next);
      });
    }
  }

  initLang();

  // F65: the "Folder language" selector (Cities tab) — the OUTPUT language of
  // folders/names, separate from the interface language. Reads the current value
  // from /api/config, and on change persists it (POST /api/config/language) and
  // re-renders the city plan preview with the new folder names.
  function initFolderLang() {
    var select = document.getElementById("folder-lang-select");
    if (!select) return;
    fetch("/api/config")
      .then(function (r) { return r.json(); })
      .then(function (cfg) { if (cfg && cfg.language) select.value = cfg.language; })
      .catch(function () { /* keep the default option */ });
    select.addEventListener("change", function () {
      var next = select.value;
      select.disabled = true;
      postJson("/api/config/language", { language: next }).then(function (resp) {
        select.disabled = false;
        if (resp && resp.ok) {
          refreshPlan();
          settingsStatus(I18N.folder_lang_saved);
        } else {
          settingsStatus((resp && resp.error === "already running")
              ? I18N.settings_busy
              : I18N.settings_error_prefix + ((resp && resp.error) || "error"));
        }
      }).catch(function () { select.disabled = false; });
    });
  }

  initFolderLang();

  // F104: the settings column of the "Cities" tab. Every control writes ONE key
  // through POST /api/settings; the server puts it into the RUNNING config and into
  // config.yaml, so none of this needs `sorta ui` restarted. The whole reason these
  // knobs got an interface is that a text editor plus a restart is not a switch.
  //
  // A rejected save (a run is in progress -> 409, garbage -> 400) is not swallowed:
  // the control is put back to the value the SERVER holds, so the form can never show
  // a setting the tool is not actually using.
  //
  // F138: the knobs that cost a run TIME are not in this list any more — they are on
  // the run screen with their price beside them, and each has exactly one place.
  var SETTING_CONTROLS = [
    { key: "vlm.model", id: "setting-vlm-model", kind: "text" },
    { key: "vlm.workers", id: "setting-vlm-workers", kind: "int" },
    { key: "vlm.max_edge", id: "setting-vlm-max-edge", kind: "int" },
    { key: "features.pet_threshold", id: "setting-features-pet-threshold", kind: "float" },
    { key: "features.sharpness_max_edge", id: "setting-features-sharpness-max-edge", kind: "int" },
    { key: "features.sharpness_band_min", id: "setting-features-sharpness-band-min", kind: "float" },
    { key: "features.sharpness_band_max", id: "setting-features-sharpness-band-max", kind: "float" },
    { key: "features.subject_score_min", id: "setting-features-subject-score-min", kind: "float" },
    { key: "imaging.preview_cache_max_gb", id: "setting-imaging-preview-cache-max-gb", kind: "int" }
  ];
  var settingsValues = {};

  function settingsStatus(text) {
    var el = document.getElementById("settings-status");
    if (el) el.textContent = text;
  }

  function renderSettings(data) {
    if (data) settingsValues = data;
    SETTING_CONTROLS.forEach(function (control) {
      var el = document.getElementById(control.id);
      if (!el || !(control.key in settingsValues)) return;
      if (control.kind === "bool") el.checked = !!settingsValues[control.key];
      else el.value = settingsValues[control.key];
    });
  }

  function readSetting(control) {
    var el = document.getElementById(control.id);
    if (!el) return null;
    if (control.kind === "bool") return el.checked;
    if (control.kind === "int") {
      var n = parseInt(el.value, 10);
      // An empty or non-numeric field is sent AS IS: the server owns the range and
      // answers 400, and one refusal in one place beats two copies of the rule.
      return isNaN(n) ? el.value : n;
    }
    if (control.kind === "float") {
      var f = parseFloat(el.value);
      return isNaN(f) ? el.value : f;
    }
    return el.value.trim();
  }

  function saveSetting(control) {
    var body = {};
    body[control.key] = readSetting(control);
    settingsStatus("");
    postJson("/api/settings", body).then(function (resp) {
      if (resp && resp.ok) {
        renderSettings(resp.settings);
        settingsStatus(I18N.settings_saved);
        return;
      }
      renderSettings(null);
      settingsStatus((resp && resp.error === "already running")
          ? I18N.settings_busy
          : I18N.settings_error_prefix + ((resp && resp.error) || "error"));
    }).catch(function () {
      renderSettings(null);
      settingsStatus(I18N.settings_error_prefix + "network");
    });
  }

  function initSettings() {
    fetch("/api/settings")
      .then(function (r) { return r.json(); })
      .then(function (data) { renderSettings(data); })
      .catch(function () { /* the column keeps its empty fields */ });
    SETTING_CONTROLS.forEach(function (control) {
      var el = document.getElementById(control.id);
      if (!el) return;
      el.addEventListener("change", function () { saveSetting(control); });
    });
  }

  initSettings();

  // Дерево по списку элементов — осталось для вкладки «Перемещения»: там приходит
  // ОДИН батч (ограниченный по размеру), а не весь план коллекции, поэтому строить
  // его из готового списка по-прежнему нормально. План города/людей/событий с F70
  // ходит другим путём — через агрегат ниже.
  function countFiles(node) {
    var n = node.files.length;
    Object.keys(node.children).forEach(function (k) { n += countFiles(node.children[k]); });
    return n;
  }

  function buildTree(items) {
    var root = { files: [], children: {} };
    items.forEach(function (item) {
      var parts = (item.target_rel || "").split("/");
      parts.pop();
      var node = root;
      parts.forEach(function (part) {
        if (!node.children[part]) node.children[part] = { files: [], children: {} };
        node = node.children[part];
      });
      node.files.push(item);
    });
    return root;
  }

  // Ленивое построение узла: содержимое папки создаётся ТОЛЬКО при первом
  // раскрытии <details> — строки со всеми <img> сразу подвешивали вкладку.
  function renderNode(name, node, depth, renderFilesFn) {
    var renderFn = renderFilesFn || renderFiles;
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    summary.textContent = name + " (" + countFiles(node) + ")";
    details.appendChild(summary);
    var built = false;
    details.addEventListener("toggle", function () {
      if (!details.open || built) return;
      built = true;
      if (node.files.length) details.appendChild(renderFn(node.files));
      Object.keys(node.children).sort().forEach(function (childName) {
        details.appendChild(renderNode(childName, node.children[childName], depth + 1, renderFn));
      });
    });
    return details;
  }

  // F70: дерево строится из АГРЕГАТА (папка -> количество), а не из списка файлов —
  // сервер больше не отдаёт 26 тысяч элементов одним куском. Каждый узел знает
  // суммарное количество файлов в своей ветке; лист (`category`) знает ключ, по
  // которому у сервера запрашивается страница файлов.
  function buildCategoryTree(categories) {
    var root = { count: 0, children: {}, category: null };
    categories.forEach(function (row) {
      var parts = String(row.category || "").split("/");
      var node = root;
      node.count += row.count;
      parts.forEach(function (part, i) {
        if (!node.children[part]) {
          node.children[part] = { count: 0, children: {}, category: null };
        }
        node = node.children[part];
        node.count += row.count;
        if (i === parts.length - 1) node.category = row.category;
      });
    });
    return root;
  }

  // --- удаление отдельного кадра (общий путь для обеих вкладок) --------

  function deletePhoto(fileId, onSuccess) {
    var remember = document.getElementById("delete-remember").checked;
    if (!remember && !window.confirm(I18N.confirm_delete_photo)) return;
    postJson("/api/photo/trash", { file_id: fileId }).then(function (resp) {
      if (resp.trashed && resp.trashed.length) onSuccess();
    });
  }

  // Массовое удаление выбранного (общий путь _trash_files, что и одиночный).
  // onSuccess получает список реально отправленных в корзину file_id.
  function deletePhotos(fileIds, onSuccess) {
    postJson("/api/photos/trash", { file_ids: fileIds }).then(function (resp) {
      if (resp.trashed) {
        onSuccess(resp.trashed.map(function (t) { return t.file_id; }));
      }
    });
  }

  // F145: the rule that used to hold for the layout button alone, stated for everything
  // that WRITES. The server refuses all of it with 409 while a run, a layout or an undo
  // is in flight (see BUSY_REFUSED_ROUTES), and a control that is alive for an action
  // that cannot happen teaches that the interface lies — you find that out by clicking.
  // So: dead while busy, with a line saying why (the `.busy-hint` spans), and alive again
  // the moment it ends, without reloading the page — hence `= busy` everywhere below and
  // never a one-way disable.
  //
  // Declared here, above the first control that uses it: the three flags themselves are
  // set further down (they belong to the polls that own them), and until a poll has run
  // nothing is running, which is what `undefined` means here anyway.
  function uiBusy() {
    return !!(sortRunning || processRunning || undoRunning);
  }

  // Some of these controls have a rule of their own ("nothing selected -> dead") and are
  // redrawn by their own tab. They register that redraw here instead of being listed by
  // id, so the two rules meet in one place and neither can undo the other.
  var busyRefreshers = [];

  function registerBusyRefresh(fn) {
    busyRefreshers.push(fn);
    fn();
  }

  // Переиспользуемый множественный выбор + «Удалить выбранное» для любого
  // контейнера со строками, где есть чекбокс `.row-select` (value=file_id).
  // Делегирование на контейнер — работает и с лениво построенными строками.
  // F104: barId — the row the button lives in; it is SHOWN only while something is
  // selected. A permanently visible "Delete selected" next to "Apply" is a destructive
  // button one row away from the button that moves the whole collection; in the context
  // of a selection it is the obvious action, and nowhere near the layout controls.
  function wireBulkDelete(containerId, buttonId, countId, barId) {
    var container = document.getElementById(containerId);
    var button = document.getElementById(buttonId);
    var countEl = countId ? document.getElementById(countId) : null;
    var barEl = barId ? document.getElementById(barId) : null;
    function checked() {
      return Array.prototype.slice.call(container.querySelectorAll(".row-select:checked"));
    }
    function refresh() {
      var n = checked().length;
      if (countEl) countEl.textContent = n ? " (" + n + ")" : "";
      button.disabled = uiBusy() || n === 0;   // F145: files go to the trash from here
      if (barEl) barEl.style.display = n === 0 ? "none" : "";
    }
    registerBusyRefresh(refresh);
    container.addEventListener("change", function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("row-select")) refresh();
    });
    button.addEventListener("click", function () {
      var boxes = checked();
      if (!boxes.length) return;
      var ids = boxes.map(function (b) { return parseInt(b.value, 10); });
      if (!window.confirm(fmt(I18N.confirm_delete_selected, { n: ids.length }))) return;
      deletePhotos(ids, function (trashedIds) {
        var done = {};
        trashedIds.forEach(function (id) { done[id] = true; });
        boxes.forEach(function (b) {
          if (done[parseInt(b.value, 10)]) {
            var tr = b.closest("tr");
            if (tr) tr.remove();
          }
        });
        refresh();
      });
    });
    refresh();
  }

  // Единое поведение превью по всему UI: клик по миниатюре (Города/Дубли/
  // Перемещения/События/Люди) открывает лайтбокс с крупным /preview, а не новую
  // вкладку с сырым /photo. samples/index позволяют листать соседние кадры (для
  // одиночных строк — [fileId]/0). thumbUrl опционален (по умолчанию /thumb/id).
  // F70: раскрытая папка — это до PLAN_PAGE_SIZE строк, то есть столько же
  // одновременных GET /thumb/<id>. Сервер ограничивает число параллельных декодов,
  // но очередь запросов браузера ничем не ограничена. Два простых ограничения:
  // (1) src ставится только когда картинка реально видна (IntersectionObserver);
  // (2) одновременно грузится не больше THUMB_CONCURRENCY штук — остальные ждут в
  // очереди. Слот освобождается по load/error, поэтому очередь не может застрять.
  var THUMB_CONCURRENCY = 6;
  var thumbQueue = [];
  var thumbActive = 0;

  function releaseThumbSlot() {
    thumbActive -= 1;
    pumpThumbQueue();
  }

  function pumpThumbQueue() {
    while (thumbActive < THUMB_CONCURRENCY && thumbQueue.length) {
      var next = thumbQueue.shift();
      thumbActive += 1;
      next.img.addEventListener("load", releaseThumbSlot);
      next.img.addEventListener("error", releaseThumbSlot);
      next.img.src = next.url;
    }
  }

  function queueThumb(img, url) {
    thumbQueue.push({ img: img, url: url });
    pumpThumbQueue();
  }

  var thumbObserver = null;
  if (window.IntersectionObserver) {
    thumbObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        thumbObserver.unobserve(entry.target);
        queueThumb(entry.target, entry.target.getAttribute("data-thumb-src"));
      });
    }, { rootMargin: "200px" });
  }

  function loadThumbWhenVisible(img, url) {
    if (!thumbObserver) { queueThumb(img, url); return; }
    img.setAttribute("data-thumb-src", url);
    thumbObserver.observe(img);
  }

  // F80: у видео плитка получает значок — до этого ролик в сетке был неотличим от
  // фото. Обёртка создаётся ТОЛЬКО для видео: у фото в ячейке остаётся тот же голый
  // <img>, что и раньше, поэтому вёрстка фото-строк не меняется вовсе.
  function videoBadge() {
    var badge = document.createElement("span");
    badge.className = "thumb-video-badge";
    var mark = icon("film");
    if (mark) badge.appendChild(mark);
    badge.appendChild(document.createTextNode(I18N.video_badge));
    return badge;
  }

  function clickableThumb(fileId, samples, index, thumbUrl, isVideo) {
    var img = document.createElement("img");
    loadThumbWhenVisible(img, thumbUrl || ("/thumb/" + fileId));
    img.alt = "";
    img.className = "clickable-thumb";
    img.title = isVideo ? I18N.video_open : I18N.lightbox_open;
    img.addEventListener("click", function () {
      openLightbox(samples || [fileId], index || 0, isVideo ? VIDEO_FRAMES : 0);
    });
    if (!isVideo) return img;
    var wrap = document.createElement("span");
    wrap.className = "thumb-video";
    wrap.appendChild(img);
    wrap.appendChild(videoBadge());
    return wrap;
  }

  // --- F77: ручные правки раскладки (не трогать / перенести в папку) -----
  // Правка только помечает файл в БД: физически ничего не двигается до общей
  // раскладки. Пометка приходит вместе со страницей плана (item.override), поэтому
  // после перерисовки список остаётся размеченным.

  var PLAN_ID_PAGE_SIZE = 1000;  // серверный максимум limit для страницы плана

  // Строка помечается ДВУМЯ разными способами: исключённая (красная рамка) и
  // перенесённая (синяя пунктирная) — это разные состояния, путать их нельзя.
  function markOverrideRow(tr, action, target) {
    tr.classList.remove("override-exclude", "override-reassign", "override-photo");
    var old = tr.querySelector(".override-mark");
    if (old) old.remove();
    tr.dataset.override = action || "";
    var btn = tr.querySelector(".override-row-btn");
    if (btn) {
      btn.textContent = action ? I18N.override_clear_button : I18N.override_exclude_button;
    }
    if (!action) {
      tr.removeAttribute("title");
      return;
    }
    // F103: a third state — "returned to the photos" (a correction made in the
    // «Служебные кадры» slice). The plan row has to show it apart from the other two:
    // this is neither "leave alone" nor "moved to a folder", it is the classifier's
    // verdict being taken off.
    var excluded = action === "exclude";
    var restored = action === "photo";
    tr.classList.add(excluded ? "override-exclude"
        : restored ? "override-photo" : "override-reassign");
    var label = excluded ? I18N.override_excluded_mark
        : restored ? I18N.junk_restored_mark
        : fmt(I18N.override_reassigned_mark, { target: target || "" });
    tr.title = label;
    var chip = document.createElement("span");
    chip.className = "chip override-mark " + (excluded ? "chip-danger"
        : restored ? "chip-good" : "chip-accent");
    chip.textContent = label;
    var meta = tr.querySelector(".plan-meta");
    if (meta) meta.appendChild(chip);
  }

  // Пометить уже отрисованные строки внутри scope (контейнер/узел дерева) —
  // «список обновляется без перезагрузки страницы».
  function markRowsOverride(scope, fileIds, action, target) {
    var wanted = {};
    fileIds.forEach(function (id) { wanted[id] = true; });
    Array.prototype.slice.call(scope.querySelectorAll(".row-select")).forEach(function (box) {
      if (!wanted[parseInt(box.value, 10)]) return;
      var tr = box.closest("tr");
      if (tr) markOverrideRow(tr, action, target);
    });
  }

  function overrideStatusEl() {
    return document.getElementById("override-status");
  }

  function applyOverride(action, fileIds, target, onSuccess) {
    var body = { file_ids: fileIds, action: action };
    if (target) body.target = target;
    return postJson("/api/overrides", body).then(function (resp) {
      if (resp && resp.ok) {
        onSuccess(resp.file_ids || fileIds);
      } else {
        overrideStatusEl().textContent = I18N.override_error_prefix +
            ((resp && resp.error) || "");
      }
    }).catch(function (err) {
      overrideStatusEl().textContent = I18N.override_error_prefix + err;
    });
  }

  // Все file_id папки — страницами у сервера, поэтому «не трогать папку» работает
  // и для нераскрытой папки, и для папки больше одной страницы.
  function fetchCategoryIds(mode, category) {
    var ids = [];
    function step(offset) {
      return fetch("/api/plan?mode=" + encodeURIComponent(mode) +
                   "&category=" + encodeURIComponent(category) +
                   "&offset=" + offset + "&limit=" + PLAN_ID_PAGE_SIZE)
        .then(function (r) { return r.json(); })
        .then(function (page) {
          var items = page.items || [];
          items.forEach(function (it) { ids.push(it.file_id); });
          if (items.length && ids.length < page.total) return step(offset + items.length);
          return ids;
        });
    }
    return step(0);
  }

  // Кнопка правки в самой строке: одиночный файл — частый случай, ради него не
  // нужно идти в выделение. Метка/подпись кнопки переключаются по состоянию строки.
  function overrideRowButton(tr, item) {
    var btn = makeBtn(null, null, I18N.override_exclude_button, "btn-sm override-row-btn");
    btn.addEventListener("click", function () {
      var action = tr.dataset.override ? "clear" : "exclude";
      applyOverride(action, [item.file_id], null, function () {
        markOverrideRow(tr, action === "clear" ? null : "exclude", null);
      });
    });
    return btn;
  }

  // Панель над деревом: правка применяется к ВЫДЕЛЕНИЮ (те же чекбоксы
  // .row-select, что и «Удалить выбранное»); одиночный файл — выделение из одного.
  function wireOverrideControls(containerId) {
    var container = document.getElementById(containerId);
    var excludeBtn = document.getElementById("city-override-exclude-btn");
    var moveBtn = document.getElementById("city-override-move-btn");
    var clearBtn = document.getElementById("city-override-clear-btn");
    var select = document.getElementById("city-override-target");
    var countEl = document.getElementById("city-override-count");

    function selectedIds() {
      return Array.prototype.slice.call(container.querySelectorAll(".row-select:checked"))
          .map(function (b) { return parseInt(b.value, 10); });
    }
    function refresh() {
      var n = selectedIds().length;
      countEl.textContent = n ? " (" + n + ")" : "";
      var dead = uiBusy() || n === 0;    // F145: these write `manual_overrides`
      excludeBtn.disabled = dead;
      clearBtn.disabled = dead;
      moveBtn.disabled = dead;
    }
    registerBusyRefresh(refresh);
    function apply(action) {
      var ids = selectedIds();
      if (!ids.length) return;
      var target = null;
      if (action === "reassign") {
        target = select.value;
        if (!target) { window.alert(I18N.override_alert_choose_target); return; }
      }
      applyOverride(action, ids, target, function (applied) {
        markRowsOverride(container, applied, action === "clear" ? null : action, target);
      });
    }
    container.addEventListener("change", function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("row-select")) refresh();
    });
    excludeBtn.addEventListener("click", function () { apply("exclude"); });
    moveBtn.addEventListener("click", function () { apply("reassign"); });
    clearBtn.addEventListener("click", function () { apply("clear"); });
    refresh();
  }

  // Список целей переноса = папки текущего плана из уже загруженного агрегата
  // (отдельный эндпойнт не нужен). Перетаскивание плитки в узел дерева не
  // реализуем: дерево ленивое, узел нераскрытой (и потому отсутствующей в DOM)
  // папки не может быть целью drop — список даёт доступ ко ВСЕМ папкам раскладки,
  // как и требует фича.
  function fillOverrideTargets(categories) {
    var select = document.getElementById("city-override-target");
    var previous = select.value;
    select.textContent = "";
    var empty = document.createElement("option");
    empty.value = "";
    empty.textContent = I18N.override_target_placeholder;
    select.appendChild(empty);
    categories.forEach(function (row) {
      var opt = document.createElement("option");
      opt.value = row.category;
      opt.textContent = row.category;
      select.appendChild(opt);
    });
    select.value = previous;
  }

  // --- F85c: место, назначенное человеком, — сразу на всю группу ---------
  // У этих файлов не осталось ни одного сигнала: ни GPS, ни соседей по времени,
  // ни имени папки. Место знает только владелец, поэтому задача не «угадать
  // точнее», а дать назначить его ПАЧКОЙ — событию целиком или исходной папке
  // целиком. Пишется отдельно от places (её geo перезаписывает целиком) и
  // применяется при построении плана; на диске здесь ничего не двигается.

  var PLACE_SEARCH_DELAY = 250;  // мс: поиск идёт по нажатию клавиш, не по каждой

  // Язык интерфейса берём из <html lang>: он уже проставлен сервером, отдельного
  // состояния для этого заводить незачем. (initLang() держит одноимённую локальную
  // переменную — имя здесь другое намеренно.)
  function uiLang() {
    return document.documentElement.getAttribute("lang") || "en";
  }

  // Поле выбора места. Сервер отвечает ТОЧНЫМИ совпадениями по локальной базе
  // (та же пара city_ids_by_name/country_cc_by_name, что и у --where), поэтому
  // список короткий и однозначный: одноимённые города различаются регионом.
  function renderPlacePicker(container) {
    var input = document.createElement("input");
    input.type = "text";
    input.className = "place-input";
    input.placeholder = I18N.place_search_placeholder;
    var select = document.createElement("select");
    select.className = "place-options";
    select.disabled = true;
    var results = [];
    var timer = null;

    function fill(list) {
      results = list || [];
      select.textContent = "";
      results.forEach(function (r, i) {
        var opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = r.label;
        select.appendChild(opt);
      });
      select.disabled = results.length === 0;
    }

    function search() {
      var q = input.value.trim();
      if (!q) { fill([]); return; }
      fetch("/api/places/search?lang=" + encodeURIComponent(uiLang()) +
            "&q=" + encodeURIComponent(q))
        .then(function (r) { return r.json(); })
        .then(function (data) { fill(data && data.results); })
        .catch(function () { fill([]); });
    }

    input.addEventListener("input", function () {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(search, PLACE_SEARCH_DELAY);
    });
    container.appendChild(input);
    container.appendChild(select);
    return {
      chosen: function () {
        if (!results.length) return null;
        return results[parseInt(select.value, 10)] || null;
      },
      typed: function () { return input.value.trim(); }
    };
  }

  function placeStatusEl() {
    return document.getElementById("place-status");
  }

  function postPlace(body, statusEl, onDone) {
    return postJson("/api/place", body).then(function (resp) {
      if (!resp || !resp.ok) {
        statusEl.textContent = I18N.place_error_prefix + ((resp && resp.error) || "");
        return;
      }
      var text = fmt(body.action === "clear" ? I18N.place_cleared_status
                                             : I18N.place_assigned_status,
                     { n: resp.affected });
      if (resp.skipped_gps) text += fmt(I18N.place_skipped_gps, { n: resp.skipped_gps });
      statusEl.textContent = text;
      // Кадры с точными координатами не перезаписываются молча: камера знала
      // место в момент съёмки лучше, чем память о поездке. Это отдельное решение,
      // и спрашивают о нём ровно один раз.
      if (resp.skipped_gps && !body.include_gps &&
          window.confirm(fmt(I18N.place_include_gps_confirm, { n: resp.skipped_gps }))) {
        body.include_gps = true;
        return postPlace(body, statusEl, onDone);
      }
      if (onDone) onDone(resp);
    }).catch(function (err) {
      statusEl.textContent = I18N.place_error_prefix + err;
    });
  }

  // Одно действие на группу: подтверждение называет и место, и размер захвата —
  // цена ошибки тем выше, чем крупнее группа.
  function assignPlace(picker, kind, selector, confirmKey, confirmVals, statusEl, onDone) {
    var chosen = picker.chosen();
    if (!chosen) {
      statusEl.textContent = picker.typed() ? I18N.place_not_found : "";
      window.alert(I18N.place_alert_choose);
      return;
    }
    confirmVals.place = chosen.label;
    if (!window.confirm(fmt(I18N[confirmKey], confirmVals))) return;
    postPlace({ kind: kind, selector: selector, action: "assign",
                country: chosen.country, city_geonameid: chosen.city_geonameid },
              statusEl, onDone);
  }

  function clearPlace(kind, selector, confirmKey, confirmVals, statusEl, onDone) {
    if (!window.confirm(fmt(I18N[confirmKey], confirmVals))) return;
    postPlace({ kind: kind, selector: selector, action: "clear" }, statusEl, onDone);
  }

  var cityPlacePicker = null;

  // Кнопка в строке плана: место назначается ИСХОДНОЙ папке кадра целиком — по ней
  // и видно, что кадры одной поездки лежат вместе. Строка с уже назначенным местом
  // предлагает обратное действие, как и кнопка ручных правок рядом.
  function placeRowButton(item) {
    var manual = item.place_confidence === "manual";
    var btn = makeBtn(null, "pin", manual ? I18N.place_clear_button
                                          : I18N.place_folder_button,
        "btn-sm place-row-btn");
    btn.disabled = !item.src_path;
    btn.addEventListener("click", function () {
      var statusEl = placeStatusEl();
      var vals = { dir: item.src_dir || item.src_path };
      var done = refreshPlan;
      if (manual) {
        clearPlace("source_dir", item.src_path, "place_folder_clear_confirm",
                   vals, statusEl, done);
      } else {
        assignPlace(cityPlacePicker, "source_dir", item.src_path,
                    "place_folder_confirm", vals, statusEl, done);
      }
    });
    return btn;
  }

  function renderFiles(files) {
    var table = document.createElement("table");
    files.forEach(function (item) {
      var tr = document.createElement("tr");
      var tdSelect = document.createElement("td");
      var checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "row-select";
      checkbox.value = String(item.file_id);
      checkbox.title = I18N.select_for_delete;
      tdSelect.appendChild(checkbox);
      tr.appendChild(tdSelect);
      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(item.file_id, null, 0, item.thumb_url, item.video));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = item.name;
      nameEl.title = item.src_path ? item.src_path + "\\" + item.name : item.name;
      tdThumb.appendChild(nameEl);
      tr.appendChild(tdThumb);
      var tdMeta = document.createElement("td");
      tdMeta.className = "plan-meta";
      // Исходная папка идёт первой: по ней чаще всего и видно, верна ли догадка
      // («Колизей» из папки «рускеала» — очевидная ошибка). Полный путь — в тултипе.
      tdMeta.textContent = [item.src_dir, item.date, item.geo, item.category]
          .filter(Boolean).join(" \u00b7 ");
      if (item.src_path) { tdMeta.title = item.src_path; }
      // F85c: место, выбранное человеком, помечено отдельно — иначе его не отличить
      // от выведенного программой, а это разные по надёжности вещи.
      if (item.place_confidence === "manual") {
        var placeChip = document.createElement("span");
        placeChip.className = "chip chip-good place-manual";
        placeChip.textContent = I18N.place_manual_mark;
        tdMeta.appendChild(placeChip);
      }
      tr.appendChild(tdMeta);
      var tdActions = document.createElement("td");
      tdActions.className = "plan-actions";
      var btnDelete = makeBtn("danger", "trash", I18N.delete, "btn-sm");
      btnDelete.addEventListener("click", function () {
        deletePhoto(item.file_id, function () { tr.remove(); });
      });
      tdActions.appendChild(btnDelete);
      tdActions.appendChild(overrideRowButton(tr, item));
      tdActions.appendChild(placeRowButton(item));
      tr.appendChild(tdActions);
      // F77: пометка из ответа плана — строка приходит уже размеченной.
      markOverrideRow(tr, item.override || null, item.override_target || null);
      table.appendChild(tr);
    });
    return wrapTable(table);
  }

  // F70: страница файлов одной папки. Первая грузится при раскрытии узла,
  // следующие — по кнопке «Загрузить ещё»; `total` из ответа показывается как
  // «показано N из M». DOM-узлы существуют только для реально загруженных строк.
  var PLAN_PAGE_SIZE = 200;

  function renderCategoryFiles(mode, category) {
    var wrap = document.createElement("div");
    var status = document.createElement("div");
    status.className = "plan-page-status";
    var moreBtn = makeBtn("ghost", null, I18N.plan_load_more, "btn-sm");
    moreBtn.style.display = "none";
    wrap.appendChild(status);
    wrap.appendChild(moreBtn);
    var loaded = 0;
    var busy = false;

    function loadNext() {
      if (busy) return;
      busy = true;
      moreBtn.disabled = true;
      fetch("/api/plan?mode=" + encodeURIComponent(mode) +
            "&category=" + encodeURIComponent(category) +
            "&offset=" + loaded + "&limit=" + PLAN_PAGE_SIZE)
        .then(function (r) { return r.json(); })
        .then(function (page) {
          var items = page.items || [];
          if (items.length) wrap.insertBefore(renderFiles(items), status);
          loaded += items.length;
          busy = false;
          moreBtn.disabled = false;
          status.textContent = fmt(I18N.plan_shown_of, { n: loaded, all: page.total });
          moreBtn.style.display = (items.length && loaded < page.total) ? "" : "none";
        })
        .catch(function (err) {
          busy = false;
          moreBtn.disabled = false;
          status.textContent = I18N.error_loading_plan + err;
        });
    }

    moreBtn.addEventListener("click", loadNext);
    loadNext();
    return wrap;
  }

  // Ленивое построение узла дерева: содержимое папки (страница файлов + дочерние
  // папки) создаётся ТОЛЬКО при первом раскрытии <details>, и файлы при этом
  // запрашиваются у сервера отдельным запросом — до раскрытия ни одного файла
  // папки в браузере нет вообще.
  function renderCategoryNode(mode, name, node) {
    var details = document.createElement("details");
    var summary = document.createElement("summary");
    summary.textContent = name + " (" + node.count + ")";
    if (node.category) {
      // F77: «не трогать» на папку целиком — кнопка в заголовке категории.
      // Клик внутри <summary> иначе раскрывает/сворачивает узел, поэтому событие
      // до <details> не доходит.
      var folderBtn = makeBtn("danger", null, I18N.override_exclude_folder_button,
          "btn-sm override-folder-btn");
      folderBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (!window.confirm(fmt(I18N.override_exclude_folder_confirm, { n: node.count }))) return;
        folderBtn.disabled = true;
        fetchCategoryIds(mode, node.category).then(function (ids) {
          if (!ids.length) { folderBtn.disabled = false; return; }
          return applyOverride("exclude", ids, null, function (applied) {
            markRowsOverride(details, applied, "exclude", null);
          });
        }).then(function () { folderBtn.disabled = false; })
          .catch(function (err) {
            folderBtn.disabled = false;
            overrideStatusEl().textContent = I18N.override_error_prefix + err;
          });
      });
      summary.appendChild(folderBtn);
    }
    details.appendChild(summary);
    var built = false;
    details.addEventListener("toggle", function () {
      if (!details.open || built) return;
      built = true;
      if (node.category) details.appendChild(renderCategoryFiles(mode, node.category));
      Object.keys(node.children).sort().forEach(function (childName) {
        details.appendChild(renderCategoryNode(mode, childName, node.children[childName]));
      });
    });
    return details;
  }

  // F43: счётчики последнего плана.
  // F104: the numbers of the confirmation itself now come from /api/sort/summary (it
  // also knows the volume and what is already in the destination); what stays here is
  // the one question the START button needs answered — is there anything to lay out at
  // all. `planLoaded` keeps "nothing to lay out" apart from "not counted yet".
  // F192: both are about the CHOSEN criterion, not about the city one — the tab lays
  // out by whatever `#layout-by` says, so "is there anything to lay out" is a question
  // about that plan and has to be re-answered every time the criterion changes.
  var planCount = 0;
  var planLoaded = false;

  // F192: the criterion the whole tab is about — `sorter.MODES`, the same values
  // `/api/plan?mode=` and `sorta sort --by` take. Read from the field rather than kept
  // in a variable of its own: the field IS the state, and a second copy of it is how
  // the tree and the apply start laying out by different things.
  function layoutBy() {
    var select = document.getElementById("layout-by");
    return select ? select.value : "city";
  }

  // renderPlanTab: дерево папок плана режима (city/person/event) из агрегата —
  // общий код, переиспользуемый всеми план-вкладками (U2).
  function renderPlanTab(mode, containerId) {
    var container = document.getElementById(containerId);
    fetch("/api/plan?mode=" + encodeURIComponent(mode))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var categories = data.categories || [];
        // F77: помеченные «не трогать» остаются в списке, но НЕ переезжают —
        // в подтверждении раскладки их считать нельзя.
        planCount = (data.total || 0) - (data.excluded || 0);
        planLoaded = true;
        updateBusyControlsDisabled();
        fillOverrideTargets(categories);
        container.textContent = "";
        if (!categories.length) {
          container.appendChild(stateEl("empty", I18N.plan_empty));
          return;
        }
        var root = buildCategoryTree(categories);
        Object.keys(root.children).sort().forEach(function (name) {
          container.appendChild(renderCategoryNode(mode, name, root.children[name]));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_plan + err));
      });
  }

  // The one call every "the plan may have changed" site makes — an apply that finished,
  // a correction, a place assignment, a folder-language change, a switch of criterion.
  // None of them may pass a criterion of its own: they all mean "redraw what the tab is
  // showing now".
  function refreshPlan() {
    renderPlanTab(layoutBy(), "tree-city");
  }

  cityPlacePicker = renderPlacePicker(document.getElementById("city-place-picker"));
  refreshPlan();
  wireBulkDelete("tree-city", "city-delete-selected-btn", "city-delete-selected-count",
                 "city-selection-controls");
  wireOverrideControls("tree-city");

  // Switching the criterion re-asks the server for the whole plan, so until the answer
  // arrives there is no count — the start button goes dead rather than staying live with
  // the number of the previous criterion behind it.
  document.getElementById("layout-by").addEventListener("change", function () {
    planCount = 0;
    planLoaded = false;
    updateBusyControlsDisabled();
    document.getElementById("layout-by-hint").textContent =
        I18N["layout_by_hint_" + this.value] || "";
    document.getElementById("tree-city").textContent = "";
    document.getElementById("tree-city").appendChild(stateEl("loading", I18N.loading));
    refreshPlan();
  });

  // F192: everything that is not one of the two questions sits behind the gear —
  // opened in place, above the tree it is used against, and closed again by the button
  // that opened it.
  function toggleLayoutOptions(open) {
    document.getElementById("layout-options").hidden = !open;
    document.getElementById("layout-options-btn")
        .setAttribute("aria-expanded", open ? "true" : "false");
  }

  document.getElementById("layout-options-btn").addEventListener("click", function () {
    toggleLayoutOptions(document.getElementById("layout-options").hidden);
  });
  document.getElementById("layout-options-close").addEventListener("click", function () {
    toggleLayoutOptions(false);
  });

  document.querySelectorAll(".expand-all-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll("details").forEach(function (d) { d.open = true; });
    });
  });
  document.querySelectorAll(".collapse-all-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      document.querySelectorAll("details").forEach(function (d) { d.open = false; });
    });
  });
  document.getElementById("top-btn").addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // --- вкладки ---------------------------------------------------------

  var dupesLoaded = false;
  var reviewLoaded = false;
  var movesLoaded = false;
  var clustersLoaded = false;
  var eventsLoaded = false;
  var junkLoaded = false;
  var animalsLoaded = false;

  // F133: four tabs named after what a person does with the collection, plus "Moves" as
  // it was. "Overview" holds the state AND the run that produces it; "Slices" holds
  // people/events/animals and the classifier's classes as switchable panels of its own.
  var TAB_NAMES = ["overview", "review", "layout", "slices", "moves"];

  function activateTab(name) {
    TAB_NAMES.forEach(function (t) {
      document.getElementById("tab-btn-" + t).classList.toggle("active", t === name);
      document.getElementById("tab-" + t).classList.toggle("active", t === name);
    });
    // #36: чекбокс «не спрашивать удаление» релевантен только там, где удаляют
    // (Раскладка/Разбор) — на остальных вкладках это шум, прячем.
    document.getElementById("delete-remember-row").style.display =
        (name === "layout" || name === "review") ? "" : "none";
    if (name === "review" && !reviewLoaded) {
      reviewLoaded = true;
      loadReview();
    }
    if (name === "slices") loadSlices();
    if (name === "moves" && !movesLoaded) {
      movesLoaded = true;
      loadMoves();
    }
    // F133: the order warning is re-asked on every open, for the same reason the numbers
    // of "Overview" are — the person has just come back from the Review, and a warning
    // one decision out of date is the one that teaches people to ignore warnings.
    if (name === "layout") loadLayoutWarning();
    // F108: обзор — единственная вкладка без флага «уже загружено». Его открывают
    // ПОСЛЕ прогона, чтобы увидеть изменения, и устаревшая цифра здесь хуже
    // отсутствующей — поэтому числа перезапрашиваются на каждом открытии.
    if (name === "overview") loadOverview();
  }

  TAB_NAMES.forEach(function (t) {
    document.getElementById("tab-btn-" + t).addEventListener("click", function () {
      activateTab(t);
    });
  });

  // --- F133: the slices ------------------------------------------------------
  // The pin row is BUILT, never written out in the markup: F129 replaces the fixed list
  // with a query, and a row of hand-written buttons would have to be thrown away then.
  // Three of the pins (people/events/animals) show a panel that used to be a tab; the
  // rest are the classifier's classes — products, screenshots, documents and the others —
  // and they all share the one panel `/api/junk` already fills, with its counts, its
  // paging and its rule that a document is never decoded for display.

  // Which classes go first. The order is the product's, not the counter's: a person looks
  // for "products, screenshots, documents", and whichever of them happens to be biggest
  // this month is not a reason to reshuffle the row under them.
  var SLICE_CLASS_ORDER = ["product", "screenshot", "document"];

  var slicePins = [];
  var sliceCurrent = null;
  var slicePending = null;
  var sliceVisibility = { person: false, event: false, animal: false, face: false };
  // F156: per built-in slice — null (it holds something), "not_run" (the stage that fills
  // it never ran) or "none_found" (it ran and the collection has none). A zero on its own
  // reads as the second when it is nearly always the first.
  var sliceReasons = {};
  var junkBucketCounts = [];
  // F152: the counters of the three face pins, `null` for each of them until the faces
  // stage has run — the pin then carries no number at all, because "0 photographs with
  // people" is a claim and "nobody has looked yet" is the truth.
  var faceSliceCounts = {};

  function sliceKeyId(key) {
    return "slice-pin-" + key.replace(/[^a-z0-9]+/g, "-");
  }

  // F156: the empty state of a built-in slice, which has to say WHICH empty it is.
  // `foundNothing` is the slice's own "none were found" line and is used unchanged when
  // the stage really did run; "the stage has not run" is a different sentence and comes
  // with the one action that changes it — a button to the run screen, where the checkbox
  // that decides whether this is computed at all also lives.
  function sliceEmptyState(key, foundNothing) {
    if (sliceReasons[key] !== "not_run") return stateEl("empty", foundNothing);
    var box = stateEl("empty", I18N.slice_not_computed);
    var goto = makeBtn("ghost", null, I18N.slice_goto_process, "btn-sm");
    goto.addEventListener("click", function () { activateTab("overview"); });
    box.appendChild(goto);
    return box;
  }

  function slicePanelId(key) {
    if (key.indexOf("junk") === 0) return "tab-junk";
    if (key.indexOf("face:") === 0) return "tab-face";
    if (key.indexOf("query:") === 0) return "tab-query";
    return "tab-" + key;
  }

  function buildSlicePins() {
    var pins = [];
    // F152: first in the row, and deliberately: on the live collection these are the
    // largest slices there are, and until now the row opened with the smallest.
    if (sliceVisibility.face) {
      FACE_SLICES.forEach(function (name) {
        var count = faceSliceCounts[name];
        pins.push({ key: "face:" + name, label: I18N["face_slice_" + name],
                    count: (count === null || count === undefined) ? undefined : count,
                    faceSlice: name });
      });
    }
    // F151: the pinned queries, and high in the row on purpose. Children are 22% of a
    // labelled sample and products 10% — the two largest populations the product had no
    // slice for at all — and a pin nobody scrolls to is a slice nobody knows exists: the
    // measurement is how the user found out there were ~4 860 photographs of children,
    // which is a thing the product should have said. They carry NO count: a ranking has
    // no size, and a number beside "Children" would read as "your archive holds this
    // many". The mark in the label is what keeps them apart from the exact slices.
    (savedSlices || []).forEach(function (s) {
      pins.push({ key: "query:" + s.slice, label: savedSlicePinLabel(s.slice),
                  savedSlice: s.slice });
    });
    if (sliceVisibility.person) pins.push({ key: "person", label: I18N.tab_person });
    if (sliceVisibility.event) pins.push({ key: "event", label: I18N.tab_event });
    if (sliceVisibility.animal) pins.push({ key: "animal", label: I18N.tab_animal });
    var rest = junkBucketCounts.slice().sort(function (a, b) {
      var ai = SLICE_CLASS_ORDER.indexOf(a.verdict);
      var bi = SLICE_CLASS_ORDER.indexOf(b.verdict);
      if (ai !== bi) return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
      return a.verdict < b.verdict ? -1 : 1;
    });
    rest.forEach(function (b) {
      pins.push({ key: "junk:" + b.verdict, label: junkBucketLabel(b.verdict),
                  count: b.count, bucket: b.verdict });
    });
    if (rest.length) {
      pins.push({ key: "junk", label: I18N.tab_junk, bucket: null,
                  count: rest.reduce(function (acc, b) { return acc + b.count; }, 0) });
    }
    return pins;
  }

  function renderSlicePins() {
    slicePins = buildSlicePins();
    var box = document.getElementById("slice-pins");
    box.textContent = "";
    slicePins.forEach(function (pin) {
      var label = pin.label + (pin.count === undefined ? "" : " (" + pin.count + ")");
      var btn = makeBtn(null, null, label, "btn-sm review-slice-btn");
      btn.id = sliceKeyId(pin.key);
      if (pin.key === sliceCurrent) btn.classList.add("active");
      btn.addEventListener("click", function () { selectSlice(pin.key); });
      box.appendChild(btn);
    });
    // F134: "no slices yet" must not sit under a search that is working — the search
    // line is a slice of its own the moment it has results on screen.
    document.getElementById("slice-empty").style.display =
        (slicePins.length || searchActive) ? "none" : "";
  }

  function selectSlice(key) {
    var pin = null;
    slicePins.forEach(function (p) { if (p.key === key) pin = p; });
    if (!pin) return;
    sliceCurrent = key;
    // F134: the search results are a panel of this tab like any other, so picking a pin
    // puts them away — one panel is visible at a time, whichever one it is.
    searchActive = false;
    var panelId = slicePanelId(key);
    ["tab-person", "tab-event", "tab-animal", "tab-junk", "tab-face",
     "tab-query", "tab-search"].forEach(function (id) {
      document.getElementById(id).classList.toggle("active", id === panelId);
    });
    slicePins.forEach(function (p) {
      var btn = document.getElementById(sliceKeyId(p.key));
      if (btn) btn.classList.toggle("active", p.key === key);
    });
    // F152: three pins, one panel — the junk-bucket arrangement. The page is refetched
    // whenever the slice changes, because the panel holds one slice at a time.
    if (pin.faceSlice !== undefined && (faceSlice !== pin.faceSlice || !faceLoaded)) {
      faceLoaded = true;
      faceSlice = pin.faceSlice;
      loadFaceSlice();
    }
    // F151: the same arrangement — several pins over one panel, refetched whenever the
    // slice changes, because the panel holds one ranking at a time.
    if (pin.savedSlice !== undefined && (querySlice !== pin.savedSlice || !queryLoaded)) {
      queryLoaded = true;
      querySlice = pin.savedSlice;
      loadSavedSlice();
    }
    if (key === "person" && !clustersLoaded) {
      clustersLoaded = true;
      loadClusters();
    }
    if (key === "event" && !eventsLoaded) {
      eventsLoaded = true;
      loadEvents();
    }
    if (key === "animal" && !animalsLoaded) {
      animalsLoaded = true;
      loadAnimals();
    }
    if (pin.bucket !== undefined && (junkBucket !== pin.bucket || !junkLoaded)) {
      junkLoaded = true;
      junkBucket = pin.bucket;
      loadJunk();
    }
  }

  function loadSlices() {
    // F134: the state of the search index is asked for on every open, for the reason the
    // numbers of "Overview" are — the person may have just come back from a run, and a
    // line that stays disabled after the run that enabled it is the worst of both states.
    fetchSearchState();
    // The counters of the class pins come from the route that already serves them, asked
    // for zero items: the counts are the whole answer here. F152 asks its own route the
    // same way — a page of zero cards, three numbers back.
    return Promise.all([
      fetch("/api/junk?offset=0&limit=0")
        .then(function (r) { return r.json(); })
        .then(function (data) { junkBucketCounts = data.buckets || []; })
        .catch(function () {}),
      fetch("/api/face-slices?offset=0&limit=0")
        .then(function (r) { return r.json(); })
        .then(function (data) { applyFaceCounts(data); })
        .catch(function () {}),
      // F151: the pinned queries come from the config, so they are ASKED FOR rather than
      // written into the row here — an edit of `features.saved_slices` reaches the pins
      // on the next open of the tab and never through a restart. Without a `slice` the
      // route ranks nothing and loads no model: this call costs a list.
      fetch("/api/saved-slices?offset=0&limit=0")
        .then(function (r) { return r.json(); })
        .then(function (data) {
          // The F152 rule for when a pin exists at all: as soon as the index holds a
          // photograph, and not once the slice is known to hold something. Their empty
          // state is a SENTENCE ("switch the search index on and process the collection")
          // and a pin that hides itself never gets to say it — while over an index with
          // no photographs in it there is nothing to say, and "no slices yet" is the
          // honest line.
          savedSlices = (data.photos ? data.slices : []) || [];
          savedSlicesMax = data.max_pinned || 0;
        })
        .catch(function () {}),
    ])
      .then(function () {
        renderSlicePins();
        if (!slicePins.length || searchActive) return;
        var want = slicePending;
        slicePending = null;
        var still = false;
        slicePins.forEach(function (p) {
          if (p.key === (want || sliceCurrent)) still = true;
        });
        selectSlice(still ? (want || sliceCurrent) : slicePins[0].key);
      });
  }

  // A number on "Overview" leads to its SLICE, not merely to the tab holding it — and
  // the pins may not have been built yet, so the wish is remembered and honoured by the
  // load that the tab switch starts.
  function gotoSlice(key) {
    slicePending = key;
    activateTab("slices");
  }

  // --- F134: поиск словами в блоке «Срезы» -----------------------------------
  // The line F133 drew and left disabled. Everything here is arranged around one state
  // that is not a failure: an index nobody has computed yet. `/api/search` answers with
  // the state of the index on EVERY request — including the empty query this tab asks on
  // open, which never reaches the model — so the line can stay disabled with the reason
  // beside it instead of ranking nothing. An empty list of results would read as "you
  // have no photographs like that": a conclusion about somebody's own archive, drawn
  // from a table that was never filled.

  var searchState = null;
  var searchActive = false;   // результаты на экране -> панель среза занята поиском

  function searchStateText(state) {
    if (!state) return I18N.search_state_checking;
    // Two unavailable states, two sentences: "run it" and "run it AGAIN, that index was
    // computed by another model" are different instructions, and the model is named
    // because a reason nobody can act on is not a reason.
    if (state.state === "other_model") {
      return fmt(I18N.search_state_other_model, { model: state.index_model || "?" });
    }
    if (state.state === "empty") return I18N.search_state_empty;
    if (state.state === "partial") {
      return fmt(I18N.search_state_partial, { n: state.indexed, all: state.total });
    }
    return fmt(I18N.search_state_ready, { all: state.total });
  }

  function applySearchState(state) {
    searchState = state;
    var available = !!(state && state.available);
    // F189: a NAME is answered without the index, so the line stays usable while there is
    // somebody named — otherwise the whole feature would be behind a disabled field on the
    // default config, which is the one a person has on the day they name their first
    // cluster. The reason the ranking cannot run is still said, with the name sentence in
    // front of it: both facts are true at once.
    var usable = available || !!(state && state.names);
    document.getElementById("slice-query").disabled = !usable;
    document.getElementById("slice-query-btn").disabled = !usable;
    document.getElementById("slice-query-hint").textContent =
        (!available && usable ? I18N.search_state_names_only + " " : "") +
        searchStateText(state);
    // The way out of both unavailable states is a run of the collection, and the run
    // lives on "Overview" — a reason without the way to it is a dead end.
    document.getElementById("slice-query-goto").style.display =
        (state && !available) ? "" : "none";
  }

  function fetchSearchState() {
    // An empty `q`: the state of the index is the whole question here, and the server
    // loads no model to answer it.
    return fetch("/api/search?q=")
      .then(function (r) { return r.json(); })
      .then(function (data) { applySearchState(data); })
      .catch(function () {});
  }

  function showSearchPanel() {
    searchActive = true;
    ["tab-person", "tab-event", "tab-animal", "tab-junk", "tab-face",
     "tab-query"].forEach(function (id) {
      document.getElementById(id).classList.remove("active");
    });
    document.getElementById("tab-search").classList.add("active");
    slicePins.forEach(function (p) {
      var btn = document.getElementById(sliceKeyId(p.key));
      if (btn) btn.classList.remove("active");
    });
    sliceCurrent = null;
    document.getElementById("slice-empty").style.display = "none";
  }

  function renderSearchCard(item) {
    var card = document.createElement("div");
    card.className = "search-card";
    if (item.thumb_url) {
      card.appendChild(
          clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    } else {
      // F133's rule, and the search must not become the way around it: a sensitive class
      // is never decoded for display. The server sent no link, so nothing here asks
      // /thumb for one.
      var stub = document.createElement("div");
      stub.className = "junk-doc-box";
      stub.textContent = I18N.junk_document_no_preview;
      card.appendChild(stub);
    }
    var name = document.createElement("span");
    name.className = "search-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "search-card-meta";
    meta.textContent = item.date || "";
    card.appendChild(meta);
    // The score is on every card of a RANKING because it is the only thing that explains
    // the order — this ranks, it does not classify, and the reader decides where the list
    // stops being about their query. F189: a selection has no order to explain and the
    // server sends no score for one; a «близость 0.000» here would be a number nobody
    // measured, on a list where every frame is present for the same reason.
    if (item.score !== undefined && item.score !== null) {
      var score = document.createElement("span");
      score.className = "search-card-score";
      score.textContent = fmt(I18N.search_score_label,
                              { score: Number(item.score).toFixed(3) });
      card.appendChild(score);
    }
    return card;
  }

  // The album of a query is the album route that already exists: kind='query' and the
  // words themselves as the selector, through the same dry-run-then-confirm path every
  // other album goes through.
  //
  // F189: when the answer on screen is a PERSON, the album is `kind='person'` with the
  // name — the very selection the frames above came out of. Gathering `kind='query'`
  // there would ask CLIP for a word and hand back a folder that does not match the list
  // it was gathered from, under that person's name.
  function renderSearchAlbumControls(query, person) {
    var box = document.getElementById("search-album");
    box.textContent = "";
    if (!query) return;
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      if (person) {
        gatherAlbum("person", person, modeSelect.value, null, null,
            destInput.value.trim() || null, albumStatus);
      } else {
        gatherAlbum("query", query, modeSelect.value, null, null,
            destInput.value.trim() || null, albumStatus);
      }
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  // F189: the other answer, never gone. A string can be both a name and a word, and the
  // rule is that finding the person does not cost the ability to find the word (nor the
  // other way round) — so whichever of the two is on screen, the link to its counterpart
  // is above it.
  function renderSearchOtherAnswer(data) {
    var box = document.getElementById("search-other");
    box.textContent = "";
    if (!data.person) return;
    var label = data.exact
        ? fmt(I18N.search_person_words_link, { q: data.query })
        : fmt(I18N.search_words_person_link, { name: data.person });
    var btn = makeBtn("ghost", null, label, "btn-sm");
    btn.id = "search-other-btn";
    btn.addEventListener("click", function () {
      searchWords = !!data.exact;   // one is the ranking, the other the person
      searchPager.load();
    });
    box.appendChild(btn);
  }

  // F173: the words the pages belong to. A "show more" that read the input field would
  // fetch the continuation of a ranking nobody is looking at as soon as somebody starts
  // typing the next query — the button continues the list on screen, not the field.
  var searchQuery = "";

  // F189: whether the reader has asked for the RANKING of a string that is also somebody's
  // name. It belongs next to `searchQuery` and for the same reason — a "show more" has to
  // continue the answer on screen, and the two answers to one string are different lists.
  var searchWords = false;
  // Whether what is on screen is a person's frames, and whose. Kept rather than read off
  // the last payload, because the counter is repainted (`sync`) without one.
  var searchExact = false;
  var searchPerson = null;

  // The hole this feature was written for. Search was the one user-facing slice with no
  // way past the first page, and the caption said "200 frames" where the truth was "the
  // first 200 of a ranking that does not end here" — over the very slice the measurement
  // found is best built by a query rather than by faces (94% against 64%, F152).
  //
  // No `pageSize`: the size of a page is `features.search_page` and it is the server's to
  // know. Asking without a `limit` is what makes the setting reach the screen — a number
  // repeated in JS is a second copy of the setting, and the copy is the one that goes stale.
  var searchPager = makePager({
    grid: "search-grid",
    cardSelector: ".search-card",
    moreBtn: "search-more-btn",
    shown: "search-shown",
    hint: "search-depth-hint",
    url: function (offset) {
      return "/api/search?q=" + encodeURIComponent(searchQuery) + "&offset=" + offset +
             (searchWords ? "&words=1" : "");
    },
    card: renderSearchCard,
    // Never "nothing was found": a usable index ranks everything it holds, so an empty
    // list is a fact about the index and the answer says which one. F189: an exact answer
    // has its own empty state — the person exists and none of their frames can be shown,
    // which is not a fact about the index at all.
    emptyText: function (data) {
      if (data.exact) return I18N.search_person_no_frames;
      return data.available ? I18N.search_no_frames : searchStateText(data);
    },
    errorText: function () { return I18N.error_loading_search; },
    shownText: function (n, total) {
      // The caption is how a reader tells the two answers apart: an exact selection
      // presented in the ranking's words would be read as the top of a list.
      if (searchExact) {
        return fmt(I18N.search_person_shown_label,
                   { name: searchPerson || searchQuery, shown: n, total: total });
      }
      return fmt(I18N.search_shown_label,
                 { q: searchQuery, shown: n, total: total });
    },
    onData: function (data, append) {
      applySearchState(data);
      searchExact = !!data.exact;
      searchPerson = data.person || null;
      // Which KIND of answer this is, said above the grid: the ranking's warning about
      // thresholds is false of a selection, and the depth trade under "show more" is too.
      document.getElementById("search-kind-hint").textContent =
          searchExact ? I18N.search_person_hint : I18N.search_ranking_hint;
      document.getElementById("search-depth-hint").textContent =
          searchExact ? I18N.search_person_more_hint : I18N.slice_depth_hint;
      // The album gathers the QUERY, not the page, so it is built once per search — and
      // rebuilding it on every "show more" would wipe the destination somebody typed.
      if (!append) {
        var some = (data.items || []).length;
        renderSearchAlbumControls(some ? data.query : "",
                                  searchExact ? data.person : null);
        // F156: and the same condition offers to PIN it. A query with nothing under it is
        // not a slice yet, and the button that saves one appears when there is something
        // to save. A name is pinned the same way — that is the whole reason this feature
        // lives in the parse of the query string.
        showPinButton(some ? data.query : "");
        renderSearchOtherAnswer(data);
      }
    },
  });

  function runSearch() {
    var q = document.getElementById("slice-query").value.trim();
    // An empty query goes nowhere near the model — not from here and not on the server.
    // F189: `names` is the other reason there is something to ask; a string that turns out
    // not to be a name then comes back with the state of the index as its answer, which is
    // the sentence this line has always given.
    if (!q || !(searchState && (searchState.available || searchState.names))) return;
    searchQuery = q;
    // A new string is asked as itself: the "search by words instead" of the previous one
    // must not silently carry over to the next name somebody types.
    searchWords = false;
    searchExact = false;
    searchPerson = null;
    showSearchPanel();
    document.getElementById("search-shown").textContent = "";
    renderSearchAlbumControls("", null);
    showPinButton("");
    document.getElementById("search-other").textContent = "";
    return searchPager.load();
  }

  document.getElementById("slice-query-btn").addEventListener("click", runSearch);
  document.getElementById("slice-query").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); runSearch(); }
  });
  document.getElementById("slice-query-goto").addEventListener("click", function () {
    activateTab("overview");
  });

  // --- F151: the pinned queries (children, products, animals by query) --------
  // The same panel as the search results, fed by words that come from `features.
  // saved_slices` instead of from the field above. Nothing here knows how to rank: the
  // route is `/api/search` with the query taken from the config, so a pinned slice cannot
  // drift away from what a person gets by typing the same words.
  //
  // Two things are deliberately different from the slices around it. The cards carry a
  // SCORE and the panel says the list is an estimate — these are rankings, not marks — and
  // the "show more" button is the primary one rather than a ghost: depth is the only lever
  // of completeness the measurement confirmed (about 60% found in the first portion, about
  // 90% in a doubled one), so the control that turns it is not the quietest thing on the
  // screen.

  var savedSlices = [];     // [{slice, queries}] — the config's own order, the pin order
  var querySlice = null;
  var queryLoaded = false;
  // F156: `features.max_pinned_slices`, straight off the route. Kept rather than hard-coded
  // for the reason the page size is: a number repeated in JS is a second copy of a setting.
  var savedSlicesMax = 0;

  function savedSliceLabel(name) {
    // A slice added to the config gets its own name on the pin: the catalog holds the
    // three that ship, and inventing a translation for the rest would be worse than the
    // word the person wrote themselves.
    return I18N["query_slice_" + name] || name;
  }

  function savedSlicePinLabel(name) {
    return fmt(I18N.query_slice_pin, { name: savedSliceLabel(name) });
  }

  // No `pageSize`: the page is `features.search_page` and it is the server's to know —
  // the search pager's reason, and the same one that keeps the setting out of the JS.
  var queryPager = makePager({
    grid: "query-grid",
    cardSelector: ".search-card",
    moreBtn: "query-more-btn",
    shown: "query-shown",
    hint: "query-depth-hint",
    url: function (offset) {
      return "/api/saved-slices?slice=" + encodeURIComponent(querySlice) +
             "&offset=" + offset;
    },
    card: renderSearchCard,
    // Never "there are no children in your archive": an index that cannot rank says which
    // of its states it is in, exactly as the typed query does (F134).
    emptyText: function (data) {
      if (data.exact) return I18N.search_person_no_frames;
      return data.available ? I18N.search_no_frames : searchStateText(data);
    },
    errorText: function () { return I18N.error_loading_saved_slices; },
    shownText: function (n, total) {
      // F189: a pinned NAME is captioned as a person and not as a slice of a ranking —
      // the same sentence the search line prints for the same string.
      if (querySliceExact) {
        return fmt(I18N.search_person_shown_label,
                   { name: querySlicePerson || savedSliceLabel(querySlice),
                     shown: n, total: total });
      }
      return fmt(I18N.query_slice_shown_label,
                 { name: savedSliceLabel(querySlice), shown: n, total: total });
    },
    onData: function (data, append) {
      applySearchState(data);
      querySliceExact = !!data.exact;
      querySlicePerson = data.person || null;
      // The phrases on screen are what makes "edit it without code" an offer rather than
      // a claim — and they are the answer to "why is this frame here". For a pinned name
      // the answer to that question is the cluster, so the two lines that call this list
      // an estimate say what it really is instead.
      document.getElementById("query-kind-hint").textContent =
          querySliceExact ? I18N.search_person_hint : I18N.query_slice_intro;
      document.getElementById("query-depth-hint").textContent =
          querySliceExact ? I18N.search_person_more_hint : I18N.slice_depth_hint;
      document.getElementById("query-phrases").textContent = querySliceExact ? "" :
          fmt(I18N.query_slice_phrases,
              { phrases: (data.queries || []).join(" · ") });
      // F156: the actions belong to the SLICE and not to the page, so a "show more" leaves
      // them alone — rebuilding the album row would ask for a default destination again and
      // wipe a path somebody had typed into it.
      if (!append) renderQuerySliceActions(data);
    },
  });

  // F189: whether the open pin is a person, and which one — the pinned twin of
  // `searchExact`/`searchPerson`, kept for the same reason.
  var querySliceExact = false;
  var querySlicePerson = null;

  function loadSavedSlice() {
    return queryPager.load();
  }

  // --- F156: a query of one's own, pinned ------------------------------------------
  // The measurement that produced this feature (2026-08-02, 200 random frames): 65 of them
  // — a third — belong to no class at all, and the ten candidate slices for those 65 cover
  // 26%, 23%, 22%, 20%, 18%, 17%, 15%, 12%, 12% and 6%. Not one of them reaches a third of
  // a third, and food, which everyone involved expected to be large, came out at 8 frames.
  // Ten slices for 65 frames out of 200 would be the thirteen-control remote F133 took
  // apart. So the product stops guessing which facets matter and the owner of the archive
  // pins their own — mountains and children for one person, receipts and cars for another.
  //
  // Nothing here ranks: the engine is F129's and the panel is F151's, and what this adds is
  // who writes the list. There are no suggestions to pin anything and there will be none —
  // a product that proposes the pins is the product this feature replaces.

  function isAscii(text) {
    for (var i = 0; i < text.length; i++) {
      if (text.charCodeAt(i) > 127) return false;
    }
    return true;
  }

  function pinStatus(id, text) {
    document.getElementById(id).textContent = text || "";
  }

  // A refusal is always a sentence. `reason` is the server's one word and the catalog holds
  // the three sentences, so the limit reads in the interface language and says the number.
  function pinErrorText(resp) {
    if (resp.reason === "limit") {
      return fmt(I18N.pin_error_limit, { max: resp.max_pinned || savedSlicesMax });
    }
    if (resp.reason === "duplicate") return I18N.pin_error_duplicate;
    if (resp.reason === "empty") return I18N.pin_error_empty;
    return I18N.pin_error_generic + (resp.error || "");
  }

  function showPinButton(query) {
    var btn = document.getElementById("slice-pin-btn");
    btn.style.display = query ? "" : "none";
    btn.setAttribute("data-query", query || "");
    if (!query) pinStatus("slice-pin-status", "");
  }

  document.getElementById("slice-pin-btn").addEventListener("click", function () {
    var query = this.getAttribute("data-query") || "";
    // Guarded here as well as on the server: an empty query saved as a slice would rank the
    // collection by an arbitrary direction and look exactly like an answer.
    if (!query) { pinStatus("slice-pin-status", I18N.pin_error_empty); return; }
    // The warning comes BEFORE the pin and not a week later, when the slice has been
    // quietly ranking badly: the phrases go to the model as they stand, and the index is
    // English until F141 reaches this collection.
    var prompt = fmt(I18N.pin_slice_prompt, { query: query });
    if (!isAscii(query)) prompt = I18N.pin_slice_language_warning + "\n\n" + prompt;
    var name = window.prompt(prompt, query);
    if (name === null) return;
    name = name.trim() || query;
    pinStatus("slice-pin-status", "");
    postJson("/api/saved-slices/pin", { name: name, query: query })
      .then(function (resp) {
        if (!resp.ok) { pinStatus("slice-pin-status", pinErrorText(resp)); return; }
        pinStatus("slice-pin-status", fmt(I18N.pin_slice_done, { name: name }));
        // The new pin is selected right away: a person who has just named a slice is
        // looking for it, and the row it lands in is at the other end of the tab.
        slicePending = "query:" + name;
        loadSlices();
      });
  });

  // The gather row of a pinned slice — the same album every other slice offers
  // (`kind='query'`, `move_batches.mode='album_query'`), through the same
  // dry-run-then-confirm path, so a pin is not a second-class slice on this tab.
  //
  // Offered when the slice asks ONE phrase, which is every slice a person pins from the
  // search line. A slice asking several is ranked by their average and the album route
  // gathers a single wording, so a button there would gather a different list under the
  // slice's name — the panel says so instead.
  function renderQuerySliceAlbum(data) {
    var box = document.getElementById("query-album");
    var queries = data.queries || [];
    var one = (queries.length === 1 && (data.items || []).length) ? queries[0] : "";
    box.textContent = "";
    if (!one) {
      if (queries.length > 1) box.appendChild(stateEl("empty", I18N.pin_album_one_query));
      return;
    }
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "album-name-input";
    nameInput.placeholder = I18N.album_name_placeholder;
    // Pre-filled with the name of the pin: the folder a person expects is the one they
    // named, not the phrase the ranking happens to use.
    nameInput.value = data.slice || "";
    box.appendChild(nameInput);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      // F189: a pinned name gathers the PERSON album, for the reason the search line
      // does — the folder has to hold the list the button was pressed under.
      gatherAlbum(data.person ? "person" : "query", data.person || one, modeSelect.value,
          null, nameInput.value.trim() || null, destInput.value.trim() || null,
          albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  // Where this slice sits in the row, or -1 while no pinned slice is open. The arrows are
  // drawn from it, so it is kept rather than recomputed: a run ending has to be able to
  // wake the controls up without the panel being reloaded.
  var querySliceIndex = -1;

  // Two rules on three controls, met in one place (`registerBusyRefresh`): the ends of the
  // row disable an arrow, and a run disables all three — every one of them writes
  // `config.yaml`, which the server refuses mid-run.
  function refreshQuerySliceControls() {
    var busy = uiBusy();
    document.getElementById("query-up-btn").disabled = busy || querySliceIndex <= 0;
    document.getElementById("query-down-btn").disabled =
        busy || querySliceIndex < 0 || querySliceIndex >= savedSlices.length - 1;
    document.getElementById("query-unpin-btn").disabled = busy || querySliceIndex < 0;
  }

  registerBusyRefresh(refreshQuerySliceControls);

  function renderQuerySliceActions(data) {
    querySliceIndex = -1;
    savedSlices.forEach(function (s, i) {
      if (s.slice === data.slice) querySliceIndex = i;
    });
    refreshQuerySliceControls();
    pinStatus("query-pin-status", "");
    renderQuerySliceAlbum(data);
  }

  // Arrows rather than dragging: one way to reorder, and the one a keyboard reaches. The
  // whole list comes back from the server, so the row on screen cannot drift from the file.
  function moveSavedSlice(delta) {
    if (!querySlice) return;
    postJson("/api/saved-slices/move", { slice: querySlice, delta: delta })
      .then(function (resp) {
        if (!resp.ok) { pinStatus("query-pin-status", pinErrorText(resp)); return; }
        // The panel is reloaded even though the slice has not changed: the arrows are
        // drawn from the POSITION, and a pin that has just reached the top has to lose
        // the arrow that took it there.
        slicePending = "query:" + querySlice;
        queryLoaded = false;
        loadSlices();
      });
  }

  document.getElementById("query-up-btn").addEventListener("click", function () {
    moveSavedSlice(-1);
  });
  document.getElementById("query-down-btn").addEventListener("click", function () {
    moveSavedSlice(1);
  });

  document.getElementById("query-unpin-btn").addEventListener("click", function () {
    if (!querySlice) return;
    // The confirmation names what goes and what stays: unpinning removes a line of the
    // config file and not one photograph.
    if (!window.confirm(fmt(I18N.pin_unpin_confirm,
                            { name: savedSliceLabel(querySlice) }))) return;
    postJson("/api/saved-slices/unpin", { slice: querySlice })
      .then(function (resp) {
        if (!resp.ok) { pinStatus("query-pin-status", pinErrorText(resp)); return; }
        querySlice = null;
        queryLoaded = false;
        slicePending = null;
        loadSlices();
      });
  });

  // F54: «Люди»/«События» скрыты по умолчанию (без мигания) и раскрываются
  // по факту наличия данных в БД (вариант B, stateless) — фетч дешёвых
  // EXISTS-проверок, вызывается при инициализации и после каждого прогона
  // (refreshTabsAfterProcess), т.к. прогон мог впервые породить кластеры/события.
  // F133: these three are pinned slices rather than tabs now, and the rule has not
  // moved — there is no slice while the database has nothing to show in it.
  function applyTabVisibility() {
    fetch("/api/tabs/visibility")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        sliceVisibility = {
          person: !!data.person,
          event: !!data.event,
          // F123: "Animals" follows the same rule — the slice exists exactly when there
          // is something to show (features.pets off => no verdicts at all).
          animal: !!data.animal,
          // F152: and the face slices deliberately do NOT follow it. They appear as soon
          // as the index holds a photograph, before any faces run, because their empty
          // state is a SENTENCE ("the faces stage has not run") and a pin that hides
          // itself never gets to say it.
          face: !!data.face,
        };
        // F156: and WHY each of them is empty, when it is. Two answers and not one —
        // "nobody has looked yet" carries a link to the run screen, "it was computed and
        // there is nothing" is a fact the collection has already stated.
        sliceReasons = data.reasons || {};
        renderSlicePins();
        // A slice that has just disappeared must not stay selected — but the person is
        // never pulled off a TAB, so the fallback is the first slice, not another tab.
        // And only while the tab is open: this runs on page load too, where selecting a
        // slice would fetch a grid nobody has asked to see.
        var still = false;
        slicePins.forEach(function (p) { if (p.key === sliceCurrent) still = true; });
        if (!still) {
          sliceCurrent = null;
          if (slicePins.length &&
              document.getElementById("tab-slices").classList.contains("active")) {
            selectSlice(slicePins[0].key);
          }
        }
      })
      .catch(function () {});
  }

  applyTabVisibility();

  // --- F133: the order warning of the "Layout" tab ---------------------------
  // Frames marked for deletion leave for "_delete" DURING `sort --apply`, at the same
  // moment the canon is built, and albums are hardlinks OUT of the canon. Gather the
  // albums before the junk is thrown out and you get links to what you decided to throw.
  //
  // A hint and nothing else: not one layout control is touched here. The collection is
  // alive, "gather" happens again and again, and steps get in the way of somebody who
  // came back for a single album — a locked tab would cost more than the mistake it
  // guards against.
  function renderLayoutWarning(data) {
    var box = document.getElementById("layout-review-warning");
    var pending = data ? Number(data.pending_total || 0) : 0;
    document.getElementById("layout-review-warning-text").textContent =
        fmt(I18N.layout_review_warning, { n: pending });
    box.style.display = pending ? "" : "none";
  }

  function loadLayoutWarning() {
    // slice=dupes carries no items — the counters are the whole answer.
    return fetch("/api/review?slice=dupes&offset=0&limit=0")
      .then(function (r) { return r.json(); })
      .then(function (data) { renderLayoutWarning(data); })
      .catch(function () { renderLayoutWarning(null); });
  }

  document.getElementById("layout-review-goto-btn").addEventListener("click", function () {
    activateTab("review");
  });

  // --- F133: the settings behind the gear ------------------------------------
  // The very same column and the very same /api/settings — only the place it is opened
  // from has moved. Thirteen keys people come back to about once a month no longer hold
  // a third of the screen at all times.
  function toggleSettingsPanel(open) {
    var panel = document.getElementById("settings-panel");
    panel.hidden = !open;
    document.getElementById("settings-toggle-btn")
        .setAttribute("aria-expanded", open ? "true" : "false");
  }

  document.getElementById("settings-toggle-btn").addEventListener("click", function () {
    toggleSettingsPanel(document.getElementById("settings-panel").hidden);
  });
  document.getElementById("settings-close-btn").addEventListener("click", function () {
    toggleSettingsPanel(false);
  });
  document.getElementById("settings-panel").addEventListener("click", function (e) {
    if (e.target === this) toggleSettingsPanel(false);
  });

  // --- вкладка «Обзор» (F108) --------------------------------------------
  // Все числа приходят одним запросом /api/overview (простые агрегаты по индексу,
  // без построения плана) и рисуются четырьмя карточками: коллекция, место,
  // разбор, раскладка.

  // F145: the empty state draws the SAME rows with a dash where the number will be.
  // Before, it drew an invitation with a button instead — a block of a different height,
  // swapped for the full one the moment the index stopped being empty, i.e. in the middle
  // of a run, right after the `index` stage. Everything below it, the run options among
  // them, jumped down the page while a person was reading. So: the block holds its height
  // from the first paint, the numbers arriving change the text and not the layout, and
  // the list doubles as a statement of what a run will produce.
  var overviewEmpty = false;

  // Числа читают глазами: 7 619 против 7619. toLocaleString берёт разделитель
  // разрядов из локали браузера.
  function overviewNum(n) {
    return Number(n || 0).toLocaleString();
  }

  // F145: the value column of an overview row — the number, or a dash while there is no
  // index to take it from. Separate from overviewNum, which the review slice counters on
  // another tab also use and which must stay a plain formatter.
  function overviewStat(n) {
    if (overviewEmpty) return "\u2014";
    return overviewNum(n);
  }

  function overviewValue(text, extraClass) {
    var el = document.createElement("span");
    el.className = "overview-value" + (extraClass ? " " + extraClass : "");
    el.textContent = text;
    return el;
  }

  // Число, у которого есть своя вкладка, само является переходом на неё. Ноль
  // ссылкой не делаем: вести на заведомо пустую вкладку не за чем.
  // F126: a review number leads to its SLICE, not just to the tab — the workspace has
  // four of them and landing on the wrong one is the same as landing nowhere.
  // F133: the same is now true of "Slices", where people, events, animals and the
  // classifier's classes live side by side.
  function overviewCount(count, tab, slice) {
    if (!tab || !count) return overviewValue(overviewStat(count));
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "overview-value-link";
    btn.textContent = overviewStat(count);
    btn.title = fmt(I18N.overview_goto_hint, { tab: I18N["tab_" + tab] || tab });
    btn.addEventListener("click", function () {
      if (tab === "slices") {
        gotoSlice(slice);
        return;
      }
      activateTab(tab);
      if (slice) selectReviewSlice(slice);
    });
    return btn;
  }

  // F152: a face-slice number, or a dash when the faces stage never ran. `overviewStat`
  // cannot answer this one — it dashes on an empty INDEX, and here the index is full
  // while this particular question has not been asked of it.
  function overviewFaceCount(count, slice) {
    if (count === null || count === undefined) return overviewValue("\u2014");
    return overviewCount(count, "slices", slice);
  }

  function overviewRow(label, valueEl, main) {
    var row = document.createElement("div");
    row.className = "overview-row" + (main ? " overview-row-main" : "");
    var name = document.createElement("span");
    name.className = "overview-label";
    name.textContent = label;
    row.appendChild(name);
    row.appendChild(valueEl);
    return row;
  }

  function overviewCard(title) {
    var card = document.createElement("div");
    card.className = "card overview-card";
    var head = document.createElement("h3");
    head.textContent = title;
    card.appendChild(head);
    return card;
  }

  function overviewSubtitle(text) {
    var el = document.createElement("p");
    el.className = "overview-subtitle";
    el.textContent = text;
    return el;
  }

  function overviewNote(text, warn) {
    var el = document.createElement("p");
    el.className = "overview-note" + (warn ? " overview-note-warn" : "");
    el.textContent = text;
    return el;
  }

  function overviewPlaceLabel(key) {
    return I18N["overview_place_" + key] || key;
  }

  function overviewVerdictLabel(key) {
    return key === "photo" ? I18N.overview_verdict_photo : junkBucketLabel(key);
  }

  function overviewSourceLabel(key) {
    return I18N["overview_source_" + key] || key;
  }

  function overviewTierLabel(key) {
    return key ? (I18N["overview_tier_" + key] || key) : I18N.overview_tier_none;
  }

  function overviewCollectionCard(data) {
    var c = data.collection;
    var card = overviewCard(I18N.overview_group_collection);
    card.appendChild(overviewRow(I18N.overview_files, overviewValue(overviewStat(c.files)), true));
    card.appendChild(overviewRow(I18N.overview_photos, overviewValue(overviewStat(c.photos))));
    card.appendChild(overviewRow(I18N.overview_videos, overviewValue(overviewStat(c.videos))));
    card.appendChild(overviewRow(I18N.overview_duplicates,
                                 overviewCount(c.duplicates, "review", "dupes")));
    card.appendChild(overviewRow(I18N.overview_errors, overviewValue(overviewStat(c.errors))));
    card.appendChild(overviewRow(I18N.overview_events,
                                 overviewCount(c.events, "slices", "event")));
    card.appendChild(overviewRow(I18N.overview_animals,
                                 overviewCount(c.animals, "slices", "animal")));
    // F152: the three face slices, each leading to its own pin. They are the only rows
    // here that can be a dash: without a faces run there is no measurement, and a zero
    // would read as "no photograph of yours has a person on it".
    card.appendChild(overviewRow(I18N.overview_with_people,
                                 overviewFaceCount(c.with_people, "face:people")));
    card.appendChild(overviewRow(I18N.overview_group_photos,
                                 overviewFaceCount(c.group_photos, "face:group")));
    card.appendChild(overviewRow(I18N.overview_portraits,
                                 overviewFaceCount(c.portraits, "face:portrait")));
    if (c.faces_reason === "no_faces_run") {
      card.appendChild(overviewNote(I18N.face_no_faces_run));
    }
    // F126: the slices of the review workspace that have a number of their own.
    card.appendChild(overviewRow(I18N.overview_blurred,
                                 overviewCount(c.blurred, "review", "blurred")));
    card.appendChild(overviewRow(I18N.overview_eyes_closed,
                                 overviewCount(c.eyes_closed, "review", "eyes")));
    card.appendChild(overviewRow(
        I18N.overview_low_resolution,
        overviewCount(c.low_resolution, "review", "low_resolution")));
    return card;
  }

  function overviewPlaceCard(data) {
    var p = data.place;
    var card = overviewCard(I18N.overview_group_place);
    // Главное число группы: каждый такой кадр уедет в «_Без места» — это и есть
    // качество будущей раскладки, поэтому доля в процентах стоит рядом.
    card.appendChild(overviewRow(
        I18N.overview_no_place,
        overviewValue(overviewEmpty ? overviewStat(p.no_place)
                      : overviewStat(p.no_place) + " (" + p.no_place_percent + "%)"),
        true));
    p.confidence.forEach(function (row) {
      // «unknown» — ровно те кадры, что уже названы строкой выше (правилом
      // раскладки); второй раз их не повторяем.
      if (row.key === "unknown") return;
      card.appendChild(overviewRow(overviewPlaceLabel(row.key),
                                   overviewValue(overviewStat(row.count))));
    });
    card.appendChild(overviewNote(I18N.overview_no_place_hint));
    return card;
  }

  function overviewClassesCard(data) {
    var cl = data.classes;
    var card = overviewCard(I18N.overview_group_classes);
    card.appendChild(overviewRow(I18N.overview_classified,
                                 overviewValue(overviewStat(cl.total)), true));
    if (!cl.total) {
      card.appendChild(overviewNote(I18N.overview_not_classified));
      return card;
    }
    cl.verdicts.forEach(function (row) {
      // F133: everything that is not a personal photograph is a pinned slice of its own
      // class — the number leads straight into it rather than into the whole list.
      card.appendChild(overviewRow(
          overviewVerdictLabel(row.key),
          overviewCount(row.count, row.key === "photo" ? null : "slices",
                        "junk:" + row.key)));
    });
    card.appendChild(overviewSubtitle(I18N.overview_by_source));
    cl.sources.forEach(function (row) {
      card.appendChild(overviewRow(overviewSourceLabel(row.key),
                                   overviewValue(overviewStat(row.count))));
    });
    card.appendChild(overviewSubtitle(I18N.overview_by_tier));
    cl.tiers.forEach(function (row) {
      card.appendChild(overviewRow(overviewTierLabel(row.key),
                                   overviewValue(overviewStat(row.count))));
    });
    // Прогонялся ли глубокий ярус — вопрос, который раньше решался запросом в БД.
    card.appendChild(overviewNote(
        cl.vlm_ran ? I18N.overview_vlm_ran : I18N.overview_vlm_not_ran));
    if (cl.updated_at) {
      card.appendChild(overviewNote(fmt(I18N.overview_updated_at, { at: cl.updated_at })));
    }
    return card;
  }

  function overviewLayoutCard(data) {
    var lay = data.layout;
    var card = overviewCard(I18N.overview_group_layout);
    if (!lay.last) {
      card.appendChild(overviewNote(I18N.overview_layout_none));
      return card;
    }
    var last = lay.last;
    // The batch mode is `city` — the canon — and the tab that builds it is "Layout".
    var mode = last.mode === "city"
        ? I18N.tab_layout : (I18N["tab_" + last.mode] || last.mode);
    var op = last.operation === "copy" ? I18N.overview_op_copy : I18N.overview_op_move;
    card.appendChild(overviewRow(I18N.overview_layout_files,
                                 overviewValue(overviewStat(last.files)), true));
    card.appendChild(overviewRow(I18N.overview_layout_done,
                                 overviewCount(last.done, "moves")));
    card.appendChild(overviewRow(I18N.overview_layout_mode,
                                 overviewValue(mode + " \u00b7 " + op, "overview-text")));
    card.appendChild(overviewRow(I18N.overview_layout_started,
                                 overviewValue(last.started_at, "overview-text")));
    card.appendChild(overviewRow(I18N.overview_layout_finished,
                                 overviewValue(last.finished_at || "\u2014", "overview-text")));
    card.appendChild(overviewRow(I18N.overview_layout_dest,
                                 overviewValue(last.dest_root, "overview-text")));
    if (lay.batches > 1) {
      card.appendChild(overviewRow(I18N.overview_layout_batches,
                                   overviewValue(overviewStat(lay.batches))));
    }
    // Незакрытый батч — след прерванного прогона; о нём говорим явно.
    if (last.unfinished || lay.unfinished) {
      card.appendChild(overviewNote(I18N.overview_layout_unfinished, true));
    }
    return card;
  }

  function renderOverview(data) {
    var body = document.getElementById("overview-body");
    body.textContent = "";
    // F145: one flag, read by overviewNum — the four cards below are built the same way
    // either way, and an empty index differs only in what stands in the value column.
    overviewEmpty = !!data.empty;
    if (overviewEmpty) body.appendChild(overviewNote(I18N.overview_empty));
    // F133, restored by F161: the invitation says "enter a photo folder", and the caret
    // goes there. This is the one screen where a first-time reader has nothing to go on,
    // and the field is on the same tab — nobody is taken anywhere. The call was written
    // by F133, disappeared while F135/F138 rebuilt this panel, and the test that asserted
    // it was weakened rather than lost, so it comes back with the assertion.
    // Only when the field is still empty: a path already typed means the caret has been
    // there and may since have moved somewhere the reader chose.
    if (overviewEmpty) {
      var picker = document.getElementById("process-source-dir");
      if (picker && !picker.value) picker.focus();
    }
    var groups = document.createElement("div");
    groups.className = "overview-groups";
    groups.appendChild(overviewCollectionCard(data));
    groups.appendChild(overviewPlaceCard(data));
    groups.appendChild(overviewClassesCard(data));
    groups.appendChild(overviewLayoutCard(data));
    body.appendChild(groups);
  }

  function loadOverview() {
    var body = document.getElementById("overview-body");
    body.textContent = "";
    body.appendChild(stateEl("loading", I18N.loading));
    return fetch("/api/overview")
      .then(function (r) { return r.json(); })
      .then(function (data) { renderOverview(data); })
      .catch(function (err) {
        body.textContent = "";
        body.appendChild(stateEl("error", I18N.error_loading_overview + err));
      });
  }

  // --- вкладка «Обработать» (F36: запуск пайплайна из веба + polling) ----

  // F57: чекбоксы deep/geo-online должны стартовать по факту config.yaml
  // (cfg.naming.vlm_enabled / cfg.geo.provider), а не всегда пустыми — иначе
  // сложно понять, что реально включено, и нельзя увидеть текущее состояние
  // до первого клика. vlmAvailable — установлен ли пакет transformers;
  // приглушённая пометка «VLM не установлен» показывается только когда
  // чекбокс отмечен, но пакета нет (запрос VLM ≠ его реальный запуск —
  // junk.classify штатно фолбэчит на CLIP).
  var vlmAvailable = true;

  function updateVlmMissingWarning() {
    var checked = document.getElementById("process-deep-checkbox").checked;
    document.getElementById("process-deep-vlm-missing").style.display =
        (checked && !vlmAvailable) ? "" : "none";
  }

  function applyProcessDefaults() {
    fetch("/api/process/defaults")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        document.getElementById("process-deep-checkbox").checked = !!data.deep;
        // F161: `vlm.products` defaults to true, so this box starts ticked and the
        // screen opens describing the run the config file has always described.
        document.getElementById("process-products-checkbox").checked = !!data.products;
        document.getElementById("process-geo-online-checkbox").checked = !!data.geo_online;
        document.getElementById("process-pets-checkbox").checked = !!data.pets;
        // F138: the one that moved here out of the settings column starts from the
        // config exactly as deep/pets do — the file is where it lives, this screen is
        // where one run overrides it. (F186 retired the other three of that set.)
        document.getElementById("process-pets-verify-checkbox").checked = !!data.pets_verify;
        vlmAvailable = !!data.vlm_available;
        updateVlmMissingWarning();
        renderCosts();
        updateStepLayout();  // сводка блока «Параметры запуска» — по фактическим галочкам
      })
      .catch(function () {});
  }

  // --- F138: the run budget -----------------------------------------------
  //
  // The prices come from the server ONCE (they depend on the collection, not on the
  // checkboxes) and the sum is recomputed here on every click. Asking the server per
  // click would put a request between a person and a toggle they are still deciding
  // about — and there is nothing to ask: switching a box does not change what the
  // index holds. A run does, so the estimate is re-fetched after one.
  //
  // A missing price is null, and null is a DASH, never a zero: a zero reads as "free",
  // and this screen may not promise twenty minutes with two hours coming. The same rule
  // carries into the sum — an unknown line makes it a floor ("at least"), not a total.
  //
  // F159: `sources` travels with the seconds and says, per line, whether the rate behind
  // it was read out of this machine's own run log or is the default shipped with the
  // tool. The note under the block reports that over the lines actually SWITCHED ON — it
  // describes the total standing above the button, and a caveat about a stage nobody
  // asked for would be a caveat about nothing.
  var costEstimate = null;
  var costSources = null;
  var costMeasuredAt = null;
  //
  // F161: `master` is the row that grants permission and does nothing else. It is not a
  // `vlm` row — those are the ones it switches off — and its price is not a number from
  // the server either: it is zero by construction, in both directions of every checkbox
  // on this screen.
  var COST_ROWS = [
    { key: "base", always: true },
    { key: "faces", id: "process-faces-checkbox" },
    { key: "events", id: "process-events-checkbox" },
    { key: "pets", id: "process-pets-checkbox" },
    { key: "pets_verify", id: "process-pets-verify-checkbox",
      parent: "process-pets-checkbox", vlm: true },
    { key: "deep", id: "process-deep-checkbox", master: true },
    { key: "products", id: "process-products-checkbox", vlm: true }
  ];

  // --- F145: "Deep analysis (VLM)" is the master switch ----------------------
  //
  // The three lines marked `vlm` above ask the SAME weights this checkbox loads, and
  // until F145 each of them could raise those weights by itself — a run started without
  // the checkbox still spent 20 GB and hours because one key was true in config.yaml.
  // The server now refuses to load a model without it (config.vlm_allowed), and this
  // screen has to say the same thing before the run rather than after it:
  //
  //   * the options stay VISIBLE and go dead. A vanished option reads as "there is no
  //     such thing", and there is;
  //   * their price becomes zero, not the old number. The estimate has to add up to what
  //     the run will actually do;
  //   * nothing is switched on or off automatically. Clearing the master leaves the
  //     subordinate boxes exactly as they were — one movement, one consequence.
  //
  // F161: the deep junk tier joined that list. It used to BE the master switch's own
  // effect, which is why it was not here — a line nobody could see could not be marked
  // as subordinate to anything.
  var VLM_SUBORDINATE_IDS = ["process-products-checkbox",
                             "process-pets-verify-checkbox"];

  function vlmMasterOn() {
    return document.getElementById("process-deep-checkbox").checked;
  }

  function updateVlmSubordinatesDisabled() {
    var off = !vlmMasterOn();
    VLM_SUBORDINATE_IDS.forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.disabled = off || processRunning; }
    });
    document.querySelectorAll(".vlm-off-hint").forEach(function (el) {
      el.style.display = off ? "" : "none";
    });
  }

  // The seconds behind one line, or null when this index cannot say.
  function costSeconds(row) {
    // F161: the master switch grants permission and runs nothing, so its own price is
    // zero on a collection this index knows nothing else about too — a dash there would
    // turn the whole sum into "at least" over a line that costs nothing.
    if (row.master) return 0;
    // F145: a subordinate line costs nothing with the master off, whatever the box next
    // to it says — that IS the run, and a dash here would mean "unknown" rather than
    // "free".
    if (row.vlm && !vlmMasterOn()) return 0;
    if (!costEstimate) return null;
    var value = costEstimate[row.key];
    return (typeof value === "number") ? value : null;
  }

  // "measured", "default" or "fixed" — see the server payload. Null when this index has
  // not answered yet, and for a line the master switch has turned off: a line that does
  // not run has no rate to have a pedigree.
  function costSource(row) {
    if (!costSources) return null;
    if (row.vlm && !vlmMasterOn()) return null;
    return costSources[row.key] || null;
  }

  function formatCost(seconds) {
    if (seconds === null) return I18N.costs_unknown;
    if (seconds <= 0) return I18N.costs_free;
    if (seconds < 60) return I18N.costs_under_minute;
    var minutes = Math.round(seconds / 60);
    if (minutes < 60) return fmt(I18N.costs_minutes, { minutes: minutes });
    return fmt(I18N.costs_hours,
               { hours: Math.floor(minutes / 60), minutes: minutes % 60 });
  }

  function costRowEnabled(row) {
    if (row.always) return true;
    if (row.parent && !document.getElementById(row.parent).checked) return false;
    return document.getElementById(row.id).checked;
  }

  function renderCosts() {
    // The subordinate control exists only while its parent is on: a check of a label
    // nobody is computing is a choice about nothing.
    var petsOn = document.getElementById("process-pets-checkbox").checked;
    document.getElementById("process-pets-verify-row").style.display = petsOn ? "" : "none";
    updateVlmSubordinatesDisabled();
    var total = 0;
    var unknown = false;
    var measured = false;
    var byDefault = false;
    var vlmOff = !vlmMasterOn();
    COST_ROWS.forEach(function (row) {
      var seconds = costSeconds(row);
      var cell = document.querySelector('[data-cost="' + row.key + '"]');
      // F145: a line the master switch has turned off is priced at zero and says why —
      // "almost free" is what a stage that RUNS and is cheap gets, and this one does not
      // run at all.
      if (cell) {
        // F161: and the master switch says the other zero — the one that means "this
        // line has no work of its own", not "this line will not run".
        cell.textContent = row.master ? I18N.costs_permission_only
            : (row.vlm && vlmOff) ? I18N.costs_off : formatCost(seconds);
      }
      if (!costRowEnabled(row)) return;
      if (seconds === null) { unknown = true; return; }
      total += seconds;
      var source = costSource(row);
      if (source === "measured") measured = true;
      else if (source === "default") byDefault = true;
    });
    var value = document.getElementById("process-budget-value");
    if (unknown && total <= 0) value.textContent = I18N.costs_unknown;
    else if (unknown) {
      value.textContent = fmt(I18N.costs_total_at_least, { time: formatCost(total) });
    } else value.textContent = formatCost(total);
    renderCostSource(measured, byDefault);
  }

  function renderCostSource(measured, byDefault) {
    var note = document.getElementById("process-costs-source");
    var when = costMeasuredAt || "";
    if (measured && byDefault) {
      note.textContent = fmt(I18N.costs_source_mixed, { date: when });
    } else if (measured) {
      note.textContent = fmt(I18N.costs_source_measured, { date: when });
    } else if (byDefault) {
      note.textContent = I18N.costs_source_default;
    } else note.textContent = "";  // nothing priced yet — nothing to have a pedigree
  }

  function loadCostEstimate() {
    fetch("/api/process/estimate")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        costEstimate = (data && data.seconds) || null;
        costSources = (data && data.sources) || null;
        costMeasuredAt = (data && data.measured_at) || null;
        renderCosts();
      })
      .catch(function () { renderCosts(); });
  }

  applyProcessDefaults();
  loadCostEstimate();
  document.getElementById("process-deep-checkbox")
      .addEventListener("change", updateVlmMissingWarning);
  ["process-faces-checkbox", "process-events-checkbox", "process-pets-checkbox",
   "process-pets-verify-checkbox", "process-deep-checkbox",
   "process-products-checkbox"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", renderCosts);
  });
  // Draw once before either answer arrives: dashes and the right nested rows, rather
  // than a block of blank price slots for as long as the two requests take.
  renderCosts();

  // F64: баннер о CPU-профиле (обработка на процессоре — медленно для лиц/VLM).
  fetch("/api/env").then(function (r) { return r.json(); })
    .then(function (data) {
      if (data && !data.gpu_profile) {
        document.getElementById("env-cpu-warning").style.display = "";
      }
    }).catch(function () {});

  var PROCESS_POLL_MS = 1500;
  var processPollTimer = null;

  function processStageLabel(stage) {
    return stage ? (I18N["process_stage_" + stage] || stage) : "";
  }

  // Чипы-этапы (F41): done/now/pending по стадиям пайплайна — тот же порядок,
  // что и сервер (_PIPELINE_STAGE_NAMES), только для отображения. F53/#39:
  // faces/events opt-in — currentProcessStages фиксируется по чекбоксам в
  // момент запуска (сервер фильтрует steps так же), иначе индексы чипов
  // разъедутся со stage_index отфильтрованного прогона.
  var ALL_PROCESS_STAGES = ["index", "geo", "landmarks", "classify", "faces", "events",
                            "junk", "phash"];
  var OPTIONAL_PROCESS_STAGES = { faces: true, events: true };
  var currentProcessStages = ALL_PROCESS_STAGES.slice();

  function filterProcessStages(faces, events) {
    var enabled = { faces: faces, events: events };
    return ALL_PROCESS_STAGES.filter(function (name) {
      return !OPTIONAL_PROCESS_STAGES[name] || enabled[name];
    });
  }

  // F135: there is no "Re-run selected" any more — one run button, and the stages
  // skip what is already done by themselves. The /api/process/rerun-optional ROUTE is
  // still there (it is public, see the API documentation): the button went, not it.

  // The last known state of the pipeline. The status poll runs once a tick while the
  // handlers on this tab fire instantly — without this flag they used to re-enable
  // what the tick had just disabled for the duration of a run.
  var processRunning = false;

  // Всё, что задаёт вход пайплайна, на время прогона недоступно: менять источник
  // у уже идущей обработки бессмысленно, а диалог выбора папки ещё и открывает
  // отдельное окно поверх работающего процесса. Галки шагов и ярусов — там же:
  // параметры уходят на сервер один раз, в момент старта, поэтому снятая на
  // середине прогона галка «лица» ничего не отменяет, а выглядит так, будто
  // отменила — и это выясняется через час, когда лица всё-таки посчитались.
  function updateProcessInputsDisabled() {
    ["process-browse-btn", "process-source-dir", "process-excludes-btn",
     "process-deep-checkbox", "process-geo-online-checkbox",
     "process-faces-checkbox", "process-events-checkbox",
     "process-pets-checkbox"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.disabled = processRunning; }
    });
    // F145: the options under the master switch have two reasons to be dead and one
    // place that applies both — listing them here as well would re-enable, on the next
    // status tick, boxes the cleared checkbox had just switched off.
    updateVlmSubordinatesDisabled();
  }

  function renderStageChips(data) {
    var container = document.getElementById("process-stages");
    container.textContent = "";
    if (!data.running && !data.finished) return;
    var success = !data.running && data.finished && !data.error;
    currentProcessStages.forEach(function (name, idx) {
      var stepIndex = idx + 1;
      var cls = "pending";
      if (success || stepIndex < data.stage_index) cls = "done";
      else if (data.running && stepIndex === data.stage_index) cls = "now";
      var chip = document.createElement("span");
      chip.className = "stage-chip " + cls;
      if (cls === "done") chip.appendChild(icon("check"));
      chip.appendChild(document.createTextNode(processStageLabel(name)));
      container.appendChild(chip);
    });
  }

  // F84: a stage can name the phase it is in (clustering inside "faces"); an empty
  // phase means the stage reports none — then nothing is drawn and the screen looks
  // exactly as it did before. On an unmeasurable phase (total is unknown, HDBSCAN is
  // one blocking call) there is no honest percent to show, so the caption carries a
  // stopwatch instead: an invented percent would discredit the bar for good.
  function renderProcessPhase(data) {
    var el = document.getElementById("process-phase");
    var key = data.running && !data.cancel_requested ? data.phase : null;
    var label = key ? (I18N["process_phase_" + key] || key) : "";
    if (!label) { el.textContent = ""; el.style.display = "none"; return; }
    el.textContent = data.total > 0 ? label : fmt(I18N.process_phase_elapsed, {
      phase: label,
      seconds: Math.round(data.phase_elapsed || 0),
    });
    el.style.display = "";
  }

  function refreshTabsAfterProcess() {
    dupesLoaded = false;
    reviewLoaded = false;  // F126: a run recomputes every signal the slices are built on
    clustersLoaded = false;
    eventsLoaded = false;
    movesLoaded = false;
    junkLoaded = false;  // F103: прогон junk-яруса меняет состав корзин
    animalsLoaded = false;  // F123: the same run recomputes the animal verdicts
    faceLoaded = false;     // F152: a faces run is what turns the reason into numbers
    refreshPlan();
    applyTabVisibility();
    loadCacheSizes();  // F94: a run is what makes the preview cache grow
    // F138: a run is also what makes the estimate knowable — the deep tier's candidate
    // count, the pet scores and the near-duplicate groups all come out of it. The
    // dashes of a fresh collection turn into numbers here and nowhere else.
    loadCostEstimate();
    // F133: a run recomputes what the order warning is about; refresh it where it is
    // shown rather than waiting for the next open of the tab.
    if (document.getElementById("tab-layout").classList.contains("active")) {
      loadLayoutWarning();
    }
    if (document.getElementById("tab-slices").classList.contains("active")) {
      loadSlices();
    }
    // F108: обзор перечитывается при каждом открытии, но если человек смотрит на
    // него прямо сейчас — обновляем немедленно: прогон только что изменил числа.
    if (document.getElementById("tab-overview").classList.contains("active")) {
      loadOverview();
    }
  }

  // F135: the source of the last run comes back into the field by itself. The
  // browser's own memory (SOURCE_DIR_KEY) covers a page reload but not a fresh profile
  // or a second browser — and "Start" in one click is half of what merging the two
  // buttons is for. A field that already has something in it is left alone: putting a
  // path over what someone is typing is worse than not restoring it at all.
  function adoptRememberedSource(data) {
    var input = document.getElementById("process-source-dir");
    if (!data.source_dir || input.value.trim()) return;
    input.value = data.source_dir;
    rememberSourceDir();
    loadExcludesInfo();
    updateStepLayout();
  }

  // F135: with one button the run walks the whole pipeline every time, and without
  // this summary "everything was already done" reads exactly like "nothing happened".
  // It shows what the CLI prints: how much the stage processed and how much it skipped
  // as already processed. Stages without such a counter are not sent by the server and
  // do not appear here — an invented zero would be a lie, not a line of a report.
  function renderProcessSummary(data) {
    var box = document.getElementById("process-summary");
    box.textContent = "";
    var stats = data.stage_stats || {};
    var names = ALL_PROCESS_STAGES.filter(function (name) { return stats[name]; });
    if (!names.length) return;
    var title = document.createElement("span");
    title.className = "process-summary-title";
    title.textContent = I18N.process_summary_title;
    box.appendChild(title);
    names.forEach(function (name) {
      var line = document.createElement("span");
      line.className = "process-summary-line";
      line.textContent = fmt(I18N.process_summary_stage, {
        stage: processStageLabel(name),
        processed: stats[name].processed || 0,
        skipped: stats[name].skipped || 0,
      });
      box.appendChild(line);
    });
  }

  function renderProcessStatus(data) {
    var startBtn = document.getElementById("process-start-btn");
    var cancelBtn = document.getElementById("process-cancel-btn");
    var bar = document.getElementById("process-progress");
    var statusEl = document.getElementById("process-status");
    processRunning = !!data.running;
    startBtn.disabled = processRunning;
    updateProcessInputsDisabled();
    updateBusyControlsDisabled();  // раскладка и «начать заново» — пока идёт прогон
    adoptRememberedSource(data);
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    bar.style.display = data.running ? "" : "none";
    if (!data.running) bar.classList.remove("indeterminate");
    renderStageChips(data);
    renderProcessPhase(data);
    renderProcessSummary(data);
    if (data.running) {
      if (data.cancel_requested) {
        // отмена запрошена — показываем фидбэк, пока стадия прерывается/дорабатывает
        bar.classList.add("indeterminate");
        bar.max = 1;
        bar.removeAttribute("value");
        statusEl.textContent = I18N.process_cancel_requested;
        return;
      }
      // #37: total>0 -> определённый прогресс (заполняется); total<=0 (индексация,
      // total неизвестен) -> бегущая indeterminate-полоса + «обработано X».
      if (data.total > 0) {
        bar.classList.remove("indeterminate");
        bar.max = data.total;
        bar.value = data.done || 0;
      } else {
        bar.classList.add("indeterminate");
        bar.max = 1;
        bar.removeAttribute("value");
      }
      statusEl.textContent = fmt(
        data.total > 0 ? I18N.process_stage_progress : I18N.process_stage_progress_indeterminate, {
        stage: processStageLabel(data.stage),
        index: data.stage_index,
        total: data.stage_total,
        done: data.done,
        all: data.total,
      });
      return;
    }
    if (!data.finished) {
      statusEl.textContent = "";
      return;
    }
    if (data.error) {
      statusEl.textContent = I18N.process_error_prefix + data.error;
    } else if (data.cancel_requested) {
      statusEl.textContent = I18N.process_cancelled;
      refreshTabsAfterProcess();
    } else {
      statusEl.textContent = I18N.process_done;
      refreshTabsAfterProcess();
    }
  }

  function pollProcessStatus() {
    fetch("/api/process/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderProcessStatus(data);
        if (data.running) {
          processPollTimer = setTimeout(pollProcessStatus, PROCESS_POLL_MS);
        }
      });
  }

  document.getElementById("process-start-btn").addEventListener("click", function () {
    var input = document.getElementById("process-source-dir");
    var path = input.value.trim();
    if (!path) { window.alert(I18N.process_enter_path); return; }
    // запуск = «настроено»: оба блока схлопываются, экран остаётся про прогресс
    stepSourceOpen = false;
    stepOptionsOpen = false;
    rememberSourceDir();
    updateStepLayout();
    var deep = document.getElementById("process-deep-checkbox").checked;
    var geoOnline = document.getElementById("process-geo-online-checkbox").checked;
    var faces = document.getElementById("process-faces-checkbox").checked;
    var events = document.getElementById("process-events-checkbox").checked;
    // F123: pets does NOT go into filterProcessStages — it is a setting of the junk
    // stage, not a stage, and the chip row has to show the run that will actually happen.
    var pets = document.getElementById("process-pets-checkbox").checked;
    // F138: three more settings of that same junk stage, and the scope of the quality
    // question. All four are sent EXPLICITLY, ticked or not, so an unticked box forces
    // OFF what config.yaml switched on (the F57 rule) instead of quietly deferring to
    // the file. `pets_verify` needs `pets` — the row is hidden without it — so it is
    // sent as false rather than as a check the junk stage would refuse anyway.
    var petsVerify = pets && document.getElementById("process-pets-verify-checkbox").checked;
    currentProcessStages = filterProcessStages(faces, events);
    postJson("/api/process", {
      source_dir: path, deep: deep, geo_online: geoOnline, faces: faces, events: events,
      pets: pets, pets_verify: petsVerify,
      // F161: sent explicitly like the four above — an unticked box has to force the
      // deep tier OFF for this run, which is the whole point of giving it a line.
      products: document.getElementById("process-products-checkbox").checked,
    }).then(function (resp) {
      if (resp && resp.error) {
        document.getElementById("process-status").textContent =
            I18N.process_start_error_prefix + resp.error;
        return;
      }
      if (processPollTimer) clearTimeout(processPollTimer);
      pollProcessStatus();
    });
  });

  // Диалог появляется через секунду-две, а кнопка всё это время оставалась
  // активной — каждый лишний клик открывал ещё один проводник. Блокируем на время
  // запроса; сервер тоже отказывает во втором диалоге (см. _browse_for_folder),
  // потому что вкладку можно открыть и вторую.
  function browseIntoField(btn, apply) {
    if (btn.disabled) { return; }
    btn.disabled = true;
    postJson("/api/browse", {})
      .then(function (resp) { if (resp && resp.path) { apply(resp.path); } })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  }

  document.getElementById("process-browse-btn").addEventListener("click", function () {
    browseIntoField(this, function (path) {
      document.getElementById("process-source-dir").value = path;
      sourceDirChanged();
    });
  });

  // --- F81: «не сканировать» + три блока первой вкладки ------------------

  // Путь помнится между открытиями страницы: этот экран открывают многократно, и
  // вводить один и тот же источник каждый раз — ровно тот штраф, который фича
  // убирает.
  var SOURCE_DIR_KEY = "sorta.sourceDir";
  // Что сейчас исключено под текущим источником — для схлопнутой строки блока
  // «Источник». Два числа хранятся раздельно: «не сканировать» и «не раскладывать» —
  // разные вещи. root пустой = про этот источник ещё не спрашивали.
  var excludesInfo = { root: "", scan: [], count: 0, size: 0,
                       layout: [], layoutCount: 0 };
  var stepSourceOpen = false;
  var stepOptionsOpen = false;

  function currentSourceDir() {
    return document.getElementById("process-source-dir").value.trim();
  }

  function formatSize(bytes) {
    var units = I18N.size_units.split(" ");
    var value = bytes || 0;
    var i = 0;
    while (value >= 1024 && i < units.length - 1) { value = value / 1024; i += 1; }
    return value.toFixed(i === 0 || value >= 100 ? 0 : 1) + " " + units[i];
  }

  function excludesSummaryText() {
    if (excludesInfo.root !== currentSourceDir()) return I18N.excludes_summary_none;
    var parts = [];
    if (excludesInfo.count) {
      parts.push(fmt(I18N.excludes_summary,
                     { count: excludesInfo.count, size: formatSize(excludesInfo.size) }));
    }
    if (excludesInfo.layoutCount) {
      parts.push(fmt(I18N.excludes_summary_layout, { count: excludesInfo.layoutCount }));
    }
    return parts.length ? parts.join(" · ") : I18N.excludes_summary_none;
  }

  function optionsSummaryText() {
    var on = [];
    [["process-deep-checkbox", I18N.process_deep_label],
     ["process-products-checkbox", I18N.process_products_label],
     ["process-geo-online-checkbox", I18N.process_geo_online_label],
     ["process-faces-checkbox", I18N.process_faces_label],
     ["process-events-checkbox", I18N.process_events_label],
     ["process-pets-checkbox", I18N.process_pets_label],
     ["process-pets-verify-checkbox", I18N.process_pets_verify_label]
    ].forEach(function (pair) {
      if (document.getElementById(pair[0]).checked) on.push(pair[1]);
    });
    return I18N.step_options_summary_prefix +
        (on.length ? on.join(", ") : I18N.step_options_summary_default);
  }

  // Настроенный блок — одна строка с «изменить», ненастроенный раскрыт. Следующие
  // блоки приглушены пояснением, но НЕ заблокированы: кнопка запуска доступна
  // всегда, когда источник задан (визард штрафует каждый следующий заход).
  // Кнопка шага — переключатель, а не «открыть»: открыл, посмотрел, ничего не менял
  // — и складываешь обратно тем же местом, куда нажал. Сворачивание чисто визуальное
  // и НИЧЕГО не отменяет: введённый путь и снятые галки остаются как есть (иначе это
  // была бы «отмена», а она в шаге, который применяется сразу, только путает).
  // Сворачивать нечего, пока источник не задан, — там кнопка скрыта.
  function updateStepToggle(stepId, buttonId, open, canCollapse) {
    var step = document.getElementById(stepId);
    var button = document.getElementById(buttonId);
    step.classList.toggle("collapsed", canCollapse && !open);
    step.classList.toggle("can-collapse", canCollapse);
    button.textContent = open ? I18N.step_collapse_button : I18N.step_change_button;
    button.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function updateStepLayout() {
    var src = currentSourceDir();
    document.getElementById("step-source-summary").textContent =
        src + " · " + excludesSummaryText();
    document.getElementById("step-options-summary").textContent = optionsSummaryText();
    updateStepToggle("step-source", "step-source-edit", stepSourceOpen, !!src);
    updateStepToggle("step-options", "step-options-edit", stepOptionsOpen, !!src);
    var options = document.getElementById("step-options");
    options.classList.toggle("step-dimmed", !src);
    document.getElementById("step-actions").classList.toggle("step-dimmed", !src);
  }

  function rememberSourceDir() {
    try { window.localStorage.setItem(SOURCE_DIR_KEY, currentSourceDir()); } catch (e) {}
  }

  function excludesInfoOf(src, data) {
    return { root: src, scan: data.skip_scan || [], count: data.count || 0,
             size: data.size || 0, layout: data.skip_layout || [],
             layoutCount: data.layout_count || 0 };
  }

  function loadExcludesInfo() {
    var src = currentSourceDir();
    if (!src) {
      excludesInfo = { root: "", scan: [], count: 0, size: 0, layout: [], layoutCount: 0 };
      updateStepLayout();
      return;
    }
    fetch("/api/source-tree/excludes?path=" + encodeURIComponent(src))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.error) return;
        excludesInfo = excludesInfoOf(src, data);
        updateStepLayout();
      })
      .catch(function () {});
  }

  function sourceDirChanged() {
    // Шаг НЕ схлопывается на выборе папки. Исключения относятся к конкретному корню
    // и являются частью этого же шага, поэтому сворачивать его в момент, когда
    // пользователь как раз собирается их отметить, — значит заставлять возвращаться
    // назад через «изменить». Схлопнутым шаг стартует только при загрузке страницы с
    // уже запомненным источником: там правда нечего делать.
    stepSourceOpen = true;
    rememberSourceDir();
    loadExcludesInfo();
    loadSourceTree();
    updateStepLayout();
  }

  // F82: три состояния узла — "" обрабатывать, "layout" не раскладывать, "scan" не
  // сканировать. Одно поле на узел, поэтому «отмечено и то и другое» невозможно по
  // построению: переключение на одно автоматически снимает другое.
  var TRI_STATES = ["", "layout", "scan"];

  function triText(state) {
    if (state === "scan") return "☒ " + I18N.tri_scan_label;
    if (state === "layout") return "◐ " + I18N.tri_layout_label;
    return "☐";
  }

  function triHint(state) {
    if (state === "scan") return I18N.tri_scan_hint;
    if (state === "layout") return I18N.tri_layout_hint;
    return I18N.tri_none_hint;
  }

  function setTriState(btn, state) {
    btn.setAttribute("data-state", state);
    btn.textContent = triText(state);
    btn.title = triHint(state);
  }

  function setSubtreeState(ul, state) {
    var marks = ul.querySelectorAll("button.tri-state");
    for (var i = 0; i < marks.length; i++) {
      setTriState(marks[i], state);
      // исключённое поддерево не редактируется по частям: состояние родителя — состояние всего
      marks[i].disabled = !!state;
    }
  }

  function renderExcludesNode(node, states, parentState) {
    var li = document.createElement("li");
    var row = document.createElement("div");
    row.className = "excludes-row";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tri-state";
    btn.setAttribute("data-rel", node.rel);
    setTriState(btn, parentState || states[node.rel] || "");
    btn.disabled = !!parentState;
    row.appendChild(btn);
    row.appendChild(document.createTextNode(node.name));
    var meta = document.createElement("span");
    meta.className = "excludes-meta";
    meta.textContent = fmt(I18N.excludes_folder_meta,
                           { count: node.files, size: formatSize(node.size) });
    row.appendChild(meta);
    li.appendChild(row);
    var ul = null;
    if (node.children && node.children.length) {
      ul = document.createElement("ul");
      node.children.forEach(function (child) {
        ul.appendChild(renderExcludesNode(child, states, btn.getAttribute("data-state")));
      });
      li.appendChild(ul);
    }
    btn.addEventListener("click", function () {
      var next = TRI_STATES[(TRI_STATES.indexOf(btn.getAttribute("data-state")) + 1)
                            % TRI_STATES.length];
      setTriState(btn, next);
      if (ul) setSubtreeState(ul, next);
    });
    return li;
  }

  function renderExcludesTree(data) {
    var container = document.getElementById("excludes-tree");
    container.textContent = "";
    var states = {};
    (data.skip_layout || []).forEach(function (rel) { states[rel] = "layout"; });
    // «не сканировать» пишется вторым: при странном файле, где папка попала в оба
    // раздела, сервер уже решил в пользу scan — дерево не должно спорить с ним
    (data.skip_scan || []).forEach(function (rel) { states[rel] = "scan"; });
    var children = (data.tree && data.tree.children) || [];
    if (!children.length) {
      container.appendChild(stateEl("empty", I18N.excludes_empty));
      return;
    }
    var ul = document.createElement("ul");
    children.forEach(function (child) {
      ul.appendChild(renderExcludesNode(child, states, ""));
    });
    container.appendChild(ul);
    if (data.truncated) {
      // ответ ограничен — говорим об этом прямо, а не молча показываем часть дерева
      var note = document.createElement("p");
      note.className = "process-toggle-hint";
      note.textContent = fmt(I18N.excludes_truncated, { limit: data.limit });
      container.appendChild(note);
    }
  }

  function collectExcludes() {
    // только верхние отмеченные: потомки отмеченной папки заблокированы и не нужны
    var result = { skip_scan: [], skip_layout: [] };
    var marks = document.getElementById("excludes-tree")
        .querySelectorAll("button.tri-state");
    for (var i = 0; i < marks.length; i++) {
      if (marks[i].disabled) continue;
      var state = marks[i].getAttribute("data-state");
      if (state === "scan") result.skip_scan.push(marks[i].getAttribute("data-rel"));
      else if (state === "layout") result.skip_layout.push(marks[i].getAttribute("data-rel"));
    }
    return result;
  }

  // Вынесено из обработчика кнопки: дерево показывается и по кнопке, и сразу после
  // выбора папки — «выбрал источник, вижу его структуру» это один шаг, а не два.
  function loadSourceTree(announce) {
    var src = currentSourceDir();
    if (!src) { if (announce) { window.alert(I18N.process_enter_path); } return; }
    document.getElementById("excludes-panel").style.display = "";
    document.getElementById("excludes-status").textContent = "";
    var container = document.getElementById("excludes-tree");
    container.textContent = "";
    container.appendChild(stateEl("loading", I18N.loading));
    fetch("/api/source-tree?path=" + encodeURIComponent(src))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.error) {
          container.textContent = "";
          container.appendChild(stateEl(
              "error", I18N.excludes_error_prefix + ((data && data.error) || "")));
          return;
        }
        renderExcludesTree(data);
      })
      .catch(function (e) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.excludes_error_prefix + e));
      });
  }

  document.getElementById("process-excludes-btn").addEventListener("click", function () {
    loadSourceTree(true);
  });

  document.getElementById("excludes-save-btn").addEventListener("click", function () {
    var src = currentSourceDir();
    if (!src) return;
    var statusEl = document.getElementById("excludes-status");
    var picked = collectExcludes();
    postJson("/api/source-tree/excludes",
             { root: src, skip_scan: picked.skip_scan, skip_layout: picked.skip_layout })
      .then(function (resp) {
        if (!resp || resp.error) {
          statusEl.textContent =
              I18N.excludes_save_error_prefix + ((resp && resp.error) || "");
          return;
        }
        excludesInfo = excludesInfoOf(src, resp);
        statusEl.textContent = I18N.excludes_saved;
        updateStepLayout();
      })
      .catch(function (e) { statusEl.textContent = I18N.excludes_save_error_prefix + e; });
  });

  document.getElementById("excludes-close-btn").addEventListener("click", function () {
    document.getElementById("excludes-panel").style.display = "none";
  });

  document.getElementById("step-source-edit").addEventListener("click", function () {
    stepSourceOpen = !stepSourceOpen;
    // Свернули источник — панель исключений уходит вместе с ним: она часть этого
    // шага и висеть отдельно от него не должна.
    if (!stepSourceOpen) {
      document.getElementById("excludes-panel").style.display = "none";
    }
    updateStepLayout();
  });

  document.getElementById("step-options-edit").addEventListener("click", function () {
    stepOptionsOpen = !stepOptionsOpen;
    updateStepLayout();
  });

  document.getElementById("process-source-dir")
      .addEventListener("input", updateStepLayout);
  document.getElementById("process-source-dir")
      .addEventListener("change", sourceDirChanged);
  ["process-deep-checkbox", "process-products-checkbox",
   "process-geo-online-checkbox", "process-faces-checkbox",
   "process-events-checkbox", "process-pets-checkbox",
   "process-pets-verify-checkbox"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", updateStepLayout);
  });

  (function restoreSourceDir() {
    var input = document.getElementById("process-source-dir");
    if (!input.value.trim()) {
      var saved = null;
      try { saved = window.localStorage.getItem(SOURCE_DIR_KEY); } catch (e) { saved = null; }
      if (saved) input.value = saved;
    }
    loadExcludesInfo();
    updateStepLayout();
  })();

  document.getElementById("process-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("process-status").textContent = I18N.process_cancel_requested;
    renderProcessPhase({});  // the phase caption is stale now, do not wait for a tick
    postJson("/api/process/cancel", {});
  });

  // F93: сброс подтверждается своим диалогом, а не window.confirm — в нём живёт
  // галочка «также очистить кэш геоданных». Галочка каждый раз сбрасывается: очистка
  // кэша — разовое решение, а не режим, который тихо остаётся включённым.
  var resetDialogEl = document.getElementById("reset-dialog");
  var resetClearGeoEl = document.getElementById("reset-clear-geo-checkbox");

  function closeResetDialog() {
    resetDialogEl.hidden = true;
  }

  document.getElementById("process-reset-btn").addEventListener("click", function () {
    resetClearGeoEl.checked = false;
    resetDialogEl.hidden = false;
  });

  document.getElementById("reset-dialog-cancel").addEventListener("click", closeResetDialog);

  resetDialogEl.addEventListener("click", function (e) {
    if (e.target === resetDialogEl) closeResetDialog();  // клик по фону — отмена
  });

  document.getElementById("reset-dialog-ok").addEventListener("click", function () {
    var clearGeo = resetClearGeoEl.checked;
    closeResetDialog();
    postJson("/api/process/reset", { clear_geo: clearGeo }).then(function (resp) {
      var statusEl = document.getElementById("process-status");
      if (resp && resp.error) {
        statusEl.textContent = I18N.process_reset_error_prefix + resp.error;
        return;
      }
      statusEl.textContent = clearGeo ? I18N.process_reset_done_geo : I18N.process_reset_done;
      // F135: the summary of the last run counted files of an index that is gone now.
      renderProcessSummary({});
      refreshTabsAfterProcess();
    });
  });

  // --- F94: the caches ------------------------------------------------------
  // Sizes are asked for rarely and on purpose: the preview cache is tens of
  // thousands of files, so the status poll must never touch it. Page load, the end
  // of a run and a clear are the only three moments the number can have changed.
  var cacheInfo = { previewBytes: 0, previewFiles: 0, geoEntries: 0 };

  function applyCacheInfo(data) {
    if (!data || !data.preview || !data.geo) return;
    cacheInfo.previewBytes = data.preview.bytes || 0;
    cacheInfo.previewFiles = data.preview.files || 0;
    cacheInfo.previewMaxGb = data.preview.max_gb || 0;
    cacheInfo.geoEntries = data.geo.entries || 0;
    document.getElementById("cache-sizes").textContent = fmt(I18N.cache_sizes, {
      preview: formatSize(cacheInfo.previewBytes),
      files: cacheInfo.previewFiles,
      geo: cacheInfo.geoEntries,
    });
    // F117: 0 is "no ceiling", a state — not a limit of zero, which would read as a
    // cache that may hold nothing at all.
    var limitEl = document.getElementById("cache-limit");
    if (cacheInfo.previewMaxGb > 0) {
      var used = cacheInfo.previewBytes / (cacheInfo.previewMaxGb * 1e9) * 100;
      limitEl.textContent = fmt(I18N.cache_limit, {
        limit: cacheInfo.previewMaxGb,
        percent: Math.round(used),
      });
    } else {
      limitEl.textContent = I18N.cache_no_limit;
    }
  }

  var cacheSizesPending = false;

  function loadCacheSizes() {
    // Page load and "the run has finished" can land together (a reload right after a
    // run) — one walk of the preview directory per moment, not two.
    if (cacheSizesPending) return;
    cacheSizesPending = true;
    fetch("/api/cache").then(function (r) { return r.json(); })
      .then(function (data) { cacheSizesPending = false; applyCacheInfo(data); })
      .catch(function () { cacheSizesPending = false; });
  }

  // Both clears are irreversible and neither is free, so each states its own price
  // before it happens — the preview one that the next run pays 336 ms per frame
  // instead of 73, the geo one that with provider: online it pays the network again.
  function clearCache(target, confirmText, doneText) {
    if (!window.confirm(confirmText)) return;
    var statusEl = document.getElementById("cache-status");
    statusEl.textContent = "";
    postJson("/api/cache/clear", { target: target }).then(function (resp) {
      if (!resp || resp.error) {
        statusEl.textContent =
            I18N.cache_clear_error_prefix + ((resp && resp.error) || "");
        return;
      }
      statusEl.textContent = fmt(doneText, { n: resp.removed || 0 });
      applyCacheInfo(resp.cache);
    });
  }

  document.getElementById("cache-clear-preview-btn").addEventListener("click", function () {
    clearCache("preview",
               fmt(I18N.cache_clear_preview_confirm,
                   { preview: formatSize(cacheInfo.previewBytes) }),
               I18N.cache_clear_preview_done);
  });

  document.getElementById("cache-clear-geo-btn").addEventListener("click", function () {
    clearCache("geo",
               fmt(I18N.cache_clear_geo_confirm, { geo: cacheInfo.geoEntries }),
               I18N.cache_clear_geo_done);
  });

  loadCacheSizes();

  pollProcessStatus();

  // --- вкладка «Города»: apply раскладки (F43) ----------------------------
  // Дерево-превью вкладки — уже dry-run; кнопка сразу открывает подтверждение
  // (текст зависит от режима/dest), только потом POST /api/sort. Фон +
  // прогресс — тот же паттерн polling, что и «Обработать» (F36) выше.

  var SORT_POLL_MS = 1500;
  var sortPollTimer = null;

  function updateSortApplyBtnStyle() {
    var btn = document.getElementById("sort-apply-btn");
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var move = !checked || checked.value === "move";
    btn.classList.toggle("btn-danger", move);
    btn.classList.toggle("btn-primary", !move);
  }

  document.querySelectorAll('input[name="sort-mode"]').forEach(function (r) {
    r.addEventListener("change", updateSortApplyBtnStyle);
  });
  updateSortApplyBtnStyle();

  // F104: before a layout the user sees NUMBERS, not a question "are you sure?". They
  // come from /api/sort/summary — the same built plan the tab's tree is drawn from, so
  // the dialog cannot name a figure the tab does not show.
  var sortDialogEl = document.getElementById("sort-dialog");

  function sortSummaryLines(data, dest, mode) {
    var lines = [fmt(I18N.sort_summary_dest,
                     { dest: data.dest || dest || I18N.sort_dest_inplace_label })];
    lines.push(mode === "move" ? I18N.sort_summary_mode_move : I18N.sort_summary_mode_copy);
    lines.push(fmt(I18N.sort_summary_files,
                   { n: data.files, dirs: data.dirs, size: formatSize(data.bytes) }));
    if (data.dest === null) lines.push(I18N.sort_summary_existing_unknown);
    else if (!data.dest_existing) lines.push(I18N.sort_summary_existing_none);
    else lines.push(fmt(I18N.sort_summary_existing,
                        { n: data.dest_existing, same: data.dest_same }));
    if (data.products || data.documents) {
      lines.push(fmt(I18N.sort_summary_service,
                     { products: data.products, documents: data.documents }));
    }
    return lines;
  }

  function openSortDialog(data, dest, mode) {
    document.getElementById("sort-dialog-text").textContent = I18N.sort_confirm_title;
    var list = document.getElementById("sort-dialog-list");
    list.textContent = "";
    sortSummaryLines(data, dest, mode).forEach(function (line) {
      var li = document.createElement("li");
      li.textContent = line;
      list.appendChild(li);
    });
    // A line of its own goes to what the numbers cannot say: an in-place run
    // restructures the SOURCE tree rather than a copy in a separate folder.
    document.getElementById("sort-dialog-warning").textContent =
        dest ? (mode === "move" ? I18N.sort_confirm_move : I18N.sort_confirm_copy)
             : I18N.sort_confirm_inplace;
    sortDialogEl.hidden = false;
  }

  function closeSortDialog() {
    sortDialogEl.hidden = true;
  }

  // Раскладка во время прогона запрещена и на сервере (409 «process is running»
  // под общим busy_lock), но кнопка до этого оставалась живой — про запрет
  // узнавали кликом. Хуже другое: на середине прогона плана попросту нет.
  // geo чистит places перед записью, junk ещё не заполнил media_class — то есть
  // раскладка, начатая сейчас, разложила бы коллекцию по недостроенному индексу.
  var sortRunning = false;

  // Кнопки, которые обязаны быть мертвы, пока занят ЛЮБОЙ из двух процессов
  // (прогон пайплайна или раскладка). Сервер их и так отбивает 409 под общим
  // busy_lock, но «Начать заново» сперва показывает страшное подтверждение и
  // только потом ошибку, а раскладка на середине прогона разложила бы коллекцию
  // по недостроенному индексу (places очищены, media_class ещё пуст).
  // F94: очистки кэшей — там же: подтверждение с ценой действия ради ответа 409
  // ничем не лучше, а превью на середине прогона пишет тот самый шаг.
  // F97: откат — третий такой же процесс: он двигает файлы на диске, совмещать его
  // с раскладкой или прогоном нельзя (сервер отбивает 409 под тем же busy_lock).
  // undoAvailable/undoBatchInfo наполняет манифест «Перемещений» (applyUndoAvailability):
  // кнопка отката жива только когда есть что откатывать, а диалог берёт числа оттуда же.
  var undoRunning = false;
  var undoAvailable = false;
  var undoBatchInfo = null;

  function updateBusyControlsDisabled() {
    var busy = uiBusy();
    ["sort-browse-btn", "sort-dest",
     "process-reset-btn",
     "cache-clear-preview-btn", "cache-clear-geo-btn",
     // F145: saving the whole set of duplicate choices writes `dedup_choice` for every
     // group on the tab at once — the largest single write the review side has.
     "dupes-save-all-btn",
     // F156: pinning writes `features.saved_slices`, and the server refuses a config
     // write mid-run like every other one — so the button says so instead of being
     // found out by clicking. The three controls INSIDE a pinned slice have a rule of
     // their own (the ends of the row) and register `refreshQuerySliceControls` below,
     // where the two rules meet in one place.
     "slice-pin-btn",
     "folder-lang-select"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.disabled = busy; }
    });
    // F145: the settings column. The server has answered 409 here since F104 ("swapping
    // the model mid-classification is not a setting but an accident") — what was missing
    // was the ban being VISIBLE: the fields stayed live, you moved one, and learned about
    // the refusal afterwards.
    SETTING_CONTROLS.forEach(function (control) {
      var el = document.getElementById(control.id);
      if (el) { el.disabled = busy; }
    });
    // Album buttons are built by four different tabs and none of them exists until its
    // tab is drawn, so they are swept by class rather than by id.
    document.querySelectorAll(".album-gather-btn").forEach(function (btn) {
      btn.disabled = busy;
    });
    document.querySelectorAll(".busy-hint").forEach(function (el) {
      el.style.display = busy ? "" : "none";
    });
    busyRefreshers.forEach(function (fn) { fn(); });
    var undoBtn = document.getElementById("undo-btn");
    // «Откатить» дополнительно требует батча в манифесте — см. applyUndoAvailability
    if (undoBtn) { undoBtn.disabled = busy || !undoAvailable; }
    // F104: an empty plan disables the start button and says WHY, instead of opening a
    // dialog full of zeroes. Until the plan has arrived the button is dead too, but
    // silently — "nothing to lay out" and "not counted yet" are different statements.
    var applyBtn = document.getElementById("sort-apply-btn");
    if (applyBtn) { applyBtn.disabled = busy || planCount === 0; }
    var emptyHint = document.getElementById("sort-empty-hint");
    if (emptyHint) {
      emptyHint.style.display = (planLoaded && planCount === 0) ? "" : "none";
    }
  }

  function renderSortStatus(data) {
    var bar = document.getElementById("sort-progress");
    var statusEl = document.getElementById("sort-status");
    var warnEl = document.getElementById("sort-warning");
    var cancelBtn = document.getElementById("sort-cancel-btn");
    sortRunning = !!data.running;
    // F104: "Cancel" is a contextual button — it exists exactly while a layout runs.
    // A permanent cancel button next to the start button cancels nothing.
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    updateBusyControlsDisabled();
    bar.style.display = data.running ? "" : "none";
    if (data.running) {
      bar.max = data.total || 0;
      bar.value = data.done || 0;
      statusEl.textContent = data.cancel_requested
          ? I18N.sort_cancel_requested
          : fmt(I18N.sort_progress_line, { done: data.done, all: data.total });
      warnEl.textContent = "";
      return;
    }
    if (!data.finished) {
      statusEl.textContent = ""; warnEl.textContent = ""; return;
    }
    if (data.error) {
      statusEl.textContent = I18N.sort_error_prefix + data.error;
      warnEl.textContent = "";
      return;
    }
    var r = data.result || {};
    // F97: отменённый прогон обязан говорить «сколько из скольких», а не «готово».
    // F104: what stayed next to it is the HINT pointing at the "Moves" tab, not a roll
    // back button. The manifest that says WHAT exactly would be rolled back lives
    // there; rolling back from the plan screen is rolling back blind.
    if (r.cancelled) {
      statusEl.textContent = fmt(I18N.sort_cancelled_text,
          { n: r.moved || 0, all: r.total || 0, f: r.failed || 0 });
    } else {
      statusEl.textContent = fmt(I18N.sort_done_text,
          { n: r.moved || 0, f: r.failed || 0, p: r.skipped_in_place || 0 });
    }
    if (r.skipped_already_copied) {
      statusEl.textContent += fmt(I18N.sort_already_copied_note,
          { c: r.skipped_already_copied });
    }
    warnEl.textContent = r.preview_stale ? I18N.sort_preview_stale_warning
        : (r.cancelled ? I18N.sort_undo_hint : "");
    movesLoaded = false;
    refreshUndoAvailability();
    refreshPlan();
  }

  function pollSortStatus() {
    fetch("/api/sort/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderSortStatus(data);
        if (data.running) sortPollTimer = setTimeout(pollSortStatus, SORT_POLL_MS);
      });
  }

  document.getElementById("sort-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("sort-status").textContent = I18N.sort_cancel_requested;
    postJson("/api/sort/cancel", {});
  });

  function startSort() {
    var dest = document.getElementById("sort-dest").value.trim();
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var mode = checked ? checked.value : "move";
    // F192: `by` is the criterion, `mode` is move-or-copy — two different questions
    // that the server has kept apart since F43 and the field names keep apart here.
    postJson("/api/sort", { dest: dest || null, mode: mode, by: layoutBy() })
      .then(function (resp) {
        if (resp && resp.error) {
          document.getElementById("sort-status").textContent =
              I18N.sort_start_error_prefix + resp.error;
          return;
        }
        if (sortPollTimer) clearTimeout(sortPollTimer);
        pollSortStatus();
      });
  }

  document.getElementById("sort-apply-btn").addEventListener("click", function () {
    // An empty plan never gets here (the button is dead, see updateBusyControlsDisabled)
    // — a dialog full of zeroes is not an explanation.
    if (!planCount) return;
    var dest = document.getElementById("sort-dest").value.trim();
    var checked = document.querySelector('input[name="sort-mode"]:checked');
    var mode = checked ? checked.value : "move";
    var statusEl = document.getElementById("sort-status");
    statusEl.textContent = "";
    // The dialog states the numbers of the plan the tab is showing, so it asks about
    // the same criterion — a summary of the city plan under a tree of people would be
    // a number nobody has to honour.
    fetch("/api/sort/summary?dest=" + encodeURIComponent(dest) +
          "&by=" + encodeURIComponent(layoutBy()))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || data.error) {
          statusEl.textContent = I18N.sort_summary_error + ((data && data.error) || "");
          return;
        }
        openSortDialog(data, dest, mode);
      })
      .catch(function (err) {
        statusEl.textContent = I18N.sort_summary_error + err;
      });
  });

  document.getElementById("sort-dialog-cancel").addEventListener("click", closeSortDialog);

  sortDialogEl.addEventListener("click", function (e) {
    if (e.target === sortDialogEl) closeSortDialog();  // клик по фону — отмена
  });

  document.getElementById("sort-dialog-ok").addEventListener("click", function () {
    closeSortDialog();
    startSort();
  });

  document.getElementById("sort-browse-btn").addEventListener("click", function () {
    browseIntoField(this, function (path) {
      document.getElementById("sort-dest").value = path;
    });
  });

  // Дефолт пути назначения = <источник>_sorted (сервер знает источник); только
  // если пользователь ещё ничего не ввёл — свой ввод не затираем.
  fetch("/api/sort/suggest-dest").then(function (r) { return r.json(); })
    .then(function (resp) {
      var input = document.getElementById("sort-dest");
      if (resp && resp.dest && !input.value.trim()) input.value = resp.dest;
    }).catch(function () {});

  pollSortStatus();

  // --- вкладка «Перемещения» (U5, read-only манифест sort --apply) -------

  var MOVE_STATUS_LABELS = {
    planned: I18N.status_planned, done: I18N.status_done, undone: I18N.status_undone,
    failed: I18N.status_failed, deleted: I18N.status_deleted,
  };

  function moveStatusLabel(status) {
    return MOVE_STATUS_LABELS[status] || status;
  }

  var MOVE_STATUS_CHIP_CLASS = {
    done: "chip-good", planned: "chip-accent", failed: "chip-danger", deleted: "chip-danger",
    undone: "chip",
  };

  function renderMoveFiles(files) {
    var table = document.createElement("table");
    files.forEach(function (item) {
      var tr = document.createElement("tr");
      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(item.file_id, null, 0, item.thumb_url, item.video));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = item.name;
      tdThumb.appendChild(nameEl);
      tr.appendChild(tdThumb);
      var tdMeta = document.createElement("td");
      var pathLine = document.createElement("div");
      pathLine.textContent = item.src + " → " + item.dst;
      tdMeta.appendChild(pathLine);
      var statusChip = document.createElement("span");
      statusChip.className = "chip " + (MOVE_STATUS_CHIP_CLASS[item.status] || "chip");
      statusChip.textContent = moveStatusLabel(item.status);
      tdMeta.appendChild(statusChip);
      tr.appendChild(tdMeta);
      table.appendChild(tr);
    });
    return wrapTable(table);
  }

  function batchSummaryText(batch, count) {
    var parts = [I18N.batch_label + " #" + batch.id, batch.mode, batch.operation || "move",
        I18N.started_label + " " + batch.started_at];
    parts.push(batch.finished_at ? I18N.finished_label + " " + batch.finished_at
        : I18N.in_progress_label);
    parts.push(I18N.files_count_label + ": " + count);
    return parts.join(" · ");
  }

  function loadMoves() {
    var container = document.getElementById("tree-moves");
    var summary = document.getElementById("moves-summary");
    fetch("/api/moves")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        container.textContent = "";
        summary.textContent = "";
        applyUndoAvailability(data);
        if (!data.batch) {
          summary.appendChild(stateEl("empty", I18N.no_moves_yet));
          return;
        }
        summary.textContent = batchSummaryText(data.batch, data.moves.length);
        var root = buildTree(data.moves);
        if (root.files.length) container.appendChild(renderMoveFiles(root.files));
        Object.keys(root.children).sort().forEach(function (name) {
          container.appendChild(renderNode(name, root.children[name], 0, renderMoveFiles));
        });
      })
      .catch(function (err) {
        applyUndoAvailability(null);
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_moves + err));
      });
  }

  // --- F97: откат последнего батча кнопкой (POST /api/undo) ----------------
  // Кнопка живёт рядом с манифестом и откатывает ровно тот батч, который манифест
  // показывает — селектора батчей нет намеренно: меньше способов ошибиться кнопкой,
  // которая удаляет файлы. Вторая точка входа — панель результата после отменённой
  // раскладки; эндпоинт и диалог у них общие.

  var UNDO_POLL_MS = 1000;
  var undoPollTimer = null;
  var undoDialogEl = document.getElementById("undo-dialog");

  // Строки, которые откат реально трогает: 'done' и хвост прерванного переноса
  // ('planned' — журнал коммитится ДО операции, статус мог не успеть записаться).
  function undoableCount(moves) {
    var n = 0;
    moves.forEach(function (m) {
      if (m.status === "done" || m.status === "planned") n += 1;
    });
    return n;
  }

  function applyUndoAvailability(data) {
    if (!data || !data.batch) {
      undoAvailable = false;
      undoBatchInfo = null;
    } else {
      undoBatchInfo = {
        operation: data.batch.operation || "move",
        dest_root: data.batch.dest_root || "",
        count: undoableCount(data.moves || []),
      };
      undoAvailable = undoBatchInfo.count > 0;
    }
    updateBusyControlsDisabled();
  }

  function refreshUndoAvailability() {
    movesLoaded = true;  // манифест перезагружаем прямо сейчас, повтор по клику не нужен
    loadMoves();
  }

  // Диалог называет операцию своими словами и числами из манифеста: без числа
  // страшную кнопку не нажимают вообще, а эта кнопка удаляет файлы.
  function undoConfirmText() {
    if (!undoBatchInfo) return I18N.undo_nothing_to_undo;
    if (undoBatchInfo.operation === "move") {
      return fmt(I18N.undo_confirm_move, { n: undoBatchInfo.count });
    }
    return fmt(I18N.undo_confirm_copy,
        { n: undoBatchInfo.count, dest: undoBatchInfo.dest_root });
  }

  function openUndoDialog() {
    if (!undoAvailable) {
      document.getElementById("undo-status").textContent = I18N.undo_nothing_to_undo;
      return;
    }
    document.getElementById("undo-dialog-text").textContent = undoConfirmText();
    undoDialogEl.hidden = false;
  }

  function closeUndoDialog() {
    undoDialogEl.hidden = true;
  }

  function renderUndoStatus(data) {
    var bar = document.getElementById("undo-progress");
    var statusEl = document.getElementById("undo-status");
    var strayEl = document.getElementById("undo-stray");
    var cancelBtn = document.getElementById("undo-cancel-btn");
    undoRunning = !!data.running;
    cancelBtn.style.display = data.running ? "" : "none";
    cancelBtn.disabled = !!data.cancel_requested;
    updateBusyControlsDisabled();
    bar.style.display = data.running ? "" : "none";
    if (data.running) {
      bar.max = data.total || 0;
      bar.value = data.done || 0;
      statusEl.textContent = data.cancel_requested
          ? I18N.undo_cancel_requested
          : fmt(I18N.undo_progress_line, { done: data.done, all: data.total });
      strayEl.textContent = "";
      return;
    }
    if (!data.finished) { statusEl.textContent = ""; strayEl.textContent = ""; return; }
    if (data.error) {
      statusEl.textContent = I18N.undo_error_prefix + data.error;
      strayEl.textContent = "";
      return;
    }
    var r = data.result || {};
    statusEl.textContent = r.cancelled
        ? fmt(I18N.undo_cancelled_text, { n: r.undone || 0 })
        : fmt(I18N.undo_done_text,
              { n: r.undone || 0, m: r.missing || 0, f: r.failed || 0 });
    // Битые копии называются поимённо: они остались лежать в результате и выглядят
    // как обычные фото — молча их не удаляем и молча про них не забываем.
    strayEl.textContent = (r.stray && r.stray.length)
        ? I18N.undo_stray_title + " " + r.stray.join(", ") : "";
    refreshUndoAvailability();
    refreshPlan();
  }

  function pollUndoStatus() {
    fetch("/api/undo/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderUndoStatus(data);
        if (data.running) undoPollTimer = setTimeout(pollUndoStatus, UNDO_POLL_MS);
      });
  }

  document.getElementById("undo-btn").addEventListener("click", openUndoDialog);
  document.getElementById("undo-dialog-cancel").addEventListener("click", closeUndoDialog);

  undoDialogEl.addEventListener("click", function (e) {
    if (e.target === undoDialogEl) closeUndoDialog();  // клик по фону — отмена
  });

  document.getElementById("undo-dialog-ok").addEventListener("click", function () {
    closeUndoDialog();
    var statusEl = document.getElementById("undo-status");
    statusEl.textContent = "";
    document.getElementById("undo-stray").textContent = "";
    postJson("/api/undo", {}).then(function (resp) {
      if (resp && resp.error) {
        statusEl.textContent = I18N.undo_start_error_prefix + resp.error;
        return;
      }
      if (undoPollTimer) clearTimeout(undoPollTimer);
      pollUndoStatus();
    });
  });

  document.getElementById("undo-cancel-btn").addEventListener("click", function () {
    this.disabled = true;  // мгновенный фидбэк, не ждём следующего polling-тика
    document.getElementById("undo-status").textContent = I18N.undo_cancel_requested;
    postJson("/api/undo/cancel", {});
  });

  pollUndoStatus();
  refreshUndoAvailability();

  // --- альбомы (F35): кнопка «Собрать в папку» на карточках Люди/События ---

  // F145: the album button moves files, so it is dead while anything runs — and the
  // reason is written next to it, the same `.busy-hint` the static blocks carry.
  function appendAlbumBusyHint(box) {
    var hint = document.createElement("span");
    hint.className = "override-hint busy-hint";
    hint.textContent = I18N.actions_busy;
    hint.style.display = uiBusy() ? "" : "none";
    box.appendChild(hint);
    return hint;
  }

  function albumModeSelect() {
    var select = document.createElement("select");
    ["link", "copy", "move"].forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m;
      opt.textContent = I18N["album_mode_" + m];
      select.appendChild(opt);
    });
    return select;
  }

  // Поле пути назначения альбома + «Обзор…» (F60, тот же мотив, что и
  // sort-dest/process-source-dir): дефолт = <источник>_sorted с сервера,
  // префилл только если поле ещё пустое (свой ввод не затираем).
  function appendAlbumDestControls(box) {
    var input = document.createElement("input");
    input.type = "text";
    input.className = "album-dest-input";
    input.placeholder = I18N.album_dest_placeholder;
    box.appendChild(input);
    var browseBtn = makeBtn("ghost", null, I18N.process_browse_button, "btn-sm album-browse-btn");
    browseBtn.addEventListener("click", function () {
      browseIntoField(this, function (path) { input.value = path; });
    });
    box.appendChild(browseBtn);
    fetch("/api/sort/suggest-dest").then(function (r) { return r.json(); })
      .then(function (resp) {
        if (resp && resp.dest && !input.value.trim()) input.value = resp.dest;
      }).catch(function () {});
    return input;
  }

  function albumPreviewText(resp) {
    var txt = fmt(I18N.album_preview_text, { n: resp.count, dest: resp.dest });
    if (resp.mode === "move" && resp.blocked_multi) {
      txt += fmt(I18N.album_blocked_text, { k: resp.blocked_multi });
    }
    return txt;
  }

  // Превью (apply=false) -> подтверждение (текст зависит от режима, move явно
  // предупреждает об изъятии из пула) -> apply=true. statusEl получает
  // прогресс/результат; при успешном apply сбрасывается кэш вкладки
  // «Перемещения», чтобы следующий заход её перезагрузил (F35 п.4).
  function gatherAlbum(kind, selector, mode, where, name, dest, statusEl) {
    var body = { kind: kind, selector: selector, mode: mode, apply: false };
    if (where) body.where = [where];
    if (name) body.name = name;
    if (dest) body.dest = dest;
    statusEl.textContent = I18N.album_in_progress;
    postJson("/api/album", body).then(function (resp) {
      if (resp.error) { statusEl.textContent = resp.error; return; }
      var confirmMsg = albumPreviewText(resp) + "\n" +
          (mode === "move" ? I18N.album_confirm_move : I18N.album_confirm_generic);
      if (!window.confirm(confirmMsg)) { statusEl.textContent = ""; return; }
      body.apply = true;
      statusEl.textContent = I18N.album_in_progress;
      postJson("/api/album", body).then(function (resp2) {
        if (resp2.error) { statusEl.textContent = resp2.error; return; }
        statusEl.textContent = fmt(I18N.album_result_text,
            { n: resp2.transferred, f: resp2.failed });
        movesLoaded = false;
      });
    });
  }

  // F139: the gather row of a slice that has no subject to choose inside it — a class
  // bucket ("Products") or a quality slice ("Blurred"). The same three controls every
  // other album has (mode, an optional folder name, a destination) and the same
  // dry-run-then-confirm path; the only thing that varies is the `kind` the server was
  // asked to gather, and `kind` = null takes the row away entirely, which is what a
  // sensitive class and the duplicates look like.
  //
  // Rebuilt only when the kind CHANGES: the row is drawn from inside the paging render,
  // and re-creating it per page would ask the server for a default destination again and
  // wipe a path somebody had typed.
  function renderSliceAlbumControls(boxId, kind) {
    var box = document.getElementById(boxId);
    if (box.getAttribute("data-kind") === (kind || "")) return;
    box.setAttribute("data-kind", kind || "");
    box.textContent = "";
    if (!kind) return;
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "album-name-input";
    nameInput.placeholder = I18N.album_name_placeholder;
    box.appendChild(nameInput);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      gatherAlbum(kind, "", modeSelect.value, null, nameInput.value.trim() || null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  // --- лайтбокс (F42): один переиспользуемый оверлей поверх /photo/<id> ---
  // Заполняется по клику (не N скрытых оверлеев). Клик по фону/Esc закрывает;
  // стрелки ←/→ листают переданный список sample-кадров (опц., F42).
  //
  // F80: у ВИДЕО те же стрелки листают кадры ОДНОГО ролика (/frame/<id>/<i>), а не
  // соседние файлы: воспроизведения нет, и несколько кадров — единственный способ
  // понять, что там снято. Для фото поведение не меняется ни на шаг: lightboxFrames
  // остаётся нулём, кадр берётся всё тем же /preview/<id>.
  //
  // Кадры тянутся лениво: src ставится ровно одному кадру, тому, что показан. Сетка
  // плиток по-прежнему знает только /thumb — шесть кадров на плитку никто не грузит.

  var lightboxEl = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightbox-img");
  var lightboxPrev = document.getElementById("lightbox-prev");
  var lightboxNext = document.getElementById("lightbox-next");
  var lightboxDots = document.getElementById("lightbox-dots");
  var lightboxSamples = null;
  var lightboxIndex = 0;
  var lightboxFrames = 0;   // > 0 <=> открыто видео, столько кадров у ленты
  var lightboxFrame = 0;

  function renderLightboxDots() {
    lightboxDots.textContent = "";
    for (var i = 0; i < lightboxFrames; i++) {
      var dot = document.createElement("button");
      dot.type = "button";
      dot.className = "lightbox-dot" + (i === lightboxFrame ? " active" : "");
      dot.title = fmt(I18N.frame_of, { n: i + 1, all: lightboxFrames });
      dot.addEventListener("click", (function (frame) {
        return function (e) { e.stopPropagation(); showLightboxFrame(frame); };
      })(i));
      lightboxDots.appendChild(dot);
    }
    var multi = lightboxFrames > 1;
    lightboxDots.hidden = !multi;
    lightboxPrev.hidden = !multi;
    lightboxNext.hidden = !multi;
  }

  function showLightboxFrame(frame) {
    lightboxFrame = frame;
    lightboxImg.src = "/frame/" + lightboxSamples[lightboxIndex] + "/" + frame;
    renderLightboxDots();
  }

  function showLightboxAt(index) {
    lightboxIndex = index;
    loadLightboxOffer();
    if (lightboxFrames) { showLightboxFrame(0); return; }
    // /preview — крупный ДЕКОДИРОВАННЫЙ JPEG (HEIC/RAW рендерятся), не сырой /photo
    lightboxImg.src = "/preview/" + lightboxSamples[index];
  }

  function stepLightboxFrame(delta) {
    showLightboxFrame((lightboxFrame + delta + lightboxFrames) % lightboxFrames);
  }

  function openLightbox(samples, index, videoFrames) {
    // A COPY of the caller's list: F168 splices the processed copy in beside the frame it
    // was made from, and the array handed over belongs to a card that is still on screen.
    lightboxSamples = (samples || []).slice();
    lightboxFrames = videoFrames || 0;
    lightboxFrame = 0;
    renderLightboxDots();
    showLightboxAt(index);
    lightboxEl.hidden = false;
  }

  function closeLightbox() {
    lightboxEl.hidden = true;
    lightboxImg.src = "";
    lightboxSamples = null;
    lightboxFrames = 0;
    lightboxFrame = 0;
    renderLightboxDots();
    forgetLightboxOffer();
  }

  // --- F168: "try to improve" on the frame that is open, in every slice --------------
  // The action itself is F149's and is not reimplemented here: the same POST, the same
  // answer, the same reason codes. What is new is WHERE it can be reached from. It used
  // to live in one slice, behind a blur filter measured to find 8% of the frames a person
  // calls soft; the gain the second measurement found belongs to SMALL frames rather than
  // to blurred ones, and small frames are everywhere — in the cities, in the people, in a
  // search. So the entrance is the expanded frame, which every slice already opens, and
  // there is deliberately no control on the tiles: thirteen of those are what F133 spent a
  // feature removing, and this one is pressed a few times a year.
  //
  // Whether to offer it at all is the SERVER's answer (`/api/restore/offer`) — the private
  // classes and the ceiling are one rule each, and a copy of them in JS would be the
  // second place to forget to update.

  var lightboxRestoreEl = document.getElementById("lightbox-restore");
  var lightboxRestoreBtn = document.getElementById("lightbox-restore-btn");
  var lightboxRestoreNote = document.getElementById("lightbox-restore-note");
  var lightboxRestoreBadge = document.getElementById("lightbox-restore-badge");
  var lightboxRestoreStatus = document.getElementById("lightbox-restore-status");
  var lightboxOffer = null;      // the answer about the frame on screen, or null
  var lightboxOfferFor = null;   // which id it was asked about — a late answer is dropped
  var lightboxRestoring = false;

  function lightboxFrameId() {
    return lightboxSamples && lightboxSamples.length ? lightboxSamples[lightboxIndex] : null;
  }

  function renderLightboxRestore() {
    var offer = lightboxOffer;
    if (!offer) { lightboxRestoreEl.hidden = true; return; }
    var source = offer.restored_from;
    lightboxRestoreBadge.hidden = !source;
    if (source) {
      // Says PROCESSED and names the frame it came from: the copy is an ordinary member
      // of the collection now, so anywhere it turns up it must not read as a second
      // similar photograph nobody remembers taking.
      lightboxRestoreBadge.textContent = fmt(I18N.review_restore_source_badge,
                                             { name: source.name });
      lightboxRestoreBadge.title = I18N.review_restore_badge_hint;
    }
    // The size decides, not the slice. Above `features.restore_max_edge` the copy would be
    // rebuilt from a reduced frame and the measurement found no gain there, so the button
    // goes and the sentence stays — a withdrawn offer without a word is the silent half of
    // the same promise.
    var offered = offer.available && !offer.rebuilt;
    lightboxRestoreBtn.hidden = !offered;
    lightboxRestoreBtn.disabled = uiBusy() || lightboxRestoring;   // F145
    lightboxRestoreBtn.textContent = lightboxRestoring ? I18N.review_restore_running
                                                       : I18N.review_restore;
    var note = offer.available && offer.rebuilt
        ? fmt(I18N.review_restore_too_large,
              { max_edge: offer.max_edge, source_edge: offer.source_edge })
        : "";
    lightboxRestoreNote.hidden = !note;
    lightboxRestoreNote.textContent = note;
    lightboxRestoreEl.hidden = !(offered || note || source);
  }

  registerBusyRefresh(renderLightboxRestore);

  function forgetLightboxOffer() {
    lightboxOffer = null;
    lightboxOfferFor = null;
    lightboxRestoreStatus.textContent = "";
    renderLightboxRestore();
  }

  function loadLightboxOffer() {
    forgetLightboxOffer();
    // A clip is not this action's business (the engine is about images, and the route
    // refuses one anyway) — nothing is asked about it.
    var id = lightboxFrames ? null : lightboxFrameId();
    if (id === null) return;
    lightboxOfferFor = id;
    fetch("/api/restore/offer?file_id=" + encodeURIComponent(id))
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (offer) {
        if (lightboxOfferFor !== id) return;   // the person has already stepped on
        lightboxOffer = offer;
        renderLightboxRestore();
      })
      .catch(function () { /* the frame simply offers nothing */ });
  }

  function restoreLightboxFrame() {
    var id = lightboxFrameId();
    if (id === null || lightboxRestoring || uiBusy()) return;
    lightboxRestoring = true;
    renderLightboxRestore();
    lightboxRestoreStatus.textContent = I18N.review_restore_running;
    return postJson("/api/review/restore", { file_id: id })
      .then(function (resp) {
        if (resp && resp.ok && resp.item) {
          // The copy appears WHERE THE PRESS WAS. In the Review grid that means a card
          // beside the original; here it means the next frame of the strip the arrows
          // step through, shown at once — a person who pressed and saw nothing change
          // would have no reason to believe anything happened. (Closed meanwhile: the
          // copy is made and indexed all the same, there is simply nothing to show it in.)
          if (!lightboxSamples) return;
          lightboxSamples.splice(lightboxIndex + 1, 0, resp.item.file_id);
          showLightboxAt(lightboxIndex + 1);
          lightboxRestoreStatus.textContent = resp.reused ? I18N.review_restore_reused
                                                          : I18N.review_restore_done;
          if (resp.rebuilt) {
            lightboxRestoreStatus.textContent += " " + fmt(I18N.review_restore_rebuilt, {
              max_edge: resp.max_edge, source_edge: resp.source_edge });
          }
          return;
        }
        // A reason, never an empty result — and the same codes the Review tab translates,
        // including the two refusals this entrance made necessary (a private class, a
        // clip), which the ROUTE decides and not the page.
        var reason = resp && resp.reason
            ? I18N["review_restore_error_" + resp.reason] : null;
        lightboxRestoreStatus.textContent = reason
            || (I18N.review_error_prefix + ((resp && (resp.detail || resp.error)) || ""));
      })
      .catch(function (err) {
        lightboxRestoreStatus.textContent = I18N.review_error_prefix + err;
      })
      .then(function () {
        lightboxRestoring = false;
        renderLightboxRestore();
      });
  }

  lightboxRestoreEl.addEventListener("click", function (e) { e.stopPropagation(); });
  lightboxRestoreBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    restoreLightboxFrame();
  });

  // Короткий ролик отдаёт меньше кадров, чем настроено, и недостающий индекс — это
  // честный 404. Обрезаем ленту по первому промаху и возвращаемся на прошлый кадр:
  // сервер не обязан заранее знать, сколько кадров вытащится из конкретного файла.
  lightboxImg.addEventListener("error", function () {
    if (!lightboxFrames || lightboxFrame < 1) return;
    lightboxFrames = lightboxFrame;
    showLightboxFrame(lightboxFrame - 1);
  });

  lightboxEl.addEventListener("click", closeLightbox);
  lightboxImg.addEventListener("click", function (e) { e.stopPropagation(); });
  lightboxPrev.addEventListener("click", function (e) {
    e.stopPropagation();
    stepLightboxFrame(-1);
  });
  lightboxNext.addEventListener("click", function (e) {
    e.stopPropagation();
    stepLightboxFrame(1);
  });
  document.addEventListener("keydown", function (e) {
    if (lightboxEl.hidden) return;
    if (e.key === "Escape") { closeLightbox(); return; }
    if (lightboxFrames > 1) {
      if (e.key === "ArrowRight") stepLightboxFrame(1);
      else if (e.key === "ArrowLeft") stepLightboxFrame(-1);
      return;
    }
    if (!lightboxSamples || lightboxSamples.length < 2) return;
    if (e.key === "ArrowRight") showLightboxAt((lightboxIndex + 1) % lightboxSamples.length);
    else if (e.key === "ArrowLeft") {
      showLightboxAt((lightboxIndex - 1 + lightboxSamples.length) % lightboxSamples.length);
    }
  });

  // --- вкладка «Люди» (F31, управление кластерами лиц) --------------------

  var clustersById = {};
  var selectedForMerge = {};
  var selectedForMergeCount = 0;

  function updateMergeButton() {
    // F145: merging two clusters rewrites `faces.cluster_id` for both of them.
    document.getElementById("clusters-merge-btn").disabled =
        uiBusy() || selectedForMergeCount !== 2;
  }

  registerBusyRefresh(updateMergeButton);

  function toggleMergeSelection(clusterId, checked) {
    if (checked) {
      if (!(clusterId in selectedForMerge)) selectedForMergeCount += 1;
      selectedForMerge[clusterId] = true;
    } else {
      if (clusterId in selectedForMerge) selectedForMergeCount -= 1;
      delete selectedForMerge[clusterId];
    }
    updateMergeButton();
  }

  function renderClusterCard(c) {
    var card = document.createElement("div");
    card.className = "card" + (c.label ? " named" : "");

    var thumbs = document.createElement("div");
    thumbs.className = "cluster-thumbs";
    // Скелетон рисуется сразу (карточка отзывчива, пока идёт /thumb) —
    // сама миниатюра грузится лениво и фоном; onload плавно проявляет её и
    // снимает скелетон-заглушку (F42).
    c.samples.forEach(function (fileId, idx) {
      var skel = document.createElement("div");
      skel.className = "thumb-skel";
      var img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "";
      img.addEventListener("load", function () { skel.className = "thumb-skel loaded"; });
      img.addEventListener("click", function () { openLightbox(c.samples, idx); });
      img.src = "/thumb/" + fileId;
      skel.appendChild(img);
      thumbs.appendChild(skel);
    });
    card.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "cluster-meta";
    meta.textContent = (c.label ? c.label : I18N.unnamed) + " \u00b7 " + c.size + " " +
        I18N.faces_unit;
    card.appendChild(meta);

    var form = document.createElement("div");
    form.className = "cluster-name-form";
    var input = document.createElement("input");
    input.type = "text";
    input.value = c.label || "";
    input.placeholder = I18N.person_name_placeholder;
    form.appendChild(input);
    var btnName = makeBtn("primary", "tag", I18N.name_button, "btn-sm");
    btnName.addEventListener("click", function () {
      var name = input.value.trim();
      if (!name) { window.alert(I18N.alert_enter_name); return; }
      postJson("/api/clusters/label", { cluster_id: c.cluster_id, name: name })
        .then(function (resp) { if (resp && resp.ok) loadClusters(); });
    });
    form.appendChild(btnName);
    card.appendChild(form);

    var mergeLabel = document.createElement("label");
    mergeLabel.className = "cluster-merge-select";
    var checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.addEventListener("change", function () {
      toggleMergeSelection(c.cluster_id, checkbox.checked);
    });
    mergeLabel.appendChild(checkbox);
    mergeLabel.appendChild(document.createTextNode(" " + I18N.select_for_merge));
    card.appendChild(mergeLabel);

    if (c.label) {
      var albumBox = document.createElement("div");
      albumBox.className = "album-controls";
      var modeSelect = albumModeSelect();
      albumBox.appendChild(modeSelect);
      var destInput = appendAlbumDestControls(albumBox);
      var whereInput = document.createElement("input");
      whereInput.type = "text";
      whereInput.placeholder = I18N.album_where_placeholder;
      albumBox.appendChild(whereInput);
      var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
      albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
      var albumStatus = document.createElement("span");
      albumStatus.className = "album-status";
      albumBtn.addEventListener("click", function () {
        var where = whereInput.value.trim();
        gatherAlbum("person", c.label, modeSelect.value, where || null, null,
            destInput.value.trim() || null, albumStatus);
      });
      albumBox.appendChild(albumBtn);
      albumBox.appendChild(albumStatus);
      appendAlbumBusyHint(albumBox);
      card.appendChild(albumBox);
    } else {
      var hint = document.createElement("div");
      hint.className = "album-hint";
      hint.textContent = I18N.album_name_first_hint;
      card.appendChild(hint);
    }

    return card;
  }

  function loadClusters() {
    var container = document.getElementById("clusters-grid");
    fetch("/api/clusters")
      .then(function (r) { return r.json(); })
      .then(function (clusters) {
        container.textContent = "";
        clustersById = {};
        selectedForMerge = {};
        selectedForMergeCount = 0;
        updateMergeButton();
        if (!clusters.length) {
          container.appendChild(sliceEmptyState("person", I18N.no_clusters));
          return;
        }
        var named = clusters.filter(function (c) { return c.label; });
        var unnamed = clusters.filter(function (c) { return !c.label; });
        named.concat(unnamed).forEach(function (c) {
          clustersById[c.cluster_id] = c;
          container.appendChild(renderClusterCard(c));
        });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_clusters + err));
      });
  }

  document.getElementById("clusters-merge-btn").addEventListener("click", function () {
    var ids = Object.keys(selectedForMerge).map(Number);
    if (ids.length !== 2) return;
    var a = clustersById[ids[0]];
    var b = clustersById[ids[1]];
    var dst = a.size >= b.size ? a.cluster_id : b.cluster_id;
    var src = dst === a.cluster_id ? b.cluster_id : a.cluster_id;
    postJson("/api/clusters/merge", { src: src, dst: dst })
      .then(function (resp) { if (resp && resp.ok) loadClusters(); });
  });

  // --- вкладка «События» (F35: список событий + «Собрать в папку») --------

  function renderEventCard(e) {
    var card = document.createElement("div");
    card.className = "card";

    var meta = document.createElement("div");
    meta.className = "event-meta";
    meta.textContent = e.count + " " + I18N.files_count_label + " \u00b7 " +
        [e.started_at, e.ended_at].filter(Boolean).join(" \u2013 ");
    card.appendChild(meta);

    // превью-кадры события (клик -> лайтбокс, стрелки листают кадры события)
    if (e.samples && e.samples.length) {
      var thumbs = document.createElement("div");
      thumbs.className = "event-thumbs";
      e.samples.forEach(function (fileId, idx) {
        thumbs.appendChild(clickableThumb(fileId, e.samples, idx));
      });
      card.appendChild(thumbs);
    }

    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "event-name-input";
    nameInput.value = e.name || "";
    nameInput.placeholder = I18N.album_name_placeholder;
    card.appendChild(nameInput);

    var albumBox = document.createElement("div");
    albumBox.className = "album-controls";
    var modeSelect = albumModeSelect();
    albumBox.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(albumBox);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      var name = nameInput.value.trim();
      gatherAlbum("event", String(e.id), modeSelect.value, null, name || null,
          destInput.value.trim() || null, albumStatus);
    });
    albumBox.appendChild(albumBtn);
    albumBox.appendChild(albumStatus);
    appendAlbumBusyHint(albumBox);
    card.appendChild(albumBox);

    // F85c: событие — самая осязаемая группа, какая есть: это одна поездка, и место
    // у неё одно. Назначение на всё событие целиком — одно действие вместо e.count.
    var placeBox = document.createElement("div");
    placeBox.className = "place-controls";
    var picker = renderPlacePicker(placeBox);
    var placeStatus = document.createElement("span");
    placeStatus.className = "override-status";
    var assignBtn = makeBtn("primary", "pin", I18N.place_assign_button,
        "btn-sm place-assign-btn");
    assignBtn.addEventListener("click", function () {
      assignPlace(picker, "event", String(e.id), "place_assign_confirm",
                  { n: e.count }, placeStatus, null);
    });
    var clearBtn = makeBtn("ghost", null, I18N.place_clear_button,
        "btn-sm place-clear-btn");
    clearBtn.addEventListener("click", function () {
      clearPlace("event", String(e.id), "place_event_clear_confirm",
                 { n: e.count }, placeStatus, null);
    });
    placeBox.appendChild(assignBtn);
    placeBox.appendChild(clearBtn);
    placeBox.appendChild(placeStatus);
    card.appendChild(placeBox);

    return card;
  }

  function loadEvents() {
    var container = document.getElementById("events-list");
    fetch("/api/events")
      .then(function (r) { return r.json(); })
      .then(function (events) {
        container.textContent = "";
        if (!events.length) {
          container.appendChild(sliceEmptyState("event", I18N.no_events));
          return;
        }
        events.forEach(function (e) { container.appendChild(renderEventCard(e)); });
      })
      .catch(function (err) {
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_events + err));
      });
  }

  // --- F174: the action names its destination ------------------------------
  // The folder is the server's answer, computed by the code that builds the plan
  // (`sorter.destinations`). NOTHING here derives a path: a second spelling of the
  // layout rules, in JS, is exactly how a caption starts disagreeing with the plan it
  // is describing — and the caption is the whole feature.

  function destLine(item, template) {
    var line = document.createElement("div");
    line.className = "dest-line";
    if (!item.dest) {
      line.textContent = I18N.dest_unknown;
      return line;
    }
    var why = I18N["dest_why_" + item.dest_reason];
    var text = fmt(template, { folder: item.dest });
    line.textContent = why ? text + ": " + why : text;
    return line;
  }

  // A bulk action states the SPREAD, not the first destination of the selection: the
  // person ticks dozens at a time, and one folder name out of twelve deceives them.
  // The groups come from the server (`_DEST_GROUPS`) — this only counts them.
  var DEST_GROUP_ORDER = ["city", "country", "no_place", "undated", "other"];

  function destBreakdown(items) {
    var counts = {};
    items.forEach(function (it) {
      var group = (it && it.dest_group) || "other";
      counts[group] = (counts[group] || 0) + 1;
    });
    return DEST_GROUP_ORDER.filter(function (g) { return counts[g]; })
      .map(function (g) {
        return fmt(I18N.dest_bulk_item,
                   { n: counts[g], group: I18N["dest_group_" + g] || g });
      })
      .join(", ");
  }

  function destSummary(items) {
    return fmt(I18N.dest_bulk_summary,
               { n: items.length, breakdown: destBreakdown(items) });
  }

  // --- F103: the «Служебные кадры» slice -----------------------------------
  // The classifier's buckets are visible AS buckets: filter chips with a counter, a
  // grid of tiles, several frames ticked at once and ONE return for the whole selection
  // (one at a time is dozens of clicks for "a couple out of 2 202"). The return is a
  // POST /api/overrides with action="photo" (the F77 mechanism, already there): the
  // verdict in media_class is not rewritten, so re-running the tier does not wipe the
  // correction.

  var JUNK_PAGE_SIZE = 200;
  var junkBucket = null;   // null — «Все»
  var junkOffset = 0;
  var junkSelected = {};
  // F174: the cards currently on screen, by file_id — the selection is made of them, so
  // the destinations the server sent with the page are what the bulk caption counts.
  // Nothing is fetched again for it: the answer is already here.
  var junkItems = {};

  function junkBucketLabel(verdict) {
    return I18N["junk_bucket_" + verdict] || verdict;
  }

  // F175: precision is a property of a CLASS, so it is shown only when a class is the
  // one open — the "all" view names no number at all, because four buckets measured
  // separately have no shared one. A class with no measurement of its own falls back to
  // "not measured": inheriting the neighbour's percentage would be the lie this whole
  // caption exists to stop.
  function junkAccuracyText(verdict) {
    if (!verdict) return "";
    return I18N["junk_accuracy_" + verdict] || I18N.junk_accuracy_unmeasured;
  }

  // F171: the promise about the ORDER, made only where the server actually ordered the
  // page by the model's estimate. A bucket the classifier settled without a number of its
  // own is still a list — it is simply not a ranking, and saying otherwise would be the
  // same borrowed claim the accuracy fallback exists to stop.
  function junkOrderText(data) {
    return data.ordered_by_score ? I18N.junk_order_hint : "";
  }

  function junkSelectedIds() {
    return Object.keys(junkSelected).map(Number);
  }

  function junkSelectedItems() {
    return junkSelectedIds().map(function (id) { return junkItems[id] || {}; });
  }

  function refreshJunkControls() {
    var n = junkSelectedIds().length;
    document.getElementById("junk-selected-count").textContent = n ? " (" + n + ")" : "";
    // F145: "back to photos" rewrites `media_class` — the table the run in flight owns.
    document.getElementById("junk-restore-btn").disabled = uiBusy() || n === 0;
    // Where the selection goes, restated on every tick: the spread changes with it, and
    // a number that only appears in the confirmation dialog is seen too late to help.
    document.getElementById("junk-dest-summary").textContent =
        n ? destSummary(junkSelectedItems()) : "";
  }

  registerBusyRefresh(refreshJunkControls);

  // F133: корзины классификатора — это и есть закреплённые срезы «товары / скриншоты /
  // документы»; отдельного ряда чипов больше нет, счётчики уезжают в ряд срезов.
  function renderJunkBuckets(buckets) {
    junkBucketCounts = buckets || [];
    renderSlicePins();
  }

  function renderJunkCard(item) {
    junkItems[item.file_id] = item;
    var card = document.createElement("div");
    card.className = "junk-card" + (item.restored ? " restored" : "") +
        (item.sensitive ? " sensitive" : "");
    // F175: the mark goes at the TOP of the card, above the picture — a bucket that must
    // not be deleted has to be readable while the eye runs over the grid, before the
    // checkbox at the bottom is anywhere near being ticked.
    if (item.sensitive) {
      var mark = document.createElement("span");
      mark.className = "chip chip-accent";
      mark.textContent = I18N.junk_document_mark;
      card.appendChild(mark);
    }
    if (item.thumb_url) {
      card.appendChild(
          clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    } else {
      // Документ: превью не строим вовсе — сервер не прислал ссылку, и запроса к
      // /thumb здесь нет. Заглушка того же размера, чтобы сетка не разъезжалась.
      var stub = document.createElement("div");
      stub.className = "junk-doc-box";
      stub.textContent = I18N.junk_document_no_preview;
      card.appendChild(stub);
    }
    var name = document.createElement("span");
    name.className = "junk-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "junk-card-meta";
    meta.textContent = [junkBucketLabel(item.verdict), item.date || ""]
        .filter(Boolean).join(" \u00b7 ");
    card.appendChild(meta);
    if (item.restored) {
      var chip = document.createElement("span");
      chip.className = "chip chip-good";
      chip.textContent = I18N.junk_restored_mark;
      card.appendChild(chip);
      // F174: the mark is written, the move is not — so the card keeps naming the
      // folder the frame is headed for until the layout actually runs.
      card.appendChild(destLine(item, I18N.dest_goes_to));
      var undoBtn = makeBtn("ghost", null, I18N.junk_undo_restore_button, "btn-sm");
      undoBtn.addEventListener("click", function () { applyJunkAction([item.file_id], "clear"); });
      card.appendChild(undoBtn);
      return card;
    }
    var label = document.createElement("label");
    label.className = "junk-card-select";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.className = "junk-select";
    box.value = String(item.file_id);
    box.checked = !!junkSelected[item.file_id];
    box.addEventListener("change", function () {
      if (box.checked) junkSelected[item.file_id] = true;
      else delete junkSelected[item.file_id];
      refreshJunkControls();
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode(I18N.slice_return_button));
    card.appendChild(label);
    // F174: this bucket is an EXTRACTION from the canon — the frame is not lying in a
    // city right now, and returning it is a real transfer on the next apply. So the
    // card says which folder, and why, before anything is ticked.
    card.appendChild(destLine(item, I18N.dest_goes_to));
    return card;
  }

  function renderJunkPage(data, append) {
    var grid = document.getElementById("junk-grid");
    // F139: the bucket is gathered into a folder like any other slice — or it is not,
    // and the server says which (a sensitive class keeps its counter and gets neither a
    // preview nor an album). The "back to photos" row above is untouched: one movement
    // must not be able to both gather and delete.
    renderSliceAlbumControls("junk-album", data.album_kind);
    // F175: which bucket is open decides which measurement is true here, so the line is
    // rewritten with every page — including the empty one, where "not measured" is still
    // the honest answer about the bucket a person is looking at.
    var accuracy = document.getElementById("junk-accuracy");
    accuracy.textContent = junkAccuracyText(data.bucket) + junkOrderText(data);
    accuracy.style.display = accuracy.textContent ? "" : "none";
    if (!append) {
      grid.textContent = "";
      junkItems = {};      // the cards go, their destinations go with them
    }
    var items = data.items || [];
    items.forEach(function (it) { grid.appendChild(renderJunkCard(it)); });
    var shown = grid.querySelectorAll(".junk-card").length;
    // Пустая корзина — внятное «здесь пусто», а не вечный спиннер.
    if (!shown) grid.appendChild(stateEl("empty", I18N.junk_empty));
    document.getElementById("junk-shown").textContent =
        shown ? fmt(I18N.junk_shown_label, { shown: shown, total: data.total }) : "";
    document.getElementById("junk-more-btn").style.display =
        shown && shown < data.total ? "" : "none";
    // The note about documents — only where such cards are actually on screen (counted
    // over the whole grid, not over the page that was just appended). F175: counted by
    // the mark the cards carry, so the note and the marks appear and disappear together.
    document.getElementById("junk-doc-hint").style.display =
        grid.querySelector(".junk-card.sensitive") ? "" : "none";
    junkOffset = shown;
  }

  function fetchJunk(offset, append) {
    var grid = document.getElementById("junk-grid");
    if (!append) {
      grid.textContent = "";
      grid.appendChild(stateEl("loading", I18N.loading));
    }
    var url = "/api/junk?offset=" + offset + "&limit=" + JUNK_PAGE_SIZE +
        (junkBucket ? "&bucket=" + encodeURIComponent(junkBucket) : "");
    return fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderJunkBuckets(data.buckets || []);
        renderJunkPage(data, append);
      })
      .catch(function (err) {
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_junk + err));
      });
  }

  function loadJunk() {
    junkSelected = {};
    refreshJunkControls();
    document.getElementById("junk-status").textContent = "";
    return fetchJunk(0, false);
  }

  function applyJunkAction(ids, action) {
    var status = document.getElementById("junk-status");
    status.textContent = "";
    return postJson("/api/overrides", { file_ids: ids, action: action })
      .then(function (resp) {
        if (resp && resp.ok) {
          junkSelected = {};
          refreshJunkControls();
          fetchJunk(0, false);
        } else {
          status.textContent = I18N.junk_error_prefix + ((resp && resp.error) || "");
        }
      })
      .catch(function (err) { status.textContent = I18N.junk_error_prefix + err; });
  }

  document.getElementById("junk-restore-btn").addEventListener("click", function () {
    var ids = junkSelectedIds();
    if (!ids.length) return;
    // F174: the question names the spread of the selection, not just its size — "12
    // frames" and "12 frames, 5 of them into no_place" are different decisions.
    if (!window.confirm(fmt(I18N.junk_restore_confirm,
                            { n: ids.length,
                              breakdown: destBreakdown(junkSelectedItems()) }))) return;
    applyJunkAction(ids, "photo");
  });
  document.getElementById("junk-select-all-btn").addEventListener("click", function () {
    document.querySelectorAll("#junk-grid .junk-select").forEach(function (box) {
      box.checked = true;
      junkSelected[parseInt(box.value, 10)] = true;
    });
    refreshJunkControls();
  });
  document.getElementById("junk-select-none-btn").addEventListener("click", function () {
    document.querySelectorAll("#junk-grid .junk-select").forEach(function (box) {
      box.checked = false;
    });
    junkSelected = {};
    refreshJunkControls();
  });
  document.getElementById("junk-more-btn").addEventListener("click", function () {
    fetchJunk(junkOffset, true);
  });
  refreshJunkControls();

  // --- F123: the "Animals" tab -------------------------------------------
  // A page of tiles ordered by confidence, plus the one action the slice affords:
  // gather it into an album. Paged for the same reason as the junk grid (F70) — 805
  // cards with previews are not put into the DOM at once. The score is printed on the
  // card: the verdict is 92% right, and the only way to see where the wrong 8% start
  // is to read down a list that is sorted by exactly that number.

  var ANIMALS_PAGE_SIZE = 200;
  // The length of the LIST, kept so a card redrawn after a mark can restate "showing
  // N of M" without asking the server for a page it already has.
  var animalsTotal = 0;

  function hasManualPet(item) {
    return item.manual !== null && item.manual !== undefined;
  }

  function renderAnimalCard(item) {
    var card = document.createElement("div");
    // F124: `is_animal` comes from the server (the one shared rule), it is never
    // recomputed here — a second spelling of that rule in JS is exactly how the tab
    // and the album start reporting different collections.
    card.className = "animal-card" + (item.is_animal ? "" : " not-animal");
    card.dataset.fileId = String(item.file_id);
    card.appendChild(
        clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    var name = document.createElement("span");
    name.className = "animal-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "animal-card-meta";
    meta.textContent = item.date || "";
    card.appendChild(meta);
    if (item.score !== null && item.score !== undefined) {
      var score = document.createElement("span");
      score.className = "animal-card-score";
      score.textContent = fmt(I18N.animals_score_label,
                              { score: Number(item.score).toFixed(2) });
      card.appendChild(score);
    }
    // A frame decided by hand says so, and says which way: without it the counter
    // moves for no visible reason and a dimmed card looks like a rendering fault.
    if (hasManualPet(item)) {
      var manual = document.createElement("span");
      manual.className = "animal-card-manual";
      manual.textContent = item.manual ? I18N.animals_manual_included
                                       : I18N.animals_manual_excluded;
      card.appendChild(manual);
    }
    // F174: this slice is a VIEW over the canon — the frame is lying in its city folder
    // and stays there whatever is decided here. Said out loud, with the folder named,
    // because the fear the wording has to answer is "will this delete something".
    if (item.is_animal) card.appendChild(destLine(item, I18N.dest_stays_in));
    var actions = document.createElement("div");
    actions.className = "animal-card-actions";
    // One toggle offering the answer the frame does NOT have right now, per card and
    // never over a band: the whole feature is that somebody looked at this frame.
    // F174: taking the mark off is the same intention as returning a product to the
    // photos, so it carries the same words — the difference is the line above.
    var toggle = makeBtn("ghost", null,
        item.is_animal ? I18N.slice_return_button : I18N.animals_mark_animal,
        "btn-sm animal-mark-btn");
    toggle.addEventListener("click", function () {
      markAnimal(item.file_id, item.is_animal ? "not_animal" : "animal");
    });
    actions.appendChild(toggle);
    if (hasManualPet(item)) {
      var back = makeBtn("ghost", null, I18N.animals_mark_clear,
                         "btn-sm animal-clear-btn");
      back.addEventListener("click", function () { markAnimal(item.file_id, "clear"); });
      actions.appendChild(back);
    }
    card.appendChild(actions);
    return card;
  }

  function animalCardEl(fileId) {
    return document.querySelector('#animals-grid .animal-card[data-file-id="' +
                                 fileId + '"]');
  }

  // The second number of the page, the one the shared pager knows nothing about: how many
  // of what is on screen count as animals. After a manual mark that is a different
  // question from "how much of the list is shown" — the card stays in the list and leaves
  // the count.
  function renderAnimalsCounted(shown, animals) {
    document.getElementById("animals-counted").textContent =
        shown ? fmt(I18N.animals_counted_label, { n: animals }) : "";
  }

  // F173: the paging, the button, the "showing N of M" line and the depth warning are the
  // shared pager's now — this slice is ranked by confidence, so the trade the button makes
  // is exactly the one `slice_depth_hint` describes.
  var animalsPager = makePager({
    grid: "animals-grid",
    cardSelector: ".animal-card",
    moreBtn: "animals-more-btn",
    shown: "animals-shown",
    hint: "animals-depth-hint",
    pageSize: ANIMALS_PAGE_SIZE,
    url: function (offset, limit) {
      return "/api/animals?offset=" + offset + "&limit=" + limit;
    },
    card: renderAnimalCard,
    emptyText: function () { return I18N.animals_empty; },
    // F156: "no animals were found" is only true once the stage has looked. With
    // `features.pets` off, or before any run, the honest answer is the other one.
    emptyEl: function () { return sliceEmptyState("animal", I18N.animals_empty); },
    errorText: function () { return I18N.error_loading_animals; },
    after: function (data, shown) {
      animalsTotal = data.total;
      renderAnimalsCounted(shown, data.animals);
    },
  });

  // The answer redraws the card in place instead of reloading the page: this list is
  // read top-down until the confidence runs out, and a reload after every decision
  // would send the reader back to the first screen. The redrawn card comes from the
  // server, so it says what a reload would say.
  function markAnimal(fileId, action) {
    var status = document.getElementById("animals-mark-status");
    status.textContent = "";
    return postJson("/api/animals/mark", { file_ids: [fileId], action: action })
      .then(function (resp) {
        if (!resp || !resp.ok) {
          status.textContent = I18N.animals_error_prefix + ((resp && resp.error) || "");
          return;
        }
        var card = animalCardEl(fileId);
        var fresh = (resp.items || [])[0];
        if (card && fresh) {
          card.parentNode.replaceChild(renderAnimalCard(fresh), card);
        } else if (card) {                    // it left the list entirely (a `clear`
          card.parentNode.removeChild(card);  // on a frame the model never marked)
          animalsTotal = Math.max(0, animalsTotal - 1);
        }
        var grid = document.getElementById("animals-grid");
        // The pager recounts the grid and restates both the counter and the button; the
        // page is not re-fetched, because a reload after every decision would send the
        // reader back to the first screen.
        var shown = animalsPager.sync(animalsTotal);
        if (!shown) grid.appendChild(sliceEmptyState("animal", I18N.animals_empty));
        renderAnimalsCounted(shown, resp.animals);
      })
      .catch(function (err) { status.textContent = I18N.animals_error_prefix + err; });
  }

  // The album controls of the People/Events cards, one per tab instead of one per
  // card: the slice is single, so there is nothing to pick a subject from. The
  // selector goes out empty and the server ignores it (kind='animal'), and the album
  // name is left to the server too — it is a folder name, and it follows `language:`.
  function renderAnimalsAlbumControls() {
    var box = document.getElementById("animals-album");
    if (box.childNodes.length) return;
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button, "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    albumBtn.addEventListener("click", function () {
      gatherAlbum("animal", "", modeSelect.value, null, null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  function loadAnimals() {
    renderAnimalsAlbumControls();
    return animalsPager.load();
  }

  // --- F152: the face slices -----------------------------------------------
  // Three pins over one panel — the junk-bucket arrangement, because these are three
  // questions of one kind and a panel each would be three copies of the same grid. What
  // is NOT shared with the slices around it is the caption: there is no score on a card
  // and no ranking hint above the grid, because a frame is here by a fact of the
  // detector and not by a position in a list. The one line that changes with the slice
  // is the rule it was selected by, thresholds and all.
  //
  // The empty state is a sentence, not a zero. Without a faces run the server answers
  // `reason='no_faces_run'` and `null` counters, and both the pins and this panel say
  // that instead of showing a number nobody measured (F125).

  var FACE_SLICES = ["people", "group", "portrait"];
  var FACE_PAGE_SIZE = 200;
  var faceSlice = "people";
  var faceLoaded = false;
  var faceReason = null;

  function applyFaceCounts(data) {
    faceReason = (data && data.reason) || null;
    faceSliceCounts = {};
    ((data && data.counts) || []).forEach(function (row) {
      faceSliceCounts[row.slice] = row.count;
    });
  }

  // Why this slice holds what it holds, in one line above the grid — with the numbers
  // the server actually selected by, so the rule on screen is the rule that ran.
  function faceHintText(data) {
    if (faceReason === "no_faces_run") return I18N.face_no_faces_run;
    if (faceSlice === "group") {
      return fmt(I18N.face_hint_group, { n: data.group_min });
    }
    if (faceSlice === "portrait") {
      return fmt(I18N.face_hint_portrait,
                 { share: (Number(data.portrait_share) * 100).toFixed(1) });
    }
    return I18N.face_hint_people;
  }

  function renderFaceCard(item) {
    var card = document.createElement("div");
    card.className = "face-card";
    if (item.thumb_url) {
      card.appendChild(
          clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    } else {
      // F133's rule, unchanged: a sensitive class is listed but never decoded for
      // display. A document with a face on it is exactly the frame that rule is for.
      var stub = document.createElement("div");
      stub.className = "junk-doc-box";
      stub.textContent = I18N.junk_document_no_preview;
      card.appendChild(stub);
    }
    var name = document.createElement("span");
    name.className = "face-card-name";
    name.textContent = item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "face-card-meta";
    meta.textContent = [item.date || "", fmt(I18N.face_count_label, { n: item.faces })]
        .filter(Boolean).join(" \u00b7 ");
    card.appendChild(meta);
    return card;
  }

  // F173: the same pager as everywhere else, minus the depth hint. These slices are not
  // ranked — a frame is here because the detector found a face on it — so "further down
  // the list the model is less sure" would be a warning about a risk this list does not
  // have, and the caption above the grid already says the slice is a fact and not an
  // estimate.
  var facePager = makePager({
    grid: "face-grid",
    cardSelector: ".face-card",
    moreBtn: "face-more-btn",
    shown: "face-shown",
    pageSize: FACE_PAGE_SIZE,
    url: function (offset, limit) {
      return "/api/face-slices?slice=" + faceSlice + "&offset=" + offset +
             "&limit=" + limit;
    },
    card: renderFaceCard,
    emptyText: function () {
      return faceReason === "no_faces_run" ? I18N.face_no_faces_run : I18N.face_empty;
    },
    errorText: function () { return I18N.error_loading_face_slices; },
    onData: function (data) {
      // Before the cards, because the empty state and the hint both read `faceReason`.
      applyFaceCounts(data);
      renderSlicePins();
      document.getElementById("face-hint").textContent = faceHintText(data);
    },
  });

  // One album per slice, the animal arrangement: the selector goes out empty and the
  // server ignores it (the collection holds a single slice of each kind), and the album
  // name is left to the server — it is a folder name and follows `language:`. Rebuilt on
  // every open because the KIND changes with the pin.
  function renderFaceAlbumControls() {
    var box = document.getElementById("face-album");
    box.textContent = "";
    if (faceReason === "no_faces_run") return;   // nothing to gather, and no button for it
    var modeSelect = albumModeSelect();
    box.appendChild(modeSelect);
    var destInput = appendAlbumDestControls(box);
    var albumBtn = makeBtn("primary", "folder", I18N.album_button,
                           "btn-sm album-gather-btn");
    albumBtn.disabled = uiBusy();   // F145: gathering an album moves files on disk
    var albumStatus = document.createElement("span");
    albumStatus.className = "album-status";
    var kind = faceSlice;
    albumBtn.addEventListener("click", function () {
      gatherAlbum(kind, "", modeSelect.value, null, null,
          destInput.value.trim() || null, albumStatus);
    });
    box.appendChild(albumBtn);
    box.appendChild(albumStatus);
    appendAlbumBusyHint(box);
  }

  function loadFaceSlice() {
    return facePager.load().then(function () { renderFaceAlbumControls(); });
  }

  // --- F126: the "Review" workspace ----------------------------------------
  // One tab, three slices, one job: look and decide. The switcher keeps every slice in
  // place at zero, because "you have no closed eyes" is an answer and a vanished entry
  // is a riddle. Duplicates are rendered by the code below this block, untouched — they
  // are the only grouped slice, the only one where a keeper is chosen, and the only path
  // in the program that deletes files. The two flat slices share the tile grid and the
  // one action they afford: a mark in `dedup_choice`, which the sorter already reads.
  // Paged like every other grid since F70 — 530 cards with previews do not go into the
  // DOM at once.

  var REVIEW_PAGE_SIZE = 200;
  var REVIEW_SLICES = ["dupes", "blurred", "eyes", "low_resolution"];
  var reviewSlice = "dupes";
  var reviewOffset = 0;
  // Each flat slice opens to a window — `features.blur_review_max`,
  // `features.eye_openness_max` (F179) — and continues past it only when asked: the
  // number is a window, not a verdict. F157: for the blurred slice it is not even a
  // window, it is the depth of the first page of a ranking, and the button below says so.
  var reviewBeyond = false;
  var reviewWindowTotal = 0;
  var reviewSelected = {};

  function reviewSelectedIds() {
    return Object.keys(reviewSelected).map(Number);
  }

  // F149: true while the model is working on a frame. It is about a second per frame and
  // the first press also downloads ~400 MB, so the button has to say that something is
  // happening — and stay dead until it is over, in both cases.
  var reviewRestoring = false;

  function refreshReviewControls() {
    var n = reviewSelectedIds().length;
    document.getElementById("review-selected-count").textContent = n ? " (" + n + ")" : "";
    var dead = uiBusy() || n === 0;   // F145: a mark is a row in `dedup_choice`
    ["review-delete-btn", "review-keep-btn", "review-clear-btn"].forEach(function (id) {
      document.getElementById(id).disabled = dead;
    });
    // F149: exactly ONE frame, and the button says so by being dead for anything else.
    // There is no bulk shape behind it to reach even by hand — the route takes a single
    // `file_id` and refuses a list.
    var restoreBtn = document.getElementById("review-restore-btn");
    restoreBtn.disabled = uiBusy() || reviewRestoring || n !== 1;
    restoreBtn.textContent = reviewRestoring ? I18N.review_restore_running
                                             : I18N.review_restore;
  }

  registerBusyRefresh(refreshReviewControls);

  function renderReviewCounts(counts) {
    counts.forEach(function (row) {
      var el = document.getElementById("review-count-" + row.slice);
      if (el) el.textContent = " (" + overviewNum(row.count) + ")";
    });
  }

  // Why this slice looks the way it does, in one line above the grid. For closed eyes
  // it is also where the F125 answer lands: without a faces run there is no data, and
  // saying so beats showing a zero that reads as "nobody blinked".
  function reviewHintText(data) {
    if (reviewSlice === "eyes") {
      return data.eyes_reason === "no_faces_run"
          ? I18N.review_eyes_no_faces
          : fmt(I18N.review_hint_eyes, { max: data.eye_max });
    }
    // F150: the ceiling comes off the answer, not out of a constant here — the number in
    // the sentence has to be the one the list was actually built with.
    if (reviewSlice === "low_resolution") {
      return fmt(I18N.review_hint_low_resolution, { mp: data.low_resolution_mp });
    }
    // F157: the second sentence only where the second number exists (`blur_order`), and
    // never as a promise the database cannot keep — on a collection indexed before F155
    // there is no face sharpness and the list is ordered by the frame alone.
    var blurred = fmt(I18N.review_hint_blurred, { max: data.blur_max });
    return data.blur_order === "face_sharpness"
        ? blurred + I18N.review_hint_blurred_faces : blurred;
  }

  // F150: "1280×960 (1.2 MP)" — the size of the picture, which the 200 px thumbnail
  // beside it cannot show. Empty for a frame whose dimensions the index never learned:
  // an unknown size is not a small one, and such a frame is in no slice anyway.
  function reviewResolutionLabel(item) {
    if (!item.width || !item.height) return "";
    return fmt(I18N.review_resolution_label, {
      w: item.width, h: item.height,
      mp: (item.width * item.height / 1000000).toFixed(1),
    });
  }

  function renderReviewCard(item) {
    var card = document.createElement("div");
    card.className = "review-card" +
        (item.action === "to_delete" ? " marked-delete" : "") +
        (item.action === "keep" ? " marked-keep" : "") +
        (item.restored ? " processed" : "");
    // F149: the copy is inserted next to its original, so a card has to be findable by
    // the id it is about — hence the attribute rather than a lookup by position.
    card.setAttribute("data-file-id", String(item.file_id));
    card.appendChild(
        clickableThumb(item.file_id, [item.file_id], 0, item.thumb_url, item.video));
    if (item.restored) {
      // Says PROCESSED, never "improved": the model draws plausible detail instead of
      // recovering what was lost, and the person comparing the two pictures has to know
      // which one is the photograph.
      var badge = document.createElement("span");
      badge.className = "review-card-processed";
      badge.textContent = I18N.review_restore_badge;
      badge.title = I18N.review_restore_badge_hint;
      card.appendChild(badge);
    }
    var name = document.createElement("span");
    name.className = "review-card-name";
    name.textContent = item.name;
    name.title = item.src_path || item.name;
    card.appendChild(name);
    var meta = document.createElement("span");
    meta.className = "review-card-meta";
    var sharp = item.sharpness === null || item.sharpness === undefined ? "" :
        fmt(I18N.review_sharpness_label, { value: Number(item.sharpness).toFixed(0) });
    meta.textContent = [item.src_dir, item.date || "", sharp,
                        reviewResolutionLabel(item), actionLabel(item.action)]
        .filter(Boolean).join(" \u00b7 ");
    card.appendChild(meta);
    var label = document.createElement("label");
    label.className = "review-card-select";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.className = "review-select";
    box.value = String(item.file_id);
    box.checked = !!reviewSelected[item.file_id];
    box.addEventListener("change", function () {
      if (box.checked) reviewSelected[item.file_id] = true;
      else delete reviewSelected[item.file_id];
      refreshReviewControls();
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + I18N.review_select_label));
    card.appendChild(label);
    return card;
  }

  function renderReviewPage(data, append) {
    var grid = document.getElementById("review-grid");
    if (!append) grid.textContent = "";
    (data.items || []).forEach(function (it) { grid.appendChild(renderReviewCard(it)); });
    // F149: `:not(.processed)` — a processed copy is not a frame of the slice and must
    // not shift the paging window or the "showing N of M" line it is counted into.
    var shown = grid.querySelectorAll(".review-card:not(.processed)").length;
    if (!shown) {
      grid.appendChild(stateEl("empty",
          data.eyes_reason === "no_faces_run" && reviewSlice === "eyes"
              ? I18N.review_eyes_no_faces : I18N.review_empty));
    }
    // F157: the blurred slice counts the LIST and not a population — "showing 2 210 of
    // 19 211" would be two numbers, neither of which is the number of blurred frames.
    // Every other slice keeps "of M": there M is a fact (groups, small frames) rather
    // than the length of a ranking cut wherever the config happens to cut it.
    document.getElementById("review-shown").textContent = !shown ? ""
        : reviewSlice === "blurred"
            ? fmt(I18N.review_shown_ranked, { shown: shown })
            : fmt(I18N.review_shown_label, { shown: shown, total: data.total });
    // Past the end of the window the button can change its meaning and not just its
    // target: for closed eyes the next page is no longer "more of the same list" but a
    // step outside the window the list opened to (F157 left the blurred slice out of
    // that — see below, its first page is not a claim about anything).
    // WHICH SLICES HAVE A WINDOW, listed rather than negated. Blurred opens down to
    // `blur_review_max` and closed eyes down to `eye_openness_max` (F179): both are
    // RANKINGS cut short, so there is a "further down the list" to step into. Duplicates
    // have no ranking, and `low_resolution` (F150) has no window either — its megapixel
    // ceiling is the membership rule itself, so `beyond` selects exactly the same frames
    // and the button would promise a page that does not exist. `!== "dupes"` was the same
    // thing right up until F150 added a fourth slice.
    var BEYOND_SLICES = ["blurred", "eyes"];
    var beyondNext = BEYOND_SLICES.indexOf(reviewSlice) >= 0 && !reviewBeyond &&
        shown >= reviewWindowTotal;
    var more = shown < data.total || beyondNext;
    var moreBtn = document.getElementById("review-more-btn");
    // F157: for the blurred slice the button keeps saying "show more", because that is
    // what it does — the first page ends where `blur_review_max` put it, and the next one
    // continues the same ordering. "Show past the window" belongs to the closed eyes,
    // where the window is the measured 62% and stepping outside it IS a change of
    // meaning. Same request either way; only the promise is different.
    moreBtn.textContent = beyondNext && reviewSlice !== "blurred"
        ? I18N.review_load_more_beyond : I18N.review_load_more;
    moreBtn.style.display = more ? "" : "none";
    reviewOffset = shown;
    reinsertRestoredCards();
  }

  function fetchReview(offset, append) {
    var flat = reviewSlice !== "dupes";
    var grid = document.getElementById("review-grid");
    if (flat && !append) {
      grid.textContent = "";
      grid.appendChild(stateEl("loading", I18N.loading));
    }
    var url = "/api/review?slice=" + reviewSlice + "&offset=" + offset +
        "&limit=" + REVIEW_PAGE_SIZE + (reviewBeyond ? "&beyond=1" : "");
    return fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderReviewCounts(data.counts || []);
        reviewWindowTotal = data.window_total || 0;
        // F139: the flat slices are gathered into a folder like people and events are;
        // the duplicates are not (`album_kind` is null there), and the marking row above
        // stays exactly where it was — gathering and deleting are two movements.
        renderSliceAlbumControls("review-album", data.album_kind);
        if (!flat) return;
        document.getElementById("review-hint").textContent = reviewHintText(data);
        renderReviewPage(data, append);
      })
      .catch(function (err) {
        if (!flat) return;   // the duplicates list reports its own failures
        grid.textContent = "";
        grid.appendChild(stateEl("error", I18N.error_loading_review + err));
      });
  }

  function selectReviewSlice(slice) {
    if (REVIEW_SLICES.indexOf(slice) < 0) return;
    reviewSlice = slice;
    reviewBeyond = false;
    reviewSelected = {};
    reviewRestoredItems = [];   // F149: the comparison belongs to the slice it was made in
    refreshReviewControls();
    document.getElementById("review-status").textContent = "";
    REVIEW_SLICES.forEach(function (name) {
      document.getElementById("review-slice-" + name)
          .classList.toggle("active", name === slice);
    });
    var grouped = slice === "dupes";
    document.getElementById("review-dupes").style.display = grouped ? "" : "none";
    document.getElementById("review-flat").style.display = grouped ? "none" : "";
    if (grouped && !dupesLoaded) {
      dupesLoaded = true;
      loadDupes();
    }
    return fetchReview(0, false);
  }

  function applyReviewMark(action) {
    var ids = reviewSelectedIds();
    if (!ids.length) return;
    var status = document.getElementById("review-status");
    status.textContent = "";
    return postJson("/api/review/mark", { file_ids: ids, action: action })
      .then(function (resp) {
        if (resp && resp.ok) {
          status.textContent = fmt(I18N.review_marked_status, { n: resp.marked });
          // F149: a processed copy is in no slice, so the redraw below cannot bring its
          // decision back from the server — it is carried on the remembered card instead.
          reviewRestoredItems.forEach(function (item) {
            if (ids.indexOf(item.file_id) >= 0) {
              item.action = action === "clear" ? null : action;
            }
          });
          reviewSelected = {};
          refreshReviewControls();
          fetchReview(0, false);
        } else {
          status.textContent = I18N.review_error_prefix + ((resp && resp.error) || "");
        }
      })
      .catch(function (err) { status.textContent = I18N.review_error_prefix + err; });
  }

  // F149: the copy appears WHERE THE ORIGINAL IS and takes part in the same choice. Not
  // "the file was saved" — a second card beside the first, marked as processed, with the
  // same actions on it, so the person compares two pictures and decides what to keep.
  // Both, either or neither: choosing the copy marks nothing about the original, which is
  // why nothing below writes a decision anywhere.
  function insertRestoredCard(item) {
    var grid = document.getElementById("review-grid");
    var already = grid.querySelector('[data-file-id="' + item.file_id + '"]');
    if (already) already.parentNode.removeChild(already);   // idempotent: one card per id
    var source = grid.querySelector('[data-file-id="' + item.source_file_id + '"]');
    var card = renderReviewCard(item);
    if (source && source.nextSibling) grid.insertBefore(card, source.nextSibling);
    else if (source) grid.appendChild(card);
    else grid.insertBefore(card, grid.firstChild);
    return card;
  }

  // The copy has no `frame_quality` row until the next run measures it, so it is in no
  // slice and the server cannot hand it back on the next page load. It is remembered for
  // as long as the slice is open — otherwise marking either frame would redraw the grid
  // and the picture being compared against would simply vanish mid-comparison.
  var reviewRestoredItems = [];

  function reinsertRestoredCards() {
    reviewRestoredItems.forEach(insertRestoredCard);
  }

  function rememberRestored(item) {
    reviewRestoredItems = reviewRestoredItems.filter(function (kept) {
      return kept.file_id !== item.file_id;
    });
    reviewRestoredItems.push(item);
  }

  function restoreSelectedFrame() {
    var ids = reviewSelectedIds();
    if (ids.length !== 1 || reviewRestoring) return;
    var status = document.getElementById("review-status");
    status.textContent = I18N.review_restore_running;
    reviewRestoring = true;
    refreshReviewControls();
    return postJson("/api/review/restore", { file_id: ids[0] })
      .then(function (resp) {
        if (resp && resp.ok && resp.item) {
          rememberRestored(resp.item);
          insertRestoredCard(resp.item);
          status.textContent = resp.reused ? I18N.review_restore_reused
                                           : I18N.review_restore_done;
          // F169: a frame above the ceiling is reduced before the model and blown back
          // up, so the copy is the same size and holds less of what was really there.
          // Said in the same breath as "done" — the copy looks sharper either way, and
          // this is the part nobody can see by looking at it.
          if (resp.rebuilt) {
            status.textContent += " " + fmt(I18N.review_restore_rebuilt, {
              max_edge: resp.max_edge, source_edge: resp.source_edge });
          }
        } else {
          // A reason, never an empty result: the weights come off the network and being
          // offline is an ordinary state for this program.
          var reason = resp && resp.reason
              ? I18N["review_restore_error_" + resp.reason] : null;
          status.textContent = reason
              || (I18N.review_error_prefix + ((resp && (resp.detail || resp.error)) || ""));
        }
      })
      .catch(function (err) { status.textContent = I18N.review_error_prefix + err; })
      .then(function () {
        reviewRestoring = false;
        refreshReviewControls();
      });
  }

  function loadReview() {
    return selectReviewSlice(reviewSlice);
  }

  REVIEW_SLICES.forEach(function (name) {
    document.getElementById("review-slice-" + name).addEventListener("click", function () {
      selectReviewSlice(name);
    });
  });
  document.getElementById("review-more-btn").addEventListener("click", function () {
    if (reviewSlice === "blurred" && !reviewBeyond && reviewOffset >= reviewWindowTotal) {
      reviewBeyond = true;
    }
    fetchReview(reviewOffset, true);
  });
  document.getElementById("review-delete-btn").addEventListener("click", function () {
    applyReviewMark("to_delete");
  });
  document.getElementById("review-keep-btn").addEventListener("click", function () {
    applyReviewMark("keep");
  });
  document.getElementById("review-restore-btn").addEventListener("click", function () {
    restoreSelectedFrame();
  });
  document.getElementById("review-clear-btn").addEventListener("click", function () {
    applyReviewMark("clear");
  });
  document.getElementById("review-select-all-btn").addEventListener("click", function () {
    document.querySelectorAll("#review-grid .review-select").forEach(function (box) {
      box.checked = true;
      reviewSelected[parseInt(box.value, 10)] = true;
    });
    refreshReviewControls();
  });
  document.getElementById("review-select-none-btn").addEventListener("click", function () {
    document.querySelectorAll("#review-grid .review-select").forEach(function (box) {
      box.checked = false;
    });
    reviewSelected = {};
    refreshReviewControls();
  });
  refreshReviewControls();

  // --- the duplicates slice (U3/F32), unchanged inside the new workspace ---

  function postJson(url, data) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(function (r) { return r.json(); });
  }

  var currentGroups = [];

  function groupFileIds(g) {
    return g.frames.map(function (f) { return f.file_id; });
  }

  function selectedKeeper(g) {
    var radios = document.getElementsByName("keep-" + g.group);
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) return parseInt(radios[i].value, 10);
    }
    return null;
  }

  function groupSkipped(g) {
    var checkbox = document.getElementById("skip-" + g.group);
    return !!(checkbox && checkbox.checked);
  }

  function actionLabel(action) {
    if (action === "keep") return I18N.action_keep;
    if (action === "to_delete") return I18N.action_to_delete;
    return "";
  }

  function renderGroup(g) {
    var box = document.createElement("div");
    box.className = "card dupe-group";

    var title = document.createElement("h3");
    title.textContent = fmt(I18N.group_title, { n: g.group + 1, count: g.frames.length });
    box.appendChild(title);

    var table = document.createElement("table");
    // клик по кадру группы -> лайтбокс; стрелки листают кадры этого дубль-набора
    var groupSamples = g.frames.map(function (fr) { return fr.file_id; });
    g.frames.forEach(function (f, frameIdx) {
      var tr = document.createElement("tr");

      var tdRadio = document.createElement("td");
      var radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "keep-" + g.group;
      radio.value = String(f.file_id);
      radio.checked = f.action === "keep" || (!f.action && f.recommended);
      tdRadio.appendChild(radio);
      tr.appendChild(tdRadio);

      var tdThumb = document.createElement("td");
      tdThumb.appendChild(clickableThumb(f.file_id, groupSamples, frameIdx, f.thumb_url));
      var nameEl = document.createElement("span");
      nameEl.className = "thumb-name";
      nameEl.textContent = f.name;
      // Как во вкладке «Города»: полный путь — в тултипе имени. У дублей имена
      // совпадают по построению, поэтому единственное, чем кадры различаются на
      // глаз, — это где они лежат.
      nameEl.title = f.src_path ? f.src_path + "\\" + f.name : f.name;
      tdThumb.appendChild(nameEl);
      if (f.recommended) {
        var badge = document.createElement("span");
        badge.className = "badge";
        badge.appendChild(icon("check"));
        // F148: a group with a recommendation OF ITS OWN (`group_keeper`) says so under
        // the frame it names, and says who advises. A pair — and any group without a
        // stored row — keeps the plain star it has always had: naming a source where
        // none was asked for invites the user to look for meaning that is not there.
        var isKeeper = !!g.keeper_source && f.file_id === g.keeper_id;
        badge.appendChild(document.createTextNode(
            isKeeper
              ? (g.keeper_source === "model"
                   ? I18N.keeper_badge_model : I18N.keeper_badge_sharpness)
              : I18N.recommended_badge));
        if (isKeeper) { badge.title = I18N.keeper_badge_hint; }
        tdThumb.appendChild(badge);
      }
      tr.appendChild(tdThumb);

      var tdMeta = document.createElement("td");
      var dims = f.width && f.height ? f.width + "×" + f.height : "?";
      var kb = Math.round((f.size || 0) / 1024) + " KB";
      // Исходная папка первой, как в «Городах»: при выборе, какой из одинаковых
      // кадров оставить, решает обычно именно она.
      tdMeta.textContent = [f.src_dir, dims, kb, actionLabel(f.action)]
          .filter(Boolean).join(" · ");
      if (f.src_path) { tdMeta.title = f.src_path; }
      tr.appendChild(tdMeta);

      var tdActions = document.createElement("td");
      tdActions.className = "plan-actions";
      var btnFrameDelete = makeBtn("danger", "trash", I18N.delete, "btn-sm");
      btnFrameDelete.addEventListener("click", function () {
        deletePhoto(f.file_id, function () { tr.remove(); });
      });
      tdActions.appendChild(btnFrameDelete);
      tr.appendChild(tdActions);

      table.appendChild(tr);
    });
    box.appendChild(wrapTable(table));

    var skipLabel = document.createElement("label");
    skipLabel.className = "skip-label";
    var skipCheckbox = document.createElement("input");
    skipCheckbox.type = "checkbox";
    skipCheckbox.id = "skip-" + g.group;
    skipLabel.appendChild(skipCheckbox);
    skipLabel.appendChild(document.createTextNode(" " + I18N.skip_group_label));
    box.appendChild(skipLabel);

    var btnTrash = makeBtn("danger", "trash", I18N.delete_dupes_button);
    btnTrash.addEventListener("click", function () {
      var keep = selectedKeeper(g);
      if (keep === null) { window.alert(I18N.alert_choose_keeper); return; }
      var remember = document.getElementById("delete-remember").checked;
      if (!remember && !window.confirm(fmt(I18N.confirm_trash_group, { n: g.group + 1 }))) {
        return;
      }
      postJson("/api/dupes/trash", { group: groupFileIds(g), keep_file_id: keep })
        .then(loadDupes);
    });
    box.appendChild(btnTrash);

    return box;
  }

  function loadDupes() {
    document.getElementById("dupes-save-status").textContent = "";
    fetch("/api/dupes")
      .then(function (r) { return r.json(); })
      .then(function (groups) {
        currentGroups = groups;
        var container = document.getElementById("dupes-list");
        container.textContent = "";
        if (!groups.length) {
          container.appendChild(stateEl("empty", I18N.no_dupes));
          return;
        }
        groups.forEach(function (g) { container.appendChild(renderGroup(g)); });
      })
      .catch(function (err) {
        var container = document.getElementById("dupes-list");
        container.textContent = "";
        container.appendChild(stateEl("error", I18N.error_loading_dupes + err));
      });
  }

  document.getElementById("dupes-save-all-btn").addEventListener("click", function () {
    var statusEl = document.getElementById("dupes-save-status");
    var groups = [];
    var skip = [];
    currentGroups.forEach(function (g) {
      if (groupSkipped(g)) {
        skip.push(groupFileIds(g));
        return;
      }
      var keep = selectedKeeper(g);
      if (keep === null) return;
      groups.push({ group: groupFileIds(g), keep_file_id: keep });
    });
    if (!groups.length) {
      statusEl.textContent = I18N.select_group_to_save;
      return;
    }
    postJson("/api/dupes/choices", { groups: groups, skip: skip }).then(function (resp) {
      if (resp && typeof resp.saved === "number") {
        statusEl.textContent = fmt(I18N.saved_groups, { n: resp.saved });
      }
      loadDupes();
    });
  });
})();
