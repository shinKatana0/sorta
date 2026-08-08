"""F182: the page template and its `{{key}}` substitution.

The markup, the styles and the browser script are files under `sorta/web/`; this
module puts them back together at import. Nothing here knows about a tab.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import i18n, imaging
from .common import _LANG_SELF_NAMES
from .strings import _UI_STRINGS


def _t(key: str, lang: i18n.Lang) -> str:
    """Resolve a chrome UI string: exact language -> en -> the key itself (see F33)."""
    entry = _UI_STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key


_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _read_web(root: Path, *parts: str) -> str:
    r"""Read one frontend file of `sorta/web/`.

    Text mode on purpose: the template is assembled with "\n" throughout, and a
    checkout that materialises these files with CRLF must not change a byte of what
    is served — universal newlines make that impossible.
    """
    return root.joinpath(*parts).read_text(encoding="utf-8")


def _load_index_template(root: Path | None = None) -> str:
    """Put the three files back together into the template `_render_index_html` fills.

    `root` is for the tests only — the server always assembles from `sorta/web/`.
    """
    web = _WEB_DIR if root is None else root
    return (_read_web(web, "page.html")
            .replace("{{style}}", _read_web(web, "style.css"))
            .replace("{{script}}", _read_web(web, "app", "app.js")))


_INDEX_HTML_TEMPLATE = _load_index_template()


def _render_index_html(lang: i18n.Lang) -> str:
    """Fill the chrome `{{key}}` placeholders and the `window.I18N` JSON (F33).

    Substitution is `str.replace`, not `.format`: the CSS/JS in the template is full
    of single `{`/`}`, which `.format` would read as substitution fields.
    """
    i18n_map = {key: _t(key, lang) for key in _UI_STRINGS}
    lang_options = "".join(
        f'<option value="{code}"{" selected" if code == lang else ""}>{name}</option>'
        for code, name in _LANG_SELF_NAMES.items()
    )
    html = _INDEX_HTML_TEMPLATE.replace("{{lang}}", lang)
    html = html.replace("{{lang_options}}", lang_options)
    # F80: an upper bound, not the truth — the real strip of a short clip can be
    # shorter, and the pager clamps on the first 404.
    html = html.replace("{{video_frames}}", str(imaging.video_frames()))
    html = html.replace("{{i18n_json}}", json.dumps(i18n_map, ensure_ascii=False))
    for key, value in i18n_map.items():
        html = html.replace("{{" + key + "}}", value)
    return html
