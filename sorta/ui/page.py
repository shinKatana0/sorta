"""F182: the page itself — the template and the `{{key}}` substitution.

The markup, the styles and the browser script are files under `sorta/web/`; this
module puts them back together at import and fills the chrome placeholders from
`strings.py`. Nothing here knows about a tab: it is the frame every tab is drawn in.
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


# F182: the page, its stylesheet and its script live in `sorta/web/` as the files they
# are, not as a 6 100-line string literal in the middle of the Python. `page.html` keeps
# two seams — `{{style}}` and `{{script}}` — and they are filled once, here, at import:
# what the server holds afterwards is the same template as before, byte for byte.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _read_web(root: Path, *parts: str) -> str:
    r"""Read one frontend file of `sorta/web/`.

    Text mode on purpose. The template is assembled with "\n" throughout, and a
    checkout that materialises these files with CRLF (the Windows default) must not
    change a single byte of what is served — universal newlines make that impossible.
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
    """Fills the chrome `{{key}}` placeholders and the `window.I18N` JSON (F33).

    Placeholders are literal `{{...}}` tokens, replaced via `str.replace` (not
    `.format`): the CSS/JS in the template is full of single `{`/`}`, which `.format`
    would interpret as substitution fields.
    """
    i18n_map = {key: _t(key, lang) for key in _UI_STRINGS}
    lang_options = "".join(
        f'<option value="{code}"{" selected" if code == lang else ""}>{name}</option>'
        for code, name in _LANG_SELF_NAMES.items()
    )
    html = _INDEX_HTML_TEMPLATE.replace("{{lang}}", lang)
    html = html.replace("{{lang_options}}", lang_options)
    # F80: how many frames the lightbox may page through. The real strip of a short
    # clip can be shorter — the pager finds that out from the first 404 and clamps.
    html = html.replace("{{video_frames}}", str(imaging.video_frames()))
    html = html.replace("{{i18n_json}}", json.dumps(i18n_map, ensure_ascii=False))
    for key, value in i18n_map.items():
        html = html.replace("{{" + key + "}}", value)
    return html
