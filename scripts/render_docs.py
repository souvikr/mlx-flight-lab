#!/usr/bin/env python3
"""
render_docs.py - turn the Markdown docs into HTML pages you can read offline in Safari,
and build LIBRARY.html, an index of everything in docs/offline/.

  python scripts/render_docs.py      (setup.sh runs this for you)
  open README.html LIBRARY.html
"""
import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OFF = ROOT / "docs" / "offline"

CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --border:#d1d9e0; --code:#f6f8fa; --link:#0969da; --accent:#fff8c5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --border:#3d444d; --code:#151b23; --link:#4493f8; --accent:#3a3000; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 24px 16px 80px; }
a { color: var(--link); }
h1, h2, h3 { line-height: 1.25; margin-top: 1.6em; }
h1, h2 { border-bottom: 1px solid var(--border); padding-bottom: .3em; }
code { background: var(--code); padding: .15em .35em; border-radius: 6px; font: 85% ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background: var(--code); padding: 14px 16px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); }
pre code { background: none; padding: 0; font-size: 13.5px; line-height: 1.45; }
table { border-collapse: collapse; display: block; overflow-x: auto; margin: 1em 0; }
th, td { border: 1px solid var(--border); padding: 6px 12px; vertical-align: top; }
th { background: var(--code); }
blockquote { margin: 1em 0; padding: .2em 1em; color: var(--muted); border-left: 4px solid var(--border); }
nav.crumbs { font-size: 14px; color: var(--muted); }
"""


def page(title, body, depth):
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main><nav class="crumbs"><a href="{up}README.html">Guide</a> · <a href="{up}LIBRARY.html">Library</a></nav>
{body}</main></body></html>"""


def md_to_html(src: Path, dst: Path):
    text = src.read_text(encoding="utf-8", errors="ignore")
    body = markdown.markdown(text, extensions=["fenced_code", "tables", "toc", "sane_lists"])
    # point local links at the rendered .html versions
    body = re.sub(r'href="(?!https?:|#)([^"#]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("#")), src.stem)
    depth = len(dst.relative_to(ROOT).parts) - 1
    dst.write_text(page(title, body, depth), encoding="utf-8")
    return dst


def link(path: Path, label=None):
    rel = path.relative_to(ROOT).as_posix()
    return f'<li><a href="{rel}">{html.escape(label or path.name)}</a></li>'


def main():
    made = [md_to_html(ROOT / "README.md", ROOT / "README.html")]
    for md in [ROOT / "docs" / "CHEATSHEET.md", *sorted(OFF.rglob("*.md"))]:
        if md.exists() and "/mlx/" not in md.as_posix():
            made.append(md_to_html(md, md.with_suffix(".html")))

    sections = [("This repo", [link(ROOT / "README.html", "README — the full guide"),
                               link(ROOT / "docs" / "CHEATSHEET.html", "Cheat sheet — every command")])]
    mlx = OFF / "mlx"
    if (mlx / "index.html").exists():
        pages = [("index.html", "MLX documentation (home)"),
                 ("usage/quick_start.html", "Quick start"),
                 ("usage/lazy_evaluation.html", "Lazy evaluation"),
                 ("usage/unified_memory.html", "Unified memory"),
                 ("usage/function_transforms.html", "Function transforms (grad, vmap)"),
                 ("usage/compile.html", "Compilation"),
                 ("usage/indexing.html", "Indexing arrays"),
                 ("usage/saving_and_loading.html", "Saving and loading"),
                 ("examples/linear_regression.html", "Example: linear regression"),
                 ("examples/mlp.html", "Example: multi-layer perceptron"),
                 ("examples/llama-inference.html", "Example: LLM inference"),
                 ("python/nn.html", "API: mlx.nn"),
                 ("python/optimizers.html", "API: mlx.optimizers"),
                 ("search.html", "Search the MLX docs")]
        sections.append(("MLX (the framework)", [link(mlx / p, label) for p, label in pages if (mlx / p).exists()]))
    groups = [("mlx-lm (the LLM toolkit)", sorted((OFF / "mlx-lm").glob("*.html"))),
              ("mlx-lm official examples", sorted((OFF / "mlx-lm" / "examples").glob("*"))),
              ("Papers (PDF)", sorted((OFF / "papers").glob("*.pdf"))),
              ("Other", sorted(OFF.glob("*.html"))),
              ("--help of every command", sorted((OFF / "cli-help").glob("*.txt")))]
    for title, files in groups:
        if files:
            sections.append((title, [link(f) for f in files]))

    body = "<h1>Offline library</h1>\n<p>Everything here works without internet.</p>\n"
    body += "".join(f"<h2>{html.escape(t)}</h2>\n<ul>\n{''.join(items)}\n</ul>\n" for t, items in sections)
    if not OFF.exists():
        body += "<p><em>docs/offline/ is missing — run <code>./setup.sh</code> while online.</em></p>"
    (ROOT / "LIBRARY.html").write_text(page("Offline library", body, 0), encoding="utf-8")
    print(f"rendered {len(made)} Markdown files to HTML + LIBRARY.html  ->  open README.html LIBRARY.html")


if __name__ == "__main__":
    main()
