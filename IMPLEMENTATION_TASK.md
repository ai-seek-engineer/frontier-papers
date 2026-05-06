# Implementation Task

Implement the minimal Hugo site described in `SPEC.md`.

## Objective

Create a low-maintenance GitHub Pages site where the maintainer only adds Markdown files under:

```text
content/daily/
content/weekly/
```

The site must automatically generate the home page, daily list, weekly list, and article pages.

## Required Deliverables

- `hugo.toml`
- Minimal Hugo layouts
- `content/daily/` and `content/weekly/` directories
- Example Markdown files for local verification
- GitHub Actions workflow for GitHub Pages deployment
- `README.md` with simple maintenance and deployment instructions

## Key Rules

- Do not require frontmatter in Markdown files.
- Generate display titles from path and filename.
- Use titles like `Daily · 2026-05-06` and `Weekly · 2026-05-03`.
- Sort daily and weekly entries by filename date, newest first.
- Keep the design minimal and readable.
- Do not add search, tags, login, database, CMS, or a heavy theme in the first version.

## Acceptance Criteria

- `hugo server` runs locally.
- `hugo` builds successfully.
- Home page shows latest daily and weekly entries automatically.
- `/daily/` lists all daily Markdown files automatically.
- `/weekly/` lists all weekly Markdown files automatically.
- Article pages render Markdown content without frontmatter.
- GitHub Actions can deploy to GitHub Pages.
- Adding a new file under `content/daily/` or `content/weekly/` requires no other manual page edits.

