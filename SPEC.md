# Frontier Papers Site Spec

## Goal

Build a minimal, reliable Hugo site for publishing AI paper digest Markdown files.

The site is intended for simple reading by company leaders and internal users. The maintainer should only need to add Markdown files under `daily` and `weekly`; all indexes, listing pages, article pages, and publishing should be automated.

## Core Principles

- Markdown files are the source of truth.
- No frontmatter is required in content files.
- No manual index maintenance is required.
- Page titles are generated from folder type and filename date.
- The site should stay simple, readable, and low maintenance.
- GitHub Actions builds and publishes the site to GitHub Pages after push.

## Content Structure

Use this content layout:

```text
content/
  daily/
    2026-05-06.md
    2026-05-07.md
  weekly/
    2026-05-03.md
    2026-05-10.md
```

Only two content sections are required for the first version:

- `content/daily/`
- `content/weekly/`

## File Naming

Content files should use this format:

```text
YYYY-MM-DD.md
```

Examples:

```text
content/daily/2026-05-06.md
content/weekly/2026-05-03.md
```

The date in the filename is treated as the canonical date for sorting and display.

If a filename does not match `YYYY-MM-DD.md`, the site should still build. In that case, templates may fall back to the raw filename as the display date.

## Markdown Format

Each Markdown file should contain only the digest body copied from Grok or another source.

No frontmatter is required.

Valid example:

```md
# Weekly AI Paper Digest

## 1. Trend Summary

...
```

Do not require this:

```md
---
title: ...
date: ...
tags: ...
---
```

The rendered article page should use the system-generated title as the page title. Any heading inside the Markdown body should be treated as normal article content.

## Generated Titles

Article titles must be generated from the content directory and filename.

Rules:

```text
content/daily/2026-05-06.md
=> Daily · 2026-05-06

content/weekly/2026-05-03.md
=> Weekly · 2026-05-03
```

Do not parse titles from Markdown content. This keeps the site reliable even when Grok changes its heading format.

## Site Pages

The first version should include these pages:

```text
/
/daily/
/weekly/
/daily/2026-05-06/
/weekly/2026-05-03/
```

### Home Page

The home page should automatically show:

- Latest daily digests
- Latest weekly digests

Example:

```text
Latest Daily
- Daily · 2026-05-07
- Daily · 2026-05-06

Latest Weekly
- Weekly · 2026-05-10
- Weekly · 2026-05-03
```

The home page must not require manual updates.

### Daily List Page

`/daily/` should automatically list all files under `content/daily/`.

Sorting:

```text
Newest first, based on filename date.
```

### Weekly List Page

`/weekly/` should automatically list all files under `content/weekly/`.

Sorting:

```text
Newest first, based on filename date.
```

### Article Page

Each article page should:

- Show the generated title.
- Show a compact date/type line if useful.
- Render the Markdown body.
- Provide simple navigation back to the section list.

## Hugo Layout Scope

Use custom minimal Hugo layouts instead of a heavy theme.

Recommended files:

```text
layouts/
  _default/
    baseof.html
    single.html
    list.html
  index.html
```

The implementation should keep templates small and explicit.

## Visual Requirements

The site should feel like a clean internal briefing page, not a personal blog.

Requirements:

- Simple top navigation: `Home`, `Daily`, `Weekly`
- Readable typography for Chinese and English mixed content
- Comfortable article width
- Good desktop and mobile readability
- No decorative hero section
- No complex theme features
- No cards inside cards
- No unnecessary animations

## Reliability Requirements

The site should handle these cases:

- Content files without frontmatter
- Empty `daily` directory
- Empty `weekly` directory
- Markdown tables
- Markdown code blocks
- Chinese text
- Long article content
- Unexpected first-level headings in Markdown body
- Filename not matching `YYYY-MM-DD.md`

## Build And Deployment

Use GitHub Actions to build Hugo and deploy to GitHub Pages.

Expected workflow:

```text
git push to main
  -> GitHub Actions runs Hugo build
  -> generated site is deployed to GitHub Pages
```

Repository setting:

```text
Settings -> Pages -> Build and deployment -> Source: GitHub Actions
```

## Daily Maintenance Workflow

Maintainer workflow:

```text
1. Create a new Markdown file under content/daily or content/weekly.
2. Paste the Grok output directly into the file.
3. Commit and push.
4. GitHub Actions publishes the updated site.
```

The maintainer should not need to:

- Write frontmatter
- Edit index pages
- Update lists manually
- Maintain a database
- Use an admin backend
- Manually build the site

## First Version Scope

Implement only:

- Minimal Hugo config
- Minimal layouts
- Home page with latest daily and weekly links
- Daily list page
- Weekly list page
- Article rendering
- GitHub Pages deployment workflow
- Basic README with maintenance instructions

Do not implement in the first version:

- Search
- Tags
- Login
- CMS
- Database
- Comments
- RSS
- Complex taxonomy

## Optional Later Enhancements

Possible future additions:

- Full-text search
- Tags or topics
- RSS feed
- File name validation in CI
- Control character cleanup before build
- Custom domain
- VPS deployment mirror

