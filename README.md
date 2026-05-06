# Frontier Papers

Minimal Hugo site for publishing AI paper digests from Markdown files.

## Local Development

```sh
hugo server
```

Then open the local URL printed by Hugo.

To build the static site:

```sh
hugo
```

The generated files are written to `public/`.

## Add A Digest

Create a Markdown file in one of these directories:

```text
content/daily/
content/weekly/
```

Use the filename format:

```text
YYYY-MM-DD.md
```

Examples:

```text
content/daily/2026-05-06.md
content/weekly/2026-05-03.md
```

Paste the digest body directly into the file. Do not add frontmatter.

Article titles are generated automatically from the folder and filename:

```text
Daily · 2026-05-06
Weekly · 2026-05-03
```

## Deployment

The workflow at `.github/workflows/pages.yml` builds the Hugo site and deploys it to GitHub Pages whenever changes are pushed to `main`.

In the GitHub repository settings, configure:

```text
Settings -> Pages -> Build and deployment -> Source: GitHub Actions
```

After that, the normal maintenance flow is:

```text
1. Add a Markdown file under content/daily or content/weekly.
2. Commit and push to main.
3. GitHub Actions publishes the updated site.
```
