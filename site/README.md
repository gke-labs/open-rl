# openrl.dev

The OpenRL website: one landing page and a blog. Built with Hugo, deployed to GitHub Pages by `.github/workflows/site.yml`. The plan is design doc 014 in `docs/designs`.

## Preview

Hugo extended v0.165.0 or later.

```
cd site
make preview        # http://localhost:1313/, drafts included
make build          # writes public/
```

`make deck` copies `docs/openrl-deck.html` to `/deck/`; the workflow does the same before every build.

## Add a post

1. `mkdir content/blog/<slug>` and write `content/blog/<slug>/index.md`. Put images beside it and reference them by file name.
2. Front matter:

   ```yaml
   ---
   title: "Post title"
   date: 2026-09-18
   authors: ["Your Name"]
   summary: "One sentence shown in the list and in link previews."
   tags: ["tutorial"]
   draft: false
   ---
   ```

3. `make preview`, read it at `/blog/<slug>/`, open a PR. Pull requests build the site as a check; merging to `main` deploys it.

## Layout

- `layouts/home.html`, `assets/css/landing.css`, `assets/js/landscape.js`: the landing page. Its figures and copy come from the deck.
- `layouts/blog/`, `layouts/baseof.html`, `layouts/_partials/`, `assets/css/site.css`: shared shell and the blog.
- `static/fonts/`: Figtree, under the SIL Open Font License (`OFL.txt`).

## Domains

`baseURL` is `https://openrl.dev/`; the workflow overrides it with the Pages URL of whichever repository runs it, so a fork previews at `https://<user>.github.io/open-rl/`. The `CNAME` file is added when the domain is pointed at Pages.
