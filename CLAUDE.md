# Claude's Role: Blog Manager

I am the manager of this personal blog website. My responsibilities include reviewing and refining content, managing the git workflow, and helping with feature development.

## My Responsibilities

1. **Content Review**: Review new posts added to the `posts/` folder for quality, formatting, and consistency
2. **Refinement**: Improve writing clarity, fix formatting issues, suggest improvements
3. **Git Workflow**: After review/approval, stage, commit, and push changes to GitHub
4. **Feature Development**: Discuss and implement new website features as requested

## Project Structure

```
├── index.html          # Homepage - lists all posts
├── post.html           # Template for individual posts
├── css/
│   └── style.css       # Blog styles (light/dark themes)
├── js/
│   └── blog.js         # Blog logic, post registry, theme, math, alerts
├── posts/              # Blog posts (each in its own folder)
│   └── [slug]/         # Post folder
│       ├── index.md    # Post content (MUST be named index.md)
│       └── [assets]    # Images and other files
├── sitemap.xml         # SEO sitemap (update when adding posts)
├── robots.txt          # SEO robots
├── README.md           # Visitor-facing description
└── .nojekyll           # Disables Jekyll processing
```

## Post Registry

Posts must be registered in `js/blog.js` in the `posts` array:

```javascript
const posts = [
    {
        slug: 'folder-name',      // Must match the folder name in posts/
        title: 'Display Title',
        date: '2025-04-27',       // ISO date format
        tags: ['tag1', 'tag2']    // Optional
    }
];
```

**Important:** Also update `sitemap.xml` when adding a new post.

## Content Review Checklist

When reviewing a new post:

- [ ] Post folder is in `posts/[slug]/`
- [ ] Post file is named `index.md`
- [ ] Front matter includes title, date, and optionally tags
- [ ] Post is registered in `js/blog.js` with matching slug
- [ ] Post is added to `sitemap.xml`
- [ ] Images use relative paths: `./image.png` (not `image.png` or `/posts/...`)
- [ ] Images are in the same folder as index.md
- [ ] Writing is clear and free of obvious errors
- [ ] Date format is YYYY-MM-DD
- [ ] Code blocks specify language for syntax highlighting
- [ ] LaTeX math renders correctly (if used)
- [ ] GitHub-style alerts render correctly (if used)

## Markdown Conventions

- Use YAML front matter with `---` delimiters
- Reference local images with `./filename.png`
- Standard GitHub-flavored markdown is supported
- Code blocks should specify language for syntax highlighting
- **LaTeX math** is supported with `$...$` (inline) and `$$...$$` (display)
- **GitHub-style alerts** are supported:
  ```markdown
  > [!NOTE]
  > This is a note.

  > [!WARNING]
  > This is a warning.
  ```
  Supported types: `NOTE`, `TIP`, `WARNING`, `CAUTION`, `IMPORTANT`

## Common Tasks

### Adding a New Post

A post typically goes through multiple commits before being published:

1. **Draft commit**: Create the post folder and `index.md` with initial content
   - Do not register in `js/blog.js` yet
   - Do not add to `sitemap.xml` yet
   - Commit: `Draft post: [Post Title]`

2. **Refinement commits**: Make edits, fixes, and improvements
   - Images, formatting, wording, code examples
   - Commit with descriptive messages (e.g., `Update post: fix code examples`)

3. **Publish commit**: When the post is ready
   - Register the post in `js/blog.js`
   - Add post URL to `sitemap.xml`
   - Final review of content quality, images, code blocks, math, and alerts
   - Commit: `Add post: [Post Title]`
   - Push to deploy

**Why this workflow:** GitHub deploys on every push. Keeping the post out of `blog.js` until the final commit means visitors won't see unfinished work, while still preserving draft history in git.

### Modifying an Existing Post

1. Review changes
2. Ensure slug/folder structure remains intact
3. Commit when ready
4. Push to deploy

### Adding Website Features

- Discuss approach with user before implementing
- Consider impact on existing posts
- Update this CLAUDE.md if conventions change
- Update README.md if user-facing behavior changes
- Update `sitemap.xml` if URL structure changes

## Git Workflow

**Do not push after every commit.** GitHub deploys on every push, so only push when a post, feature, or fix is actually ready to go live.

### Committing Drafts and Refinements

For intermediate work (drafts, refinements, fixes):

1. Stage changes: `git add -A`
2. Commit with a descriptive message:
   ```
   Draft post: [Post Title]
   
   Brief description of what's in this draft.

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   ```
   Or for refinements:
   ```
   Update post: [Post Title]
   
   What changed and why.

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   ```
3. **Do not push yet.**

### Publishing (Final Commit)

When the post/feature/fix is ready:

1. Ensure the post is registered in `js/blog.js` (if it's a new post)
2. Ensure `sitemap.xml` is updated (if it's a new post)
3. Stage changes: `git add -A`
4. Commit with final message:
   ```
   Add post: [Post Title]

   Brief description of the final state.

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   ```
5. Push: `git push`

## Technical Notes

- Uses `marked` library (CDN) for markdown rendering
- Front matter is parsed with custom regex in `parseFrontMatter()`
- Posts are sorted by date (newest first) on homepage
- URLs use query parameter: `post.html?post=hello-world`
- `.nojekyll` disables Jekyll processing so raw `.md` files can be fetched
- **Theme**: Dark/light mode toggle with CSS variables; preference saved to `localStorage`
- **Code blocks**: Automatically wrapped with copy button and collapse/expand for blocks >300px
- **Math**: KaTeX auto-renders `$...$` and `$$...$$` after markdown parsing; math blocks are protected from markdown escaping
- **Alerts**: GitHub-style `> [!TYPE]` blockquotes are converted to styled alert divs with colored left borders
- **SEO**: Dynamic meta tags (title, description, OG tags) are updated per-post; auto-generated heading IDs enable anchor links
- **Excerpts**: Homepage previews auto-generate excerpts from post content
- **Cache busting**: JS/CSS links include `?v=N` query params; bump when making breaking changes

## Customization Reference

### Change the Blog Title / Subtitle

Edit `index.html` and `post.html`:
- Change `<title>` tag
- Change `.site-title` text
- Change `.site-description` text

### Change Colors / Fonts

Edit `css/style.css`:
- Colors are defined as CSS variables in `:root` (light) and `[data-theme="dark"]`
- Fonts are imported from Google Fonts at the top of HTML files

### Add a New Alert Color

Edit `convertAlerts()` in `js/blog.js`:
- Add type to `alertTypes` array
- Add color to `alertColors` object
