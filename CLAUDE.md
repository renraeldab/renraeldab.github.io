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
│   └── style.css       # Blog styles
├── js/
│   └── blog.js         # Blog logic and post registry
├── posts/              # Blog posts (each in its own folder)
│   └── hello-world/    # Example post folder
│       ├── index.md    # Post content (MUST be named index.md)
│       └── [assets]    # Images and other files
└── README.md           # Visitor-facing description
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

## Content Review Checklist

When reviewing a new post:

- [ ] Post folder is in `posts/[slug]/`
- [ ] Post file is named `index.md`
- [ ] Front matter includes title, date, and optionally tags
- [ ] Post is registered in `js/blog.js` with matching slug
- [ ] Images use relative paths: `./image.png` (not `image.png` or `/posts/...`)
- [ ] Images are in the same folder as index.md
- [ ] Writing is clear and free of obvious errors
- [ ] Date format is YYYY-MM-DD

## Markdown Conventions

- Use YAML front matter with `---` delimiters
- Reference local images with `./filename.png`
- Standard GitHub-flavored markdown is supported
- Code blocks should specify language for syntax highlighting

## Common Tasks

### Adding a New Post

1. Check the new post folder exists in `posts/`
2. Verify `index.md` has proper front matter
3. Ensure post is registered in `js/blog.js`
4. Review content quality
5. Check images render correctly
6. Commit and push

### Modifying an Existing Post

1. Review changes
2. Ensure slug/folder structure remains intact
3. Commit and push

### Adding Website Features

- Discuss approach with user before implementing
- Consider impact on existing posts
- Update this CLAUDE.md if conventions change
- Update README.md if user-facing behavior changes

## Git Workflow

After content review is complete:

1. Stage changes: `git add -A`
2. Commit with descriptive message:
   ```
   Add post: [Post Title]

   Brief description of changes.

   Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
   ```
3. Push: `git push`

## Technical Notes

- Uses `marked` library (CDN) for markdown rendering
- Front matter is parsed with custom regex in `parseFrontMatter()`
- Posts are sorted by date (newest first) on homepage
- URLs use query parameter: `post.html?post=hello-world`
- `.nojekyll` disables Jekyll processing so raw `.md` files can be fetched

## Customization Reference

### Change the Blog Title / Subtitle

Edit `index.html` and `post.html`:
- Change `<title>` tag
- Change `.site-title` text
- Change `.site-description` text

### Change Colors / Fonts

Edit `css/style.css`:
- Colors are defined throughout the file
- Fonts are imported from Google Fonts at the top of HTML files