# My Personal Blog

A simple, elegant personal blog built with HTML, CSS, and vanilla JavaScript. No build process required - works directly on GitHub Pages!

## Features

- **Classic/Elegant design** with beautiful serif typography
- **Markdown support** for writing posts
- **Tag system** for organizing content
- **Responsive design** that works on all devices
- **Zero build step** - just write Markdown and push

## How to Add a New Post

1. Create a new `.md` file in the `posts/` folder
2. Add front matter at the top:
   ```yaml
   ---
   title: "Your Post Title"
   date: "2025-04-27"
   tags: ["tag1", "tag2"]
   ---
   ```
3. Write your content in Markdown below the front matter
4. Open `js/blog.js` and add your post to the `posts` array:
   ```javascript
   {
       filename: 'your-post.md',
       title: 'Your Post Title',
       date: '2025-04-27',
       tags: ['tag1', 'tag2']
   }
   ```
5. Commit and push to GitHub
6. Your post will appear automatically!

## File Structure

```
├── index.html          # Homepage - lists all posts
├── post.html           # Template for individual posts
├── css/
│   └── style.css       # Blog styles
├── js/
│   └── blog.js         # Blog logic and post list
├── posts/              # Your blog posts (Markdown files)
│   └── hello-world.md  # Example post
└── README.md           # This file
```

## Customization

### Change the Blog Title

Edit `index.html` and `post.html`:
- Change `<title>` tag
- Change `.site-title` text
- Change `.site-description` text

### Change Colors/Fonts

Edit `css/style.css`:
- Colors are defined throughout the file
- Fonts are imported from Google Fonts at the top of HTML files

## Deploying to GitHub Pages

1. Push this repository to GitHub
2. Go to Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select "main" branch and "/ (root)" folder
5. Save - your blog will be live at `https://yourusername.github.io`

## Markdown Reference

This blog supports standard Markdown:

- Headers: `# H1`, `## H2`, `### H3`
- **Bold** and *italic* text
- [Links](url)
- Lists (ordered and unordered)
- `Inline code` and code blocks
- > Blockquotes
- Images: `![alt](url)`
- Horizontal rules: `---`

Happy blogging!
