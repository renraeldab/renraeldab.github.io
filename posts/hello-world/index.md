---
title: "Hello World"
date: "2025-04-27"
tags: ["introduction", "first-post"]
---

Welcome to my new blog! This is my first post.

## About This Blog

I've built this blog using a simple, lightweight system that works directly on GitHub Pages. No complex build process, no dependencies - just write Markdown and publish.

## How to Add Posts

To add a new blog post:

1. Create a new folder in the `posts/` directory (e.g., `posts/my-post/`)
2. Create `index.md` inside with front matter (title, date, tags)
3. Write your content in Markdown below the front matter
4. **Add images or files** to the same folder and reference them with `./`:
   ```markdown
   ![My Image](./screenshot.png)
   ```
5. Add the post to the `posts` array in `js/blog.js` using the folder name as `slug`
6. Commit and push to GitHub

## Markdown Features

This blog supports all standard Markdown features:

- **Bold text** and *italic text*
- [Links](https://github.com)
- Lists (like this one!)
- > Blockquotes

### Code Examples

Inline code: `console.log('Hello')`

Code blocks:
```javascript
function hello() {
    console.log('Hello, world!');
}
```

## Thanks for Reading!

Stay tuned for more posts. Happy blogging!
