// Blog configuration
const POSTS_DIRECTORY = '/posts/';

// List of posts - add your posts here
const posts = [
    {
        slug: 'shadowrocket',
        title: 'Setting Up a Shared Proxy with Shadowrocket',
        date: '2026-05-20',
        tags: ['shadowrocket', 'proxy', 'vpn']
    },
    {
        slug: 'gmm_em',
        title: 'GMM and EM',
        date: '2026-06-04',
        tags: ['machine-learning']
    },
    {
        slug: 'concurrency_python',
        title: 'Concurrency in Python',
        date: '2026-06-10',
        tags: ['python', 'concurrency', 'async', 'threading', 'multiprocessing']
    }
];

/**
 * Parse front matter from markdown content
 * Returns { frontMatter: object, content: string }
 */
function parseFrontMatter(text) {
    const frontMatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
    const match = text.match(frontMatterRegex);

    if (!match) {
        return { frontMatter: {}, content: text };
    }

    const frontMatterText = match[1];
    const content = match[2];

    // Parse YAML-like front matter
    const frontMatter = {};
    frontMatterText.split('\n').forEach(line => {
        const colonIndex = line.indexOf(':');
        if (colonIndex > 0) {
            const key = line.substring(0, colonIndex).trim();
            let value = line.substring(colonIndex + 1).trim();

            // Remove quotes if present
            if ((value.startsWith('"') && value.endsWith('"')) ||
                (value.startsWith("'") && value.endsWith("'"))) {
                value = value.slice(1, -1);
            }

            // Parse arrays (tags: [tag1, tag2])
            if (value.startsWith('[') && value.endsWith(']')) {
                value = value.slice(1, -1).split(',').map(s => s.trim().replace(/^[\"']|[\"']$/g, ''));
            }

            frontMatter[key] = value;
        }
    });

    return { frontMatter, content };
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

/**
 * Create excerpt from content
 */
function createExcerpt(content, maxLength = 200) {
    // Remove markdown formatting for excerpt
    const plainText = content
        .replace(/#+ /g, '')
        .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/`/g, '')
        .replace(/\n/g, ' ')
        .trim();

    if (plainText.length <= maxLength) {
        return plainText;
    }

    return plainText.substring(0, maxLength).trim() + '...';
}

/**
 * Load and display the list of posts on the homepage
 */
async function loadPostsList() {
    const container = document.getElementById('posts-list');

    try {
        // Sort posts by date (newest first)
        const sortedPosts = [...posts].sort((a, b) => new Date(b.date) - new Date(a.date));

        if (sortedPosts.length === 0) {
            container.innerHTML = '<p class="loading">No posts yet. Add a markdown file to the posts folder!</p>';
            return;
        }

        // Generate HTML for each post preview
        const postsHtml = await Promise.all(sortedPosts.map(async (post) => {
            // Try to load post to get excerpt
            let excerpt = '';
            try {
                const response = await fetch(`${POSTS_DIRECTORY}${post.slug}/index.md`);
                if (response.ok) {
                    const text = await response.text();
                    const { content } = parseFrontMatter(text);
                    excerpt = createExcerpt(content);
                }
            } catch (e) {
                // Silently fail - we'll just show no excerpt
            }

            const tagsHtml = post.tags && post.tags.length > 0
                ? `<span class="post-preview-tags">${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}</span>`
                : '';

            return `
                <article class="post-preview">
                    <h2 class="post-preview-title">
                        <a href="post.html?post=${encodeURIComponent(post.slug)}">${escapeHtml(post.title)}</a>
                    </h2>
                    <div class="post-preview-meta">
                        <span class="post-preview-date">${formatDate(post.date)}</span>
                        ${tagsHtml}
                    </div>
                    ${excerpt ? `<p class="post-preview-excerpt">${escapeHtml(excerpt)}</p>` : ''}
                </article>
            `;
        }));

        container.innerHTML = postsHtml.join('');

    } catch (error) {
        container.innerHTML = `<p class="error">Error loading posts: ${escapeHtml(error.message)}</p>`;
    }
}

/**
 * Extract LaTeX math blocks from markdown so marked doesn't mangle backslashes.
 * Returns { text: string with placeholders, blocks: Array<{placeholder, math}> }
 */
function extractMathBlocks(text) {
    const blocks = [];
    let id = 0;

    function makePlaceholder() {
        return `<span data-math="${id++}"></span>`;
    }

    // Extract display math ($$...$$) first so inline regex doesn't match inside them
    let result = text.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
        const p = makePlaceholder();
        blocks.push({ placeholder: p, math: match });
        return p;
    });

    // Then extract inline math ($...$)
    result = result.replace(/\$([^\$\n]+?)\$/g, (match) => {
        const p = makePlaceholder();
        blocks.push({ placeholder: p, math: match });
        return p;
    });

    return { text: result, blocks };
}

/**
 * Convert GitHub-style markdown alerts (> [!NOTE], > [!WARNING], etc.)
 * from regular blockquotes into styled alert divs.
 */
function convertAlerts(html) {
    const alertTypes = ['NOTE', 'TIP', 'WARNING', 'CAUTION', 'IMPORTANT'];
    const alertColors = {
        NOTE: '#0969da',
        TIP: '#1a7f37',
        WARNING: '#9a6700',
        CAUTION: '#cf222e',
        IMPORTANT: '#8250df'
    };
    const typePattern = alertTypes.join('|');

    // Pattern 1: Alert title and content in the same <p>
    let result = html.replace(
        new RegExp(`<blockquote>\\s*<p>\\[!(${typePattern})\\]\\s+(.*?)</p>\\s*</blockquote>`, 'gs'),
        (match, type, content) => {
            const color = alertColors[type] || '#0969da';
            return `<div class="markdown-alert" style="border-left-color: ${color};"><p><strong>${type}:</strong> ${content}</p></div>`;
        }
    );

    // Pattern 2: Alert title in its own <p>, followed by more paragraphs
    result = result.replace(
        new RegExp(`<blockquote>\\s*<p>\\[!(${typePattern})\\]\\s*</p>((?:\\s*<p>.*?</p>)+)\\s*</blockquote>`, 'gs'),
        (match, type, content) => {
            const color = alertColors[type] || '#0969da';
            return `<div class="markdown-alert" style="border-left-color: ${color};"><p><strong>${type}:</strong></p>${content}</div>`;
        }
    );

    return result;
}

/**
 * Restore LaTeX math blocks into HTML after markdown parsing.
 */
function restoreMathBlocks(html, blocks) {
    let result = html;
    for (const { placeholder, math } of blocks) {
        result = result.replace(placeholder, math);
    }
    return result;
}

/**
 * Load and display an individual post
 */
async function loadPost() {
    const container = document.getElementById('post-content');

    // Get post slug from URL query parameter
    const urlParams = new URLSearchParams(window.location.search);
    const postSlug = urlParams.get('post');

    if (!postSlug) {
        container.innerHTML = '<p class="error">No post specified. <a href="/">Go back home</a></p>';
        return;
    }

    try {
        const response = await fetch(`${POSTS_DIRECTORY}${postSlug}/index.md`);

        if (!response.ok) {
            throw new Error('Post not found');
        }

        const text = await response.text();
        const { frontMatter, content } = parseFrontMatter(text);

        // Rewrite relative image paths to be absolute to the post directory
        const postBaseUrl = `${POSTS_DIRECTORY}${postSlug}/`;
        const processedContent = content.replace(
            /!\[([^\]]*)\]\(([^)]+)\)/g,
            (match, alt, url) => {
                if (/^(https?:\/\/|\/)/.test(url)) {
                    return match;
                }
                return `![${alt}](${postBaseUrl}${url})`;
            }
        );

        // Get post metadata from front matter or posts list
        const postInfo = posts.find(p => p.slug === postSlug) || {};
        const title = frontMatter.title || postInfo.title || 'Untitled';
        const date = frontMatter.date || postInfo.date || '';
        const tags = frontMatter.tags || postInfo.tags || [];

        // Protect math blocks from markdown parsing
        const { text: markdownWithPlaceholders, blocks: mathBlocks } = extractMathBlocks(processedContent);

        // Convert markdown to HTML
        let htmlContent = marked.parse(markdownWithPlaceholders);

        // Convert GitHub-style alerts from plain blockquotes to styled blocks
        htmlContent = convertAlerts(htmlContent);

        // Restore raw math blocks so KaTeX sees the original LaTeX
        htmlContent = restoreMathBlocks(htmlContent, mathBlocks);

        // Generate tags HTML
        const tagsHtml = tags && tags.length > 0
            ? `<span class="post-tags">${tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</span>`
            : '';

        container.innerHTML = `
            <header class="post-header">
                <h1 class="post-title">${escapeHtml(title)}</h1>
                <div class="post-meta">
                    <span class="post-date">${date ? formatDate(date) : ''}</span>
                    ${tagsHtml}
                </div>
            </header>
            <div class="post-content">
                ${htmlContent}
            </div>
        `;

        // Update page title
        document.title = `${title} - My Blog`;

        // Ensure all headings have IDs so anchor links work
        document.querySelectorAll('.post-content h1, .post-content h2, .post-content h3, .post-content h4, .post-content h5, .post-content h6').forEach(heading => {
            if (!heading.id) {
                heading.id = heading.textContent.toLowerCase().replace(/[^\w]+/g, '-').replace(/^-|-$/g, '');
            }
        });

        // Render LaTeX math with KaTeX if available
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(document.querySelector('.post-content'), {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false
            });
        }

        // Scroll to anchor if present in URL (needed because content loads dynamically)
        if (window.location.hash) {
            const el = document.getElementById(window.location.hash.slice(1));
            if (el) {
                el.scrollIntoView();
            }
        }

    } catch (error) {
        container.innerHTML = `<p class="error">Error loading post: ${escapeHtml(error.message)}</p>`;
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
