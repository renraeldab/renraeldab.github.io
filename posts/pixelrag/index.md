---
title: "PixelRAG: How Does Image-Based Retrieval Compare to Text RAG?"
date: "2026-06-25"
tags: ["RAG", "embedding", "VLM"]
---

[PixelRAG](https://github.com/StarTrail-org/PixelRAG) is a project that renders documents as screenshots and retrieves over 
the images directly instead of parsing text. One claim is that this approach preserves visual structure, tables, charts, page 
layout, that text-based RAG may throw away, and that it can be more token-efficient. In this post, we dig in: try the public API, 
build an index locally, and compare it against a traditional text-based RAG pipeline.

## What is PixelRAG?

PixelRAG is a retrieval system that takes a different approach to document indexing. Instead of extracting text from web pages 
or PDFs, it renders them as screenshot tiles and embeds those images directly. The key insight is that images carry semantic 
information that can get lost when you convert everything to plain text.

Two pieces make this work:

1. **Rendering pipeline**: Converts documents (web pages, PDFs, images) into screenshot tiles.
2. **Vision-language embedding model**: `Qwen/Qwen3-VL-Embedding-2B`, embeds these page images into a retrievable vector space.

## Trying the Public API

PixelRAG hosts a public API at `pixelrag.ai`. Here is a few quick queries to see how it behaves.

### Text Query

**Input:**

```bash
curl -s -X POST https://pixelrag.ai/api/search \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"text": "What is the capital of Tasmania?"}], "n_docs": 3}'
```

**Output:**

```json
{
    "results": [
        {
            "hits": [
                {
                    "score": 0.624868631362915,
                    "vector_id": 23310349,
                    "article_id": 7234200,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 1024,
                    "path": "shard_873/shard_00002/7234200.png.tiles/chunk_0000_00.png",
                    "url": "Tasmania",
                    "article_pages": "0:0-7,1:0-7,2:0-7,3:0-7,4:0-7,5:0-6",
                    "image_base64": null
                },
                {
                    "score": 0.5778354406356812,
                    "vector_id": 9818959,
                    "article_id": 2938301,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 1024,
                    "path": "shard_354/shard_00011/2938301.png.tiles/chunk_0000_00.png",
                    "url": "Geography_of_Tasmania",
                    "article_pages": "0:0-7,1:0-5",
                    "image_base64": null
                },
                {
                    "score": 0.5738669633865356,
                    "vector_id": 23310359,
                    "article_id": 7234200,
                    "tile_index": 1,
                    "chunk_index": 2,
                    "y_offset": 2048,
                    "tile_height": 1024,
                    "path": "shard_873/shard_00002/7234200.png.tiles/chunk_0001_02.png",
                    "url": "Tasmania",
                    "article_pages": "0:0-7,1:0-7,2:0-7,3:0-7,4:0-7,5:0-6",
                    "image_base64": null
                }
            ]
        }
    ]
}
```

The text query successfully retrieved [the correct Wikipedia article](https://en.wikipedia.org/wiki/Tasmania) as the top result 
with a score of 0.625.

### Fetching a Result Tile

**Input:**

```bash
curl -s https://pixelrag.ai/api/tile/7234200/0/0 --output result_tile_tasmania.png
```

**Output:**

![](result_tile_tasmania.png)

The result is a PNG image file (875 x 1024 pixels) containing a screenshot tile of the Tasmania Wikipedia page. This confirms 
that the API returns actual rendered page screenshots as retrievable units.

### Image Query

Querying with an image is also possible. Let's use a screenshot of the Tasmania Wikipedia page instead of text.

![](tasmania.png)

**Input:**

```bash
curl -s -X POST https://pixelrag.ai/api/search \
  -H "Content-Type: application/json" \
  -d "{\"queries\": [{\"image\": \"$(base64 < tasmania.png | tr -d '\n')\"}], \"n_docs\": 3}"
```

**Output:**

```json
{
    "results": [
        {
            "hits": [
                {
                    "score": 0.5702254772186279,
                    "vector_id": 6649190,
                    "article_id": 1859601,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 1024,
                    "path": "shard_224/shard_00002/1859601.png.tiles/chunk_0000_00.png",
                    "url": "Coat_of_arms_of_New_Zealand",
                    "article_pages": "0:0-4",
                    "image_base64": null
                },
                {
                    "score": 0.5504581928253174,
                    "vector_id": 9219754,
                    "article_id": 2736954,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 1024,
                    "path": "shard_330/shard_00010/2736954.png.tiles/chunk_0000_00.png",
                    "url": "Flag_of_the_British_Antarctic_Territory",
                    "article_pages": "0:0-3",
                    "image_base64": null
                },
                {
                    "score": 0.5503813028335571,
                    "vector_id": 17981114,
                    "article_id": 5357499,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 1024,
                    "path": "shard_646/shard_00006/5357499.png.tiles/chunk_0000_00.png",
                    "url": "National_symbols_of_Singapore",
                    "article_pages": "0:0-3",
                    "image_base64": null
                }
            ]
        }
    ]
}
```

The image query did not return the Tasmania article. Instead, it retrieved visually similar heraldic symbols.

![](result_tile_incorrect.png)

## Building a PixelRAG Index Locally

> [!NOTE]
> The package version is 0.3.0 at the time of writing, which has a bug while using `source.type: local`. `LocalSource` uses 
> the raw file stem as the document ID, but `build()` later assumes every document ID is an integer. If your filenames are 
> not valid integers, the pipeline crashes with `ValueError: invalid literal for int() with base 10`.

Here is the wrapper to build the index:

```python
import shutil
from pathlib import Path

from pixelrag_index.pipelines import build


def _stage_source(source_path: Path, staging_dir: Path) -> Path:
    """Copy files from source_path into a subdir of staging_dir with integer names."""
    stage = staging_dir / source_path.name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in source_path.rglob("*") if f.is_file())
    for i, f in enumerate(files):
        dest = stage / f"{i:04d}{f.suffix}"
        shutil.copy2(f, dest)
        print(f"  Staged {f.name} -> {dest.name}")

    return stage


def build_index(
    source_path: Path,
    output_path: Path,
    *,
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    device: str = "cpu",
    stage: bool = False,
    staging_dir: Path | None = None,
) -> dict:
    """Build a PixelRAG index for a single source folder.

    Args:
        source_path: Path to the folder containing source documents.
        output_path: Path where the built index should be written.
        model: HuggingFace model name for embeddings.
        device: Device to run embeddings on.
        stage: If True, temporarily copy files to a staging directory with
            integer names before indexing, then clean up afterwards.
        staging_dir: Parent directory for temporary staging. Required if
            stage=True.

    Returns:
        Dict with 'status' ('ok' or 'error') and either 'output' or 'error'.
    """
    if not source_path.exists():
        return {"status": "error", "error": f"Source not found: {source_path}"}

    if stage:
        if staging_dir is None:
            raise ValueError("staging_dir is required when stage=True")
        build_source = _stage_source(source_path, staging_dir)
    else:
        build_source = source_path

    print(f"\n{'=' * 60}")
    print(f"Building index for: {source_path.name}")
    print(f"  Source : {build_source}")
    print(f"  Output : {output_path}")
    print(f"{'=' * 60}\n")

    config = {
        "source": {
            "type": "local",
            "path": str(build_source),
        },
        "embed": {
            "model": model,
            "device": device,
        },
        "output": str(output_path),
    }

    try:
        build(config, force=True)
        result = {"status": "ok", "output": str(output_path)}
    except Exception as e:
        print(f"ERROR building index for {source_path.name}: {e}")
        result = {"status": "error", "error": str(e)}
    finally:
        if stage and staging_dir is not None:
            stage_path = staging_dir / source_path.name
            if stage_path.exists():
                shutil.rmtree(stage_path)
                print(f"  Cleaned up staging dir: {stage_path}")

    return result
```

### Pipeline Concepts

PixelRAG's pipeline has four stages: **render → chunk → embed → index**. The relationship between tiles, chunks, and vectors 
depends on the document type.

#### For HTML / Web Pages

1. **Render**: A headless browser captures the page as screenshot tiles. Each tile is a vertical strip up to **8192 px tall** at the browser's **viewport width of 875 px**.
2. **Chunk**: The chunking stage splits tall tiles into smaller pieces the embedding model can ingest. The model expects images around 1024 px tall, so each tile is sliced into **1024 px horizontal strips** (with tiny tails < 28 px merged into the previous strip).
3. **Embed**: Each chunk becomes one vector.

#### For PDFs

1. **Render**: `pdf2image` (Poppler) renders each PDF page to a JPEG at **200 DPI**. Each page becomes its own tile.
2. **Chunk**: Each PDF page is treated as exactly one chunk because a page is considered a natural semantic unit.
3. **Embed**: Each page (page = tile = chunk) becomes one vector.

## Building a Traditional RAG Index Locally

For a fair comparison, we should build a standard text-based RAG index on the same documents. In many ways, taking screenshots 
is the easy solution; extracting and chunking text well is tricky. Let's stick to a basic implementation, and advanced tricks are 
out of scope for this post.

### Parsing Documents

#### HTML

For HTML, we extract plain text and throw away everything else.

```python
from pathlib import Path

from bs4 import BeautifulSoup


def parse_html_text_only(html_path: Path) -> str:
    """Extract plain text only, no structure preservation."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator=" ", strip=True)
```

#### PDF

For PDFs, let's try two approaches. The simple one uses `pypdf` for basic text extraction:

```python
from pathlib import Path

from pypdf import PdfReader


def parse_pdf_default(pdf_path: Path) -> str:
    """Extract text from PDF using pypdf."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)
```

The other approach is to use the [MinerU](https://mineru.net) API, which converts PDFs to structured markdown with the help of VLM:

```python
import os
import io
import time
import zipfile
from pathlib import Path

import requests

MINERU_BASE_URL = "https://mineru.net/api/v4"


def _mineru_headers() -> dict:
    token = os.getenv("MINERU_API_KEY", "")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def call_mineru_api(pdf_path: Path) -> str:
    """Upload a PDF to MinerU, wait for extraction, and return markdown text."""
    headers = _mineru_headers()

    # 1. Request upload URLs
    data = {
        "files": [{"name": pdf_path.name, "data_id": pdf_path.stem}],
        "model_version": "vlm",
    }
    resp = requests.post(
        f"{MINERU_BASE_URL}/file-urls/batch", headers=headers, json=data, timeout=30
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"MinerU upload URL request failed: {result.get('msg')}")

    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"]

    # 2. Upload file
    with open(pdf_path, "rb") as f:
        up_resp = requests.put(urls[0], data=f, timeout=60)
    up_resp.raise_for_status()

    # 3. Poll for results
    max_retries = 120
    for i in range(max_retries):
        time.sleep(3)
        res_resp = requests.get(
            f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=30,
        )
        res_resp.raise_for_status()
        res_data = res_resp.json()
        if res_data.get("code") != 0:
            raise RuntimeError(f"MinerU result fetch failed: {res_data.get('msg')}")

        # Actual API returns extract_result array with state/done
        items = res_data.get("data", {}).get("extract_result", [])
        if not items:
            continue
        state = items[0].get("state")
        if state == "done":
            zip_url = items[0].get("full_zip_url", "")
            if not zip_url:
                raise RuntimeError(f"MinerU result missing zip URL for {pdf_path.name}")
            # Download ZIP and extract full.md
            zip_resp = requests.get(zip_url, timeout=60)
            zip_resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                with zf.open("full.md") as md_file:
                    return md_file.read().decode("utf-8")
        elif state == "failed" or items[0].get("err_msg"):
            raise RuntimeError(
                f"MinerU extraction failed for {pdf_path.name}: {items[0].get('err_msg')}"
            )
        # else: pending / processing — keep polling

    raise TimeoutError(f"MinerU extraction timed out for {pdf_path.name}")


def parse_pdf_mineru(pdf_path: Path) -> str:
    """Wrapper with clearer error on auth failure."""
    try:
        return call_mineru_api(pdf_path)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            raise RuntimeError(
                "MinerU API authentication failed (401). Please check your MINERU_API_KEY."
            ) from e
        raise
```

### Chunking

Use LangChain's `RecursiveCharacterTextSplitter` for simple chunking:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)
```

Choosing chunking hyperparameters is harder than it looks. First of all, embedding models have limited input length, so chunks 
cannot be too large. But if chunks are too small, they may lack context. Overlap helps avoid cutting things in half, yet too 
much overlap creates redundancy: your top-5 retrieved chunks might all contain the same sentence. Finally, naive splitting ignores 
document structure. A chunk that starts in the middle of a table or breaks a heading from its paragraph can be semantically useless. 
We gloss over these issues here.

### Embedding

For a fair comparison, we use the same model that PixelRAG uses: `Qwen/Qwen3-VL-Embedding-2B`. The implementation below is 
adapted from PixelRAG's code. It follows the standard recipe for LLM-based embeddings: format each input as a chat message, 
run it through the model, take the hidden state of the last token, and L2-normalize the result.

```python
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_EMBED_MODEL: tuple | None = None
_EMBED_MODEL_NAME: str | None = None


def _load_embed_model(model_name: str):
    """Load (and cache) the embedding model + processor."""
    global _EMBED_MODEL, _EMBED_MODEL_NAME
    if _EMBED_MODEL is not None and _EMBED_MODEL_NAME == model_name:
        return _EMBED_MODEL

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"Loading embedding model: {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()
    _EMBED_MODEL = (model, processor, device)
    _EMBED_MODEL_NAME = model_name
    print(f"Model loaded on {device}.")
    return _EMBED_MODEL


def embed_texts(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    *,
    instruction: str = "Represent the user's input.",
) -> np.ndarray:
    """Embed a list of texts using Qwen3-VL-Embedding (or compatible model)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model, processor, device = _load_embed_model(model_name)

    messages_batch = [
        [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [{"type": "text", "text": t}]},
        ]
        for t in texts
    ]
    prompts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]

    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        last_token_indices = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device), last_token_indices
        ]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

    return pooled.cpu().float().numpy()
```

One practical advantage of LLM-based embedding models is that they accept much longer inputs than traditional embedding models, 
which often cap context at 512 tokens.

### Index Building

We store embeddings in a FAISS `IndexFlatIP` index. Because the vectors are L2-normalized, inner product equals cosine similarity, 
which is what both PixelRAG and our baseline use for retrieval. The pipeline below ties together parsing, chunking, embedding, and 
indexing into one function.

```python
import os
import json
import warnings
from typing import Literal
from pathlib import Path

import numpy as np
import faiss


def _collect_documents(
    folder: Path,
    pdf_mode: Literal["default", "mineru"] | None = None
) -> list[dict]:
    """Walk folder and return list of dicts with keys: source, text."""
    docs: list[dict] = []
    files = sorted(p for p in folder.rglob("*") if p.is_file())

    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix == ".html" or suffix == ".htm":
                text = parse_html_text_only(f)
            elif suffix == ".pdf":
                mode = pdf_mode or "default"
                if mode == "default":
                    text = parse_pdf_default(f)
                else:
                    text = parse_pdf_mineru(f)
            else:
                continue

            if text.strip():
                docs.append({"source": str(f.relative_to(folder)), "text": text})
        except Exception as exc:
            warnings.warn(f"Failed to parse {f}: {exc}")

    return docs


def build_index(
    folder_path: str | Path,
    output_dir: str | Path,
    *,
    embed_model: str = "Qwen/Qwen3-VL-Embedding-2B",
    pdf_mode: Literal["default", "mineru"] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> dict:
    """Build a FAISS index for documents in folder_path."""
    folder = Path(folder_path).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Source folder not found: {folder}")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse documents
    docs = _collect_documents(folder, pdf_mode=pdf_mode)
    print(f"Parsed {len(docs)} documents.")

    # 2. Chunk
    chunks: list[dict] = []
    for doc in docs:
        for piece in chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append({"source": doc["source"], "text": piece})
    print(f"Generated {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No text chunks produced. Check source folder and parsers.")

    # 3. Embed in batches
    batch_size = 16  # small batch to keep memory reasonable on MPS/CPU
    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i : i + batch_size]]
        embs = embed_texts(batch_texts, embed_model)
        all_embeddings.append(embs)
        print(f"  Embedded batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    embeddings = np.vstack(all_embeddings)
    embed_dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Build FAISS index (inner product on L2-normalized vectors = cosine)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)

    # 5. Save
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    config = {
        "embed_model": embed_model,
        "pdf_mode": pdf_mode,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "embed_dim": embed_dim,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nIndex saved to {output_dir}")
    return {
        "output_dir": str(output_dir),
        "num_chunks": len(chunks),
        "embed_dim": embed_dim,
    }
```

## Benchmark

These two approaches should behave differently depending on the document. We test four document groups, ranging from simple 
HTML to complex PDFs.

> [!NOTE]
> This is a small-scale comparison, not a comprehensive benchmark. The goal is to observe qualitative differences rather 
> than prove quantitative superiority.

We evaluate **indexing cost**, **querying quality**, and **answer-generation token cost**. All answer generation uses 
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), and `top_k=3`.

### HTML

A simple page with very little text: https://grayv.com

#### Indexing

| Approach | Chunks (Vectors) | Index File | Total Directory |
|----------|------------------|------------|-----------------|
| PixelRAG | 8 | 74 KB | 3.8 MB |
| RAG | 2 | 16 KB | 28 KB |

On indexing cost, PixelRAG loses here. The HTML renders into a long image that is sliced into 8 tiles, 3 of which are blank 
but still take ~5 KB each. The content is so sparse that the 8 images carry little semantic information. Content splitting 
across tiles also persists: text chunking problems reappear as image chunking problems.

#### Querying

**How many times does "9.00pm" appear?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | No | Three chunks together can yield the correct answer, but only one is retrieved | 2656 |
| RAG | Yes | Duplicate chunks from overlap confused the model, leading to wrong reasoning | 457 |

**Who is at Bowery Ballroom?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | - | 2651 |
| RAG | Yes | - | 452 |

This HTML page is a poor fit for PixelRAG. For simple documents and purely text-based questions, image indexing is expensive 
and underperforms: it misses chunks that text splitting captures easily, and the prompt token cost is nearly 6× higher.

### Simple PDF

Two small PDFs with plain text and simple layout.

- https://github.com/py-pdf/pypdf/blob/main/resources/crazyones.pdf
- https://github.com/py-pdf/sample-files/blob/main/001-trivial/minimal-document.pdf

#### Indexing

| Approach | Chunks (Vectors) | Index File | Total Directory |
|----------|------------------|------------|-----------------|
| PixelRAG | 2 | 25 KB | 396 KB |
| RAG (Default) | 4 | 32 KB | 44 KB |
| RAG (MinerU) | 4 | 32 KB | 44 KB |

As documents grow slightly more text-dense, PixelRAG now produces fewer vectors than the text baselines.

#### Querying

**Are the crazy ones bad?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | - | 5083 |
| RAG (Default) | Yes | - | 345 |
| RAG (MinerU) | Yes | - | 340 |

**What are the real Latin words?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | - | 5084 |
| RAG (Default) | Yes | - | 392 |
| RAG (MinerU) | Yes | - | 386 |

All three approaches retrieve the correct chunks, but PixelRAG costs an order of magnitude more in prompt tokens because 
each chunk is a full image.

### Common PDF

Two arXiv papers with text, equations, figures, and standard multi-page layout.

- https://arxiv.org/pdf/2201.00200
- https://arxiv.org/pdf/2201.00214

#### Indexing

| Approach | Chunks (Vectors) | Index File | Total Directory |
|----------|------------------|------------|-----------------|
| PixelRAG | 29 | 246 KB | 15 MB |
| RAG (Default) | 199 | 1.6 MB | 1.7 MB |
| RAG (MinerU) | 253 | 2.1 MB | 2.1 MB |

At this scale, PixelRAG's vector count is far lower: a single page becomes one vector, whereas text chunking explodes into 
hundreds. The trade-off is storage: 15 MB of images versus ~0.1 MB of text.

#### Querying

**Why are the evolutionary models self-consistent?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | - | 7620 |
| RAG (Default) | No | Poor parsing produced incomplete content even after chunk overlap | 470 |
| RAG (MinerU) | Yes | - | 245 |

**In the histogram of the temperature-period percentages for the loops’ strips of the ﬂaring and non-ﬂaring ARs, what is the main temperature period for non-ﬂaring ARs?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | - | 7621 |
| RAG (Default) | No | The answer is in a figure, which text-based indexing ignores | 671 |
| RAG (MinerU) | Yes | The correct figure is located, but the answer cannot be inferred from markdown image placeholder text | 375 |

This is where PixelRAG starts to shine. On documents with figures, equations, and dense layout, retrieving whole pages as images 
preserves visual context that text parsing loses. The trade-off is still token cost.

One secondary finding is that MinerU becomes valuable for traditional RAG as documents grow complex. It produced better retrieval 
and lower token costs than default PDF parsing on both questions.

### Complex PDF

A dense CVPR poster and a newspaper page. Both are single-page PDFs with complex visual layout.

- https://jefftan969.github.io/dasr/poster.pdf
- https://doss.xhby.net/zpaper/xhrb/pc/att/202605/04/2dc24357-0c3c-47a8-88dc-0fb51b881d4b.pdf

#### Indexing

| Approach | Chunks (Vectors) | Index File | Total Directory |
|----------|------------------|------------|-----------------|
| PixelRAG | 2 | 25 KB | 11 MB |
| RAG (Default) | 22 | 180 KB | 208 KB |
| RAG (MinerU) | 27 | 221 KB | 248 KB |

PixelRAG stores 11 MB for just 2 chunks because the source PDFs render to very large images. The newspaper page in particular 
becomes a 16800 x 8400 px image that compresses poorly to JPEG; PIL even emitted a `DecompressionBombWarning` during rendering.

#### Querying

**Is BANMo the slowest method?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | - | 4625 |
| RAG (Default) | No | No decisive information is retrieved | 556 |
| RAG (MinerU) | No | The markdown table is split across chunks, so only a partial table is retrieved | 702 |

**What is the name of the reporter who took the picture of "苏超有面"?**

| Approach | Necessary Chunks Retrieved? | Failure Reason | Prompt Tokens |
|----------|-----------------------------|----------------|---------------|
| PixelRAG | Yes | The model failed to read the correct answer from the compressed image | 4635 |
| RAG (Default) | No | Retrieved information was insufficient because of poor parsing and chunking | 831 |
| RAG (MinerU) | Yes | - | 1057 |

On the poster, PixelRAG demonstrates its main strength: there is no struggle with parsing tables or tuning chunk size. A single 
rendered page carries the full layout, and retrieval succeeds where text-based methods return nothing or fragmented tables.

The newspaper page tells a different story. When an entire dense page becomes one high-resolution image, you face a hard choice: 
compress the image and lose fine details, or keep the resolution and pay even more tokens, with no guarantee that the VLM can 
find the details. Text-based RAG does not have this problem — once MinerU parses the layout correctly, it can retrieve the exact 
text at a fraction of the token cost.

## Discussion

### How We Got Here

RAG emerged because early language models were expensive and had limited context windows. A whole industry of tricks grew around 
the idea of feeding models only the most relevant snippets. Three areas have evolved dramatically since then.

**Parsing.** The traditional approach pipelined CV models for layout detection and OCR. Today, VLMs, like the one behind MinerU, 
do the same job with higher accuracy at higher cost.

**Embedding.** LLM-based embedding models now swallow far longer inputs than the 512-token caps of traditional embedders. Multimodal 
embeddings, the foundation of PixelRAG, are also practical now.

**Generation.** Perhaps the biggest shift is that LLMs are now cheap enough, and context windows large enough, that "just dump 
everything" is no longer absurd. PixelRAG pushes this logic one step further: instead of using a VLM at indexing time to parse 
and structure documents, it moves the VLM to query time, reading abundant information from images, and skips parsing entirely. 
The cost shifts from indexing to querying.

### When to Use Which

#### Quality

Based on the tests above, the answer to "which is better?" is firmly "it depends":

- **Simple text, clean layout**: Text-based RAG wins. PixelRAG costs more with no quality gain.
- **Complex layout with figures and tables**: PixelRAG wins. It preserves visual structure that text parsing destroys.
- **Extremely dense, high-resolution pages**: Text-based RAG wins again. The problem of "not enough context window" reappears in image form.

#### Cost

**Indexing**

PixelRAG skips parsing entirely, so there are no chunk size or overlap parameters to tune. For complex documents, its vector 
count is often much smaller than text-based RAG because one page equals one vector. The trade-off is storage: image tiles are 
always larger than the text they replace.

**Querying**

In our experiments, PixelRAG always used significantly more prompt tokens per query because each retrieved chunk is a full image, 
while the retrieved text is short (this may change if you have to increase `top_k` to hit the right text chunk). Retrieval itself 
is cheaper with a smaller index, but that is negligible compared to generation cost. If you query frequently, the token premium adds up fast.

**Overall**

PixelRAG makes sense when you index many documents but query them infrequently. The savings at indexing time eventually pay 
for the expensive queries. If your workload is query-heavy, text-based RAG is cheaper in the long run.

#### Other Considerations

Similarity scores differ in scale. Traditional RAG often returns chunks above 0.8 cosine similarity; PixelRAG rarely exceeds 0.6. 
If your application surfaces scores to users, expect to explain the difference or hide them entirely.

Retrieval latency is another factor. A smaller index can mean faster nearest-neighbor search, so if your text-based index has grown 
unwieldy, PixelRAG's compact vector count may help.

Ultimately, the choice depends on your documents, your query pattern, and what failure mode you would rather manage.