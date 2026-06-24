---
title: "PixelRAG: How Does Image-Based Retrieval Compare to Text RAG?"
date: "2026-06-25"
tags: ["RAG", "embedding"]
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

...

## Benchmark

### documents

we prepare four groups of documents:
- html:
    - https://grayv.com
- easy-pdf:
    - https://github.com/py-pdf/pypdf/blob/main/resources/crazyones.pdf
    - https://github.com/py-pdf/sample-files/blob/main/001-trivial/minimal-document.pdf
- common-pdf:
    - https://arxiv.org/pdf/2201.00200
    - https://arxiv.org/pdf/2201.00214
- hard-pdf:
    - https://jefftan969.github.io/dasr/poster.pdf
    - https://doss.xhby.net/zpaper/xhrb/pc/att/202605/04/2dc24357-0c3c-47a8-88dc-0fb51b881d4b.pdf

### indexing

...

### querying

...

## Discussion

...