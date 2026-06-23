---
title: ""
date: "2026-06-"
tags: []
---

## API

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

The text query successfully retrieved [the correct Wikipedia article](https://en.wikipedia.org/wiki/Tasmania) as the top result with a score of 0.625.

### Fetching a Result Tile

**Input:**

```bash
curl -s https://pixelrag.ai/api/tile/7234200/0/0 --output result_tile_tasmania.png
```

**Output:**

![](result_tile_tasmania.png)

The result is a PNG image file (875 x 1024 pixels) containing a screenshot tile of the Tasmania Wikipedia page. This confirms that the API returns actual rendered page screenshots as retrievable units.

### Image Query

**Input:**

```bash
curl -s -X POST https://pixelrag.ai/api/search \
  -H "Content-Type: application/json" \
  -d "{\"queries\": [{\"image\": \"$(base64 < tasmania.png | tr -d '\n')\"}], \"n_docs\": 3}"
```

![](tasmania.png)

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