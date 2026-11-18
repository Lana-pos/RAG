# Mandatory Part: `src/chunking/markdown.py`

## 1. Overview & Architectural Role

[`src/chunking/markdown.py`](../src/chunking/markdown.py) implements the **Heading-Aware Markdown Chunker** required by Chapter VI.1.

Technical documentation in `vLLM` is structured using Markdown headers (`#`, `##`, `###`). Slicing Markdown by raw character counts damages document context and cuts through code fences (` ``` `). `MarkdownChunker` segments documentation hierarchically along headers while keeping paragraphs and code blocks intact.

---

## 2. Heading Detection & Code Fence Protection

### 2.1 Code Fence Protection (`find_code_fence_intervals`)
Python code blocks in Markdown often contain Python comments starting with `#` (e.g. `# TODO`, `# Set configuration`).
To ensure these comments are never mistaken for Markdown headers:
1. `find_code_fence_intervals(text)` scans the document for opening and closing code fences (both backticks ```` ``` ```` and tildes `~~~`).
2. It returns a list of character intervals `[(start, end), ...]`.
3. Any `#` occurring inside an active code fence is strictly ignored.

### 2.2 ATX Heading Detection
The chunker scans lines outside code fences using the regex:
```python
_ATX_HEADING_PATTERN = re.compile(r"^[ ]{0,3}(#{1,6})(?:[ \t]+.*|[ \t]*)$")
```
It identifies section start offsets for `#` through `######`.

---

## 3. Section Segmentation & Oversized Splitting

### 3.1 Contiguous Section Spans
1. **Preamble:** Any text before the first header is grouped into an introductory section.
2. **Sections:** Each section spans from its heading up to the character immediately before the next heading:
   $$[\text{offset}_i, \text{offset}_{i+1})$$

### 3.2 Splitting Oversized Sections (`_subdivide_oversized_section`)
If a section exceeds `--max_chunk_size` (2000 characters):
1. **Never fracture a code fence:** If the 2000-character boundary falls in the middle of a code fence, the chunker attempts to split *before* the code fence at the preceding paragraph break (`\n\n`).
2. **Paragraph Breaks:** If no code fence is cut, it splits at the latest double newline (`\n\n`).
3. **Line Breaks:** If no paragraph break exists, it splits at a single newline (`\n`).
4. **Exact Offsets:** Every sub-chunk maintains 0-based character offsets (`first_character_index`, `last_character_index`) corresponding to the raw file on disk.
