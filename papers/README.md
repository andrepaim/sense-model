# Local paper inputs

Place your PDFs here. They are ignored by Git. Copy the repository's
`papers.example.json` to `papers.json` and replace the example paths with your
actual files. Paths are relative to the manifest, not the command's working directory.

Minimal entry:

```json
{"path": "papers/my-paper.pdf", "language": "en"}
```

Use `en` or `pt`; optional fields include title, area, source URL and license.
List each paper once. No papers are bundled with this template.
