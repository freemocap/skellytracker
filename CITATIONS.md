# Citation policy

Register external sources in root `references.bib` and cite stable keys beside supported claims using bracketed at-key markers. Entries use braced BibTeX values and require title, author/editor, URL, access date, review scope and claim. Limited review requires a note. Record only verified metadata; access date is not publication date.

Review scopes: metadata, abstract, excerpt, full-text, source-code. Distinguish source inspection from execution and measurement. Keep source/model/dependency license statements separate.

Run from this repository:

```text
python -B tools/check_references.py
python -B -m unittest discover -s tools -p test_references.py -v
```

The offline validator and in-memory regression tests are adopted from the workspace's FreeMoCap core tools as independent local copies. They validate BibTeX metadata and unresolved citations; they cannot establish claim truth or URL availability. They do not import another subproject. Generated reports belong in ignored `.test-artifacts/`; the commands above write only to the terminal.

