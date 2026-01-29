# Mapping Docs

This folder will hold Excel mapping documents generated from the profiling outputs.

Planned structure:
- `mapping/output/` for generated ES→BBF field mapping workbooks
- `mapping/templates/` if we add manual review templates

## LLM Mapping (Ollama)

Generate a mapping workbook from a profile output:

```
python3 tools/generate_mappings_ollama.py \
  --profile profiles/output/Account__to__Account__profile_20260128_190025.xlsx \
  --out-dir mapping/output \
  --model llama3.1
```

Notes:
- Set `OLLAMA_MODEL` and `OLLAMA_URL` env vars to avoid passing flags each time.
- Output workbook replaces `profile` with `mapping` in the filename.
- ES fields with `Null %` > 90 are excluded from candidates by default (override with `--es-null-threshold`).
- Reference fields are ignored by default (use `--include-reference` to override).
- BBF fields with `Null %` <= 0 are skipped by default (override with `--skip-bbf-null-at-or-below`).
- Fields matching geocode name patterns are skipped by default (use `--exclude-name-pattern` to override).

## Column legend (mappings sheet)
- `BBF Field API`: Target BBF field API name.
- `BBF Label`: Target BBF field label.
- `BBF Type`: Target BBF field data type.
- `BBF Null %`: Percent of sampled BBF records with null/blank values.
- `BBF Top Values`: Most common BBF values in the sample (with counts).
- `BBF Sample Values`: A few distinct BBF sample values.
- `BBF Picklist Values`: Allowed BBF picklist values (if applicable).
- `ES Field API`: Suggested source ES field API name (or `NO_MATCH`).
- `ES Label`: Source ES field label.
- `ES Type`: Source ES field data type.
- `ES Null %`: Percent of sampled ES records with null/blank values.
- `ES Top Values`: Most common ES values in the sample (with counts).
- `ES Sample Values`: A few distinct ES sample values.
- `ES Picklist Values`: Allowed ES picklist values (if applicable).
- `Confidence`: LLM confidence score (0–100).
- `Transform`: Suggested transform or mapping logic (blank if direct).
- `Reasoning`: Short explanation for why the match was chosen.
- `Picklist Mapping`: `REQUIRED` when BBF field is a picklist/multipicklist.
- `Decision`: Accept/Reject/Review (manual decision).
- `Decision`: Auto-marked as `Yes (auto)` when confidence >= 80 and no transform needed.
- `Owner`: Person responsible for the decision.
- `Notes`: Freeform notes.

## picklist_mappings sheet (template)
- `BBF Field API`: Target picklist field.
- `ES Field API`: Source field.
- `ES Value`: ES picklist value (or observed sample).
- `BBF Value`: BBF picklist value (fill in).
- `Match Type`: `exact` when auto-matched (case-insensitive).
- `ES Value Source`: `picklist` or `sample`.
- `Status`: `auto` or `needs_review`.
- `Notes`: Freeform notes.

## transform_library sheet
- Pre-seeded transform patterns for reuse.

## required_fields_check sheet
- Lists required BBF fields (Nillable = false, Createable = true), and whether a mapping exists.
