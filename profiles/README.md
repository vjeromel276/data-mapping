# Subset Profiling

This folder contains config and output files for profiling ES/BBF fields using the **subset** defined in `profiles/profile_config.json`.

## Prereqs
- Python 3
- `simple_salesforce`, `pandas`, `openpyxl`
- ES + BBF UAT credentials in env vars (no secrets in repo)

### Required env vars
```
export ES_USERNAME="..."
export ES_PASSWORD="..."
export ES_TOKEN="..."
export ES_DOMAIN="test"

export BBF_USERNAME="..."
export BBF_PASSWORD="..."
export BBF_TOKEN="..."
export BBF_DOMAIN="test"
```

## Run the profiler
```
python3 tools/profile_subset.py \
  --config profiles/profile_config.json \
  --out-dir profiles/output
```

### Optional flags
- `--object Account` (repeatable) to only profile specific object pairs
- `--sample-size 300` to override the default sample size

## Output
- Excel workbooks per object pair, written to `profiles/output/`
- Sheets: `summary`, `es_fields`, `bbf_fields`, `es_excluded`, `bbf_excluded`

## Notes
- Subset is order-driven and BAN-driven (see `profile_config.json`)
- Fields excluded from profiling include long/rich text, system fields, and fields already populated in the initial move (configurable)
- Formula fields are included only if their return type is `boolean` or `string` (configurable)
