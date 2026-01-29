#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import urllib.request


DEFAULT_EXCLUDE_NAME_PATTERNS = [
    r"(?i).*GeocodeAccuracy$",
    r"(?i)^(Billing|Shipping|Mailing|Other)?Latitude$",
    r"(?i)^(Billing|Shipping|Mailing|Other)?Longitude$",
]

@dataclass
class FieldRow:
    api: str
    label: str
    ftype: str
    length: Any
    nillable: Any
    createable: Any
    updateable: Any
    calculated: Any
    picklist_values: str
    null_pct: Any
    top_values: str
    sample_values: str


def read_profile(path: str) -> Tuple[List[FieldRow], List[FieldRow], Dict[str, Any]]:
    xls = pd.ExcelFile(path)
    es_df = pd.read_excel(xls, sheet_name="es_fields")
    bbf_df = pd.read_excel(xls, sheet_name="bbf_fields")
    summary = pd.read_excel(xls, sheet_name="summary")
    summary_dict = {row["Metric"]: row["Value"] for _, row in summary.iterrows()}

    def to_fields(df: pd.DataFrame) -> List[FieldRow]:
        fields = []
        for _, r in df.iterrows():
            fields.append(
                FieldRow(
                    api=str(r.get("Field API Name", "")),
                    label=str(r.get("Field Label", "")),
                    ftype=str(r.get("Field Type", "")),
                    length=r.get("Length"),
                    nillable=r.get("Nillable"),
                    createable=r.get("Createable"),
                    updateable=r.get("Updateable"),
                    calculated=r.get("Calculated"),
                    picklist_values=str(r.get("Picklist Values", "")),
                    null_pct=r.get("Null %"),
                    top_values=str(r.get("Top Values", "")),
                    sample_values=str(r.get("Sample Values", "")),
                )
            )
        return fields

    return to_fields(es_df), to_fields(bbf_df), summary_dict


def type_group(ftype: str) -> str:
    t = ftype.lower()
    if t in ("string", "textarea", "phone", "email", "url", "encryptedstring", "id", "reference"):
        return "string"
    if t in ("picklist", "multipicklist"):
        return "picklist"
    if t in ("boolean",):
        return "boolean"
    if t in ("double", "int", "currency", "percent"):
        return "number"
    if t in ("date", "datetime"):
        return "date"
    return "other"


def compatible(es_type: str, bbf_type: str) -> bool:
    es_g = type_group(es_type)
    bbf_g = type_group(bbf_type)
    if es_g == bbf_g:
        return True
    # Allow picklist to string or string to picklist (LLM can decide)
    if (es_g == "picklist" and bbf_g == "string") or (es_g == "string" and bbf_g == "picklist"):
        return True
    # Allow number to string (sometimes text fields store numeric identifiers)
    if es_g == "number" and bbf_g == "string":
        return True
    # Allow date to string (text fields that store dates)
    if es_g == "date" and bbf_g == "string":
        return True
    return False


def label_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def value_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aset = set(v.strip().lower() for v in a.split(";") if v.strip())
    bset = set(v.strip().lower() for v in b.split(";") if v.strip())
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(len(aset), len(bset))


def score_candidate(es: FieldRow, bbf: FieldRow) -> float:
    if not compatible(es.ftype, bbf.ftype):
        return -1.0
    score = 0.0
    score += label_similarity(es.label, bbf.label) * 0.6
    score += value_overlap(es.top_values, bbf.top_values) * 0.3
    score += value_overlap(es.sample_values, bbf.sample_values) * 0.1
    return score


def null_pct_value(v: Any) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def name_matches_patterns(name: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, name or ""):
                return True
        except re.error:
            continue
    return False


def split_values(raw: Any) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    parts = [p.strip() for p in text.split(";")]
    return [p for p in parts if p]


def parse_top_values(raw: Any) -> List[str]:
    values = []
    for item in split_values(raw):
        values.append(re.sub(r"\s*\\(\\d+\\)$", "", item).strip())
    return [v for v in values if v]


def top_candidates(
    es_fields: List[FieldRow],
    bbf: FieldRow,
    limit: int,
    es_null_threshold: Optional[float],
    ignore_reference: bool,
    exclude_name_patterns: List[str],
) -> List[FieldRow]:
    scored = []
    for es in es_fields:
        if exclude_name_patterns and name_matches_patterns(es.api, exclude_name_patterns):
            continue
        if ignore_reference and str(es.ftype).lower() == "reference":
            continue
        es_null = null_pct_value(es.null_pct)
        if es_null_threshold is not None and es_null is not None and es_null > es_null_threshold:
            continue
        s = score_candidate(es, bbf)
        if s >= 0:
            scored.append((s, es))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [es for _, es in scored[:limit]]


def build_prompt(bbf: FieldRow, candidates: List[FieldRow]) -> str:
    def field_block(prefix: str, f: FieldRow) -> str:
        return (
            f"{prefix} Field API: {f.api}\n"
            f"Label: {f.label}\n"
            f"Type: {f.ftype}\n"
            f"Null %: {f.null_pct}\n"
            f"Top Values: {f.top_values}\n"
            f"Sample Values: {f.sample_values}\n"
            f"Picklist Values: {f.picklist_values}\n"
            f"Calculated: {f.calculated}\n"
        )

    header = (
        "You are mapping ES (source) fields to BBF (target) fields.\n"
        "Choose the single best ES field that matches the BBF field based on data usage, \n"
        "not API name similarity. If no good match exists, return NO_MATCH.\n"
        "Use formula fields if they fit (read-only is fine).\n"
        "Output strict JSON with keys: best_match, confidence, transform, reasoning.\n"
        "Confidence is 0-100. Transform can be empty string if not needed.\n\n"
    )

    prompt = header + "TARGET (BBF):\n" + field_block("BBF", bbf) + "\nCANDIDATES (ES):\n"
    for idx, es in enumerate(candidates, 1):
        prompt += f"\nCandidate {idx}:\n" + field_block("ES", es)
    return prompt


def call_ollama(prompt: str, model: str, url: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj.get("response", "")


def parse_json_response(text: str) -> Dict[str, Any]:
    # Try to find a JSON object in the response
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"best_match": "NO_MATCH", "confidence": 0, "transform": "", "reasoning": "No JSON"}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"best_match": "NO_MATCH", "confidence": 0, "transform": "", "reasoning": "Invalid JSON"}


def find_es_field(es_fields: List[FieldRow], api_name: str) -> Optional[FieldRow]:
    for f in es_fields:
        if f.api == api_name:
            return f
    return None


def picklist_required(bbf: FieldRow) -> bool:
    return str(bbf.ftype).lower() in ("picklist", "multipicklist")


def to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "1"):
        return True
    if text in ("false", "f", "no", "n", "0"):
        return False
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ES→BBF mappings using Ollama.")
    parser.add_argument("--profile", required=True, help="Profile workbook (.xlsx)")
    parser.add_argument("--out-dir", default="mapping/output", help="Output directory")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", ""), help="Ollama model name")
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
        help="Ollama generate endpoint",
    )
    parser.add_argument("--candidate-limit", type=int, default=8, help="Max ES candidates")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--limit-bbf", type=int, default=None, help="Limit BBF fields")
    parser.add_argument(
        "--skip-bbf-null-at-or-below",
        type=float,
        default=0.0,
        help="Skip BBF fields with Null % at or below this value (default: 0.0)",
    )
    parser.add_argument(
        "--es-null-threshold",
        type=float,
        default=90.0,
        help="Exclude ES fields with Null % above this threshold (default: 90)",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        default=False,
        help="Include reference fields for mapping (default: false)",
    )
    parser.add_argument(
        "--exclude-name-pattern",
        action="append",
        default=None,
        help="Regex pattern for field API names to exclude (repeatable)",
    )

    args = parser.parse_args()
    if not args.model:
        raise RuntimeError("Missing --model or OLLAMA_MODEL")

    es_fields, bbf_fields, summary = read_profile(args.profile)

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    skip_reasons: Dict[str, str] = {}

    exclude_name_patterns = (
        args.exclude_name_pattern if args.exclude_name_pattern else DEFAULT_EXCLUDE_NAME_PATTERNS
    )

    for idx, bbf in enumerate(bbf_fields, 1):
        if args.limit_bbf and idx > args.limit_bbf:
            break
        if exclude_name_patterns and name_matches_patterns(bbf.api, exclude_name_patterns):
            skip_reasons[bbf.api] = "Skipped: name pattern match"
            continue
        bbf_null = null_pct_value(bbf.null_pct)
        if (
            bbf_null is not None
            and args.skip_bbf_null_at_or_below is not None
            and bbf_null <= args.skip_bbf_null_at_or_below
        ):
            skip_reasons[bbf.api] = (
                f"Skipped: BBF Null % <= {args.skip_bbf_null_at_or_below}"
            )
            continue

        ignore_reference = not args.include_reference
        if ignore_reference and str(bbf.ftype).lower() == "reference":
            parsed = {
                "best_match": "NO_MATCH",
                "confidence": 0,
                "transform": "",
                "reasoning": "BBF reference field ignored",
            }
            skip_reasons[bbf.api] = "Ignored: BBF reference field"
        else:
            candidates = top_candidates(
                es_fields,
                bbf,
                args.candidate_limit,
                args.es_null_threshold,
                ignore_reference,
                exclude_name_patterns,
            )
            if not candidates:
                parsed = {
                    "best_match": "NO_MATCH",
                    "confidence": 0,
                    "transform": "",
                    "reasoning": "No ES candidates after null%/reference filter",
                }
            else:
                prompt = build_prompt(bbf, candidates)
                response = call_ollama(prompt, args.model, args.ollama_url, args.temperature)
                parsed = parse_json_response(response)

        best = parsed.get("best_match", "NO_MATCH")
        confidence = parsed.get("confidence", 0)
        transform = parsed.get("transform", "")
        reasoning = parsed.get("reasoning", "")

        es_match = find_es_field(es_fields, best)

        auto_decision = ""
        try:
            conf_val = float(confidence)
        except Exception:
            conf_val = 0
        if best != "NO_MATCH" and conf_val >= 80 and (not str(transform).strip()):
            auto_decision = "Yes (auto)"

        results.append(
            {
                "BBF Field API": bbf.api,
                "BBF Label": bbf.label,
                "BBF Type": bbf.ftype,
                "BBF Null %": bbf.null_pct,
                "BBF Top Values": bbf.top_values,
                "BBF Sample Values": bbf.sample_values,
                "BBF Picklist Values": bbf.picklist_values,
                "ES Field API": es_match.api if es_match else best,
                "ES Label": es_match.label if es_match else "",
                "ES Type": es_match.ftype if es_match else "",
                "ES Null %": es_match.null_pct if es_match else "",
                "ES Top Values": es_match.top_values if es_match else "",
                "ES Sample Values": es_match.sample_values if es_match else "",
                "ES Picklist Values": es_match.picklist_values if es_match else "",
                "Confidence": confidence,
                "Transform": transform,
                "Reasoning": reasoning,
                "Picklist Mapping": "REQUIRED" if picklist_required(bbf) else "",
                "Decision": auto_decision,
                "Owner": "",
                "Notes": "auto>=80_no_transform" if auto_decision else "",
            }
        )

    out_name = os.path.basename(args.profile).replace("profile", "mapping")
    out_path = os.path.join(args.out_dir, out_name)

    summary_rows = [{"Metric": k, "Value": v} for k, v in summary.items()]

    # Build picklist mapping template
    picklist_rows = []
    for row in results:
        if row.get("Picklist Mapping") != "REQUIRED":
            continue
        bbf_values = split_values(row.get("BBF Picklist Values", ""))
        if not bbf_values:
            continue
        es_values = split_values(row.get("ES Picklist Values", ""))
        es_value_source = "picklist"
        if not es_values:
            es_values = parse_top_values(row.get("ES Top Values", "")) or split_values(
                row.get("ES Sample Values", "")
            )
            es_value_source = "sample"
        if not es_values:
            continue
        bbf_index = {v.strip().lower(): v for v in bbf_values}
        seen = set()
        for es_val in es_values:
            norm = es_val.strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            bbf_match = bbf_index.get(norm, "")
            match_type = "exact" if bbf_match else ""
            status = "auto" if bbf_match else "needs_review"
            picklist_rows.append(
                {
                    "BBF Field API": row.get("BBF Field API"),
                    "ES Field API": row.get("ES Field API"),
                    "ES Value": es_val,
                    "BBF Value": bbf_match,
                    "Match Type": match_type,
                    "ES Value Source": es_value_source,
                    "Status": status,
                    "Notes": "",
                }
            )

    # Required fields check
    results_by_bbf = {r["BBF Field API"]: r for r in results}
    required_rows = []
    for bbf in bbf_fields:
        nillable = to_bool(bbf.nillable)
        createable = to_bool(bbf.createable)
        if nillable is not False or createable is not True:
            continue
        mapped = results_by_bbf.get(bbf.api)
        required_rows.append(
            {
                "BBF Field API": bbf.api,
                "BBF Label": bbf.label,
                "BBF Type": bbf.ftype,
                "Nillable": bbf.nillable,
                "Createable": bbf.createable,
                "BBF Null %": bbf.null_pct,
                "Has Mapping": "Yes" if mapped else "No",
                "ES Field API": mapped.get("ES Field API") if mapped else "",
                "Decision": mapped.get("Decision") if mapped else "",
                "Skip Reason": skip_reasons.get(bbf.api, ""),
                "Notes": "",
            }
        )

    transform_library_rows = [
        {
            "Transform": "trim",
            "Description": "Trim leading/trailing whitespace",
            "Example": "trim(value)",
            "Applies To": "string",
        },
        {
            "Transform": "upper",
            "Description": "Uppercase text",
            "Example": "upper(value)",
            "Applies To": "string",
        },
        {
            "Transform": "lower",
            "Description": "Lowercase text",
            "Example": "lower(value)",
            "Applies To": "string",
        },
        {
            "Transform": "concat",
            "Description": "Concatenate fields with separator",
            "Example": "concat(A, ' ', B)",
            "Applies To": "string",
        },
        {
            "Transform": "date_format",
            "Description": "Format dates to target pattern",
            "Example": "date_format(value, 'YYYY-MM-DD')",
            "Applies To": "date/datetime",
        },
        {
            "Transform": "boolean_map",
            "Description": "Map truthy strings to boolean",
            "Example": "map({'Y':True,'N':False})",
            "Applies To": "boolean",
        },
        {
            "Transform": "picklist_map",
            "Description": "Map picklist values using mapping table",
            "Example": "picklist_map(value)",
            "Applies To": "picklist",
        },
        {
            "Transform": "truncate",
            "Description": "Truncate string to max length",
            "Example": "truncate(value, 80)",
            "Applies To": "string",
        },
    ]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(results).to_excel(writer, sheet_name="mappings", index=False)
        if picklist_rows:
            pd.DataFrame(picklist_rows).to_excel(
                writer, sheet_name="picklist_mappings", index=False
            )
        pd.DataFrame(transform_library_rows).to_excel(
            writer, sheet_name="transform_library", index=False
        )
        if required_rows:
            pd.DataFrame(required_rows).to_excel(
                writer, sheet_name="required_fields_check", index=False
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
