#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from simple_salesforce import Salesforce  # type: ignore


# ----------------------------
# Config / Helpers
# ----------------------------


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def connect_salesforce(prefix: str) -> Salesforce:
    return Salesforce(
        username=get_env(f"{prefix}_USERNAME"),
        password=get_env(f"{prefix}_PASSWORD"),
        security_token=get_env(f"{prefix}_TOKEN"),
        domain=os.getenv(f"{prefix}_DOMAIN", "test"),
    )


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ----------------------------
# Subset selection (ES)
# ----------------------------


def query_all(sf: Salesforce, soql: str) -> List[Dict[str, Any]]:
    result = sf.query_all(soql)
    return result.get("records", [])


def build_subset_ids(es_sf: Salesforce, cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    subset_cfg = cfg.get("subset", {})
    statuses = subset_cfg.get("active_order_statuses", [])
    pg_like = subset_cfg.get("exclude_project_group_like", "")
    record_type = subset_cfg.get("service_order_record_type", "")
    require_bbf_ban = subset_cfg.get("require_bbf_ban_flag", True)
    account_type_filter = subset_cfg.get("account_type_filter")

    # BANs (Billing_Invoice__c)
    ban_query = "SELECT Id, Account__c FROM Billing_Invoice__c WHERE BBF_Ban__c = true"
    ban_rows = query_all(es_sf, ban_query)
    ban_ids = [r["Id"] for r in ban_rows if r.get("Id")]
    raw_account_ids = [r["Account__c"] for r in ban_rows if r.get("Account__c")]

    # Orders (Service subset)
    status_str = "','".join(statuses)
    pg_filter = (
        ""
        if not pg_like
        else f"AND (Project_Group__c = null OR (NOT Project_Group__c LIKE '%{pg_like}%'))"
    )
    ban_filter = "AND Billing_Invoice__r.BBF_Ban__c = true" if require_bbf_ban else ""
    rt_filter = (
        f"AND Service_Order_Record_Type__c = '{record_type}'" if record_type else ""
    )

    order_query = f"""
        SELECT Id, Address_A__c, Address_Z__c, Billing_Invoice__c
        FROM Order
        WHERE Status IN ('{status_str}')
          {pg_filter}
          {rt_filter}
          {ban_filter}
    """
    order_rows = query_all(es_sf, order_query)
    order_ids = [r["Id"] for r in order_rows if r.get("Id")]
    address_ids = []
    for r in order_rows:
        if r.get("Address_A__c"):
            address_ids.append(r["Address_A__c"])
        if r.get("Address_Z__c"):
            address_ids.append(r["Address_Z__c"])

    # Account IDs (optionally filter by Type)
    filtered_account_ids = []
    if raw_account_ids:
        for account_chunk in chunked(list(dict.fromkeys(raw_account_ids)), 200):
            ids_str = "','".join(account_chunk)
            type_filter = (
                f" AND Type = '{account_type_filter}'" if account_type_filter else ""
            )
            acct_query = (
                f"SELECT Id FROM Account WHERE Id IN ('{ids_str}'){type_filter}"
            )
            acct_rows = query_all(es_sf, acct_query)
            filtered_account_ids.extend([r["Id"] for r in acct_rows if r.get("Id")])

    # Contacts from subset accounts
    contact_ids: List[str] = []
    if filtered_account_ids:
        for account_chunk in chunked(list(dict.fromkeys(filtered_account_ids)), 200):
            ids_str = "','".join(account_chunk)
            contact_query = f"SELECT Id FROM Contact WHERE AccountId IN ('{ids_str}')"
            contact_rows = query_all(es_sf, contact_query)
            contact_ids.extend([r["Id"] for r in contact_rows if r.get("Id")])

    # OrderItems from subset orders
    orderitem_ids: List[str] = []
    if order_ids:
        for order_chunk in chunked(list(dict.fromkeys(order_ids)), 200):
            ids_str = "','".join(order_chunk)
            oi_query = f"SELECT Id FROM OrderItem WHERE OrderId IN ('{ids_str}')"
            oi_rows = query_all(es_sf, oi_query)
            orderitem_ids.extend([r["Id"] for r in oi_rows if r.get("Id")])

    # Off_Net__c from subset orders (SOF1__c relationship)
    offnet_ids: List[str] = []
    if order_ids:
        for order_chunk in chunked(list(dict.fromkeys(order_ids)), 200):
            ids_str = "','".join(order_chunk)
            offnet_query = f"SELECT Id FROM Off_Net__c WHERE SOF1__c IN ('{ids_str}')"
            offnet_rows = query_all(es_sf, offnet_query)
            offnet_ids.extend([r["Id"] for r in offnet_rows if r.get("Id")])

    return {
        "Billing_Invoice__c": list(dict.fromkeys(ban_ids)),
        "Account": list(dict.fromkeys(filtered_account_ids)),
        "Contact": list(dict.fromkeys(contact_ids)),
        "Order": list(dict.fromkeys(order_ids)),
        "Address__c": list(dict.fromkeys(address_ids)),
        "OrderItem": list(dict.fromkeys(orderitem_ids)),
        "Off_Net__c": list(dict.fromkeys(offnet_ids)),
    }


# ----------------------------
# Field metadata + filtering
# ----------------------------


def describe_fields(sf: Salesforce, object_name: str) -> List[Dict[str, Any]]:
    desc = sf.__getattr__(object_name).describe()
    return desc.get("fields", [])


def build_picklist_values(field: Dict[str, Any]) -> str:
    if field.get("type") not in ("picklist", "multipicklist"):
        return ""
    values = []
    for pv in field.get("picklistValues", []) or []:
        value = pv.get("value")
        if value is not None:
            values.append(value)
    return "; ".join(values)


def filter_fields(
    fields: List[Dict[str, Any]],
    filters: Dict[str, Any],
    excluded_by_object: List[str],
    org_filters: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    include = []
    excluded = []

    exclude_types = set(filters.get("exclude_types", []))
    exclude_name_patterns = list(filters.get("exclude_field_name_patterns", []))
    if org_filters:
        exclude_name_patterns.extend(org_filters.get("exclude_field_name_patterns", []))
    exclude_system = set(filters.get("exclude_system_fields", []))
    exclude_calculated = bool(filters.get("exclude_calculated", False))
    include_calculated_types = set(filters.get("include_calculated_types", []))
    exclude_autonumber = bool(filters.get("exclude_autonumber", False))

    for f in fields:
        reasons = []
        name = f.get("name")
        if not f.get("queryable", True):
            reasons.append("not_queryable")
        for pattern in exclude_name_patterns:
            try:
                if re.search(pattern, name or ""):
                    reasons.append(f"name_pattern:{pattern}")
                    break
            except re.error:
                reasons.append("name_pattern:invalid_regex")
                break
        if name in exclude_system:
            reasons.append("system_field")
        if name in excluded_by_object:
            reasons.append("excluded_by_object")
        if f.get("type") in exclude_types:
            reasons.append(f"type:{f.get('type')}")
        if f.get("calculated"):
            if exclude_calculated:
                reasons.append("calculated")
            elif (
                include_calculated_types
                and f.get("type") not in include_calculated_types
            ):
                reasons.append(f"calculated_type_excluded:{f.get('type')}")
        if exclude_autonumber and f.get("autoNumber"):
            reasons.append("autonumber")

        if reasons:
            excluded.append(
                {
                    "Field API Name": name,
                    "Field Label": f.get("label"),
                    "Field Type": f.get("type"),
                    "Reasons": ", ".join(reasons),
                }
            )
        else:
            include.append(f)

    return include, excluded


# ----------------------------
# Query + stats
# ----------------------------


def coerce_value(v: Any) -> Any:
    if isinstance(v, dict):
        # e.g., geolocation or nested
        return str(v)
    return v


def compute_stats(values: List[Any], field_type: str) -> Dict[str, Any]:
    total = len(values)
    cleaned = [coerce_value(v) for v in values]
    non_null = [v for v in cleaned if v not in (None, "")]
    null_count = total - len(non_null)
    null_pct = (null_count / total * 100) if total else 0.0

    str_values = [str(v) for v in non_null]
    distinct_count = len(set(str_values))

    top_values = Counter(str_values).most_common(5)
    top_values_str = "; ".join([f"{v} ({c})" for v, c in top_values])

    sample_values = []
    for v in str_values:
        if v not in sample_values:
            sample_values.append(v)
        if len(sample_values) >= 5:
            break
    sample_values_str = "; ".join(sample_values)

    stats = {
        "Total Count": total,
        "Non-Null Count": len(non_null),
        "Null %": round(null_pct, 2),
        "Distinct Count": distinct_count,
        "Top Values": top_values_str,
        "Sample Values": sample_values_str,
        "Min Value": "",
        "Max Value": "",
        "Avg Value": "",
        "Min Length": "",
        "Max Length": "",
        "Avg Length": "",
    }

    if not non_null:
        return stats

    if field_type in ("double", "int", "currency", "percent"):
        nums = []
        for v in non_null:
            try:
                nums.append(float(v))
            except Exception:
                continue
        if nums:
            stats["Min Value"] = min(nums)
            stats["Max Value"] = max(nums)
            stats["Avg Value"] = round(sum(nums) / len(nums), 4)
        return stats

    if field_type in ("date", "datetime"):
        # Use string min/max for lightweight comparison
        stats["Min Value"] = min(str_values)
        stats["Max Value"] = max(str_values)
        return stats

    # string length stats
    lengths = [len(str(v)) for v in non_null]
    stats["Min Length"] = min(lengths)
    stats["Max Length"] = max(lengths)
    stats["Avg Length"] = round(sum(lengths) / len(lengths), 2)
    return stats


def fetch_records_by_ids(
    sf: Salesforce,
    object_name: str,
    field_names: List[str],
    ids: List[str],
    sample_size: int,
    id_field: str = "Id",
    max_fields_per_query: int = 80,
) -> List[Dict[str, Any]]:
    if not ids:
        return []
    sample_ids = ids[:sample_size] if sample_size else ids
    records_by_id: Dict[str, Dict[str, Any]] = {}

    for field_chunk in chunked(field_names, max_fields_per_query):
        select_fields = ", ".join([id_field] + field_chunk)
        for id_chunk in chunked(sample_ids, 200):
            ids_str = "','".join(id_chunk)
            soql = (
                f"SELECT {select_fields} FROM {object_name} "
                f"WHERE {id_field} IN ('{ids_str}')"
            )
            rows = query_all(sf, soql)
            for r in rows:
                rid = r.get(id_field)
                if not rid:
                    continue
                if rid not in records_by_id:
                    records_by_id[rid] = {id_field: rid}
                for k, v in r.items():
                    if k == "attributes":
                        continue
                    records_by_id[rid][k] = v
    return list(records_by_id.values())


def build_field_profile(
    records: List[Dict[str, Any]],
    fields: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for f in fields:
        name = f.get("name")
        values = [r.get(name) for r in records]
        stats = compute_stats(values, f.get("type"))

        rows.append(
            {
                "Field API Name": name,
                "Field Label": f.get("label"),
                "Field Type": f.get("type"),
                "Length": f.get("length"),
                "Nillable": f.get("nillable"),
                "Createable": f.get("createable"),
                "Updateable": f.get("updateable"),
                "Calculated": f.get("calculated"),
                "Picklist Values": build_picklist_values(f),
                "Reference To": ", ".join(f.get("referenceTo", []) or []),
                **stats,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile ES/BBF fields on subset records."
    )
    parser.add_argument(
        "--config",
        default="profiles/profile_config.json",
        help="Path to profile_config.json",
    )
    parser.add_argument(
        "--out-dir",
        default="profiles/output",
        help="Directory for profile Excel outputs",
    )
    parser.add_argument(
        "--object",
        action="append",
        help="ES or BBF object API name to limit profiling (repeatable)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample size for all objects",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    # Connect to Salesforce orgs
    es_sf = connect_salesforce("ES")
    bbf_sf = connect_salesforce("BBF")

    # Build subset ids from ES
    subset_ids = build_subset_ids(es_sf, cfg)

    # Filter object pairs if requested
    obj_filter = set(args.object or [])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for pair in cfg.get("object_pairs", []):
        es_obj = pair["es"]
        bbf_obj = pair["bbf"]
        if obj_filter and es_obj not in obj_filter and bbf_obj not in obj_filter:
            continue

        es_fields = describe_fields(es_sf, es_obj)
        bbf_fields = describe_fields(bbf_sf, bbf_obj)

        es_excluded_list = (
            cfg.get("exclude_fields_by_object", {}).get("es", {}).get(es_obj, [])
        )
        bbf_excluded_list = (
            cfg.get("exclude_fields_by_object", {}).get("bbf", {}).get(bbf_obj, [])
        )

        org_filters = cfg.get("field_filters_by_org", {})
        es_included, es_excluded = filter_fields(
            es_fields,
            cfg.get("field_filters", {}),
            es_excluded_list,
            org_filters.get("es"),
        )
        bbf_included, bbf_excluded = filter_fields(
            bbf_fields,
            cfg.get("field_filters", {}),
            bbf_excluded_list,
            org_filters.get("bbf"),
        )

        sample_size = (
            args.sample_size
            or cfg.get("sampling", {}).get("per_object", {}).get(es_obj)
            or cfg.get("sampling", {}).get("default_sample_size", 500)
        )

        # ES records
        es_ids = subset_ids.get(es_obj, [])
        es_records = fetch_records_by_ids(
            es_sf,
            es_obj,
            [f.get("name") for f in es_included],
            es_ids,
            sample_size,
            id_field="Id",
        )

        # BBF records (by ES_Legacy_ID__c)
        bbf_records = []
        if es_ids:
            bbf_records = fetch_records_by_ids(
                bbf_sf,
                bbf_obj,
                [f.get("name") for f in bbf_included],
                es_ids,
                sample_size,
                id_field="ES_Legacy_ID__c",
            )

        es_profile = build_field_profile(es_records, es_included)
        bbf_profile = build_field_profile(bbf_records, bbf_included)

        summary_rows = [
            {"Metric": "ES Object", "Value": es_obj},
            {"Metric": "BBF Object", "Value": bbf_obj},
            {"Metric": "ES Subset ID Count", "Value": len(es_ids)},
            {"Metric": "ES Sample Size", "Value": len(es_records)},
            {"Metric": "BBF Sample Size", "Value": len(bbf_records)},
            {"Metric": "Sample Size Target", "Value": sample_size},
        ]

        out_name = f"{es_obj}__to__{bbf_obj}__profile_{timestamp}.xlsx"
        out_path = os.path.join(args.out_dir, out_name)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name="summary", index=False
            )
            es_profile.to_excel(writer, sheet_name="es_fields", index=False)
            bbf_profile.to_excel(writer, sheet_name="bbf_fields", index=False)
            pd.DataFrame(es_excluded).to_excel(
                writer, sheet_name="es_excluded", index=False
            )
            pd.DataFrame(bbf_excluded).to_excel(
                writer, sheet_name="bbf_excluded", index=False
            )

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
