#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from simple_salesforce import Salesforce  # type: ignore

# ---------------------------------------------------------------------
# Salesforce credentials
# ---------------------------------------------------------------------
# Salesforce login credentials (Replace with actual credentials if needed)
sf_username = "vlettau@everstream.net"
sf_password = "MNlkpo0987)(*&"
sf_token = "I4xmQLmm03cXl1O9qI2Z3XAAX"
sf_domain = "test"  # or 'test' for sandbox

# ---------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Export field metadata for one or more Salesforce objects using the standard REST describe() API."
)
parser.add_argument(
    "--object",
    "-o",
    nargs="+",
    required=True,
    help="One or more Object API names separated by space or comma (e.g., Account Order MyObject__c)",
)
parser.add_argument(
    "--output-dir",
    "-d",
    default=".",
    help="Output directory for generated files (default: current directory)",
)
parser.add_argument(
    "--include-managed",
    action="store_true",
    help="Include fields belonging to managed packages (namespaced fields).",
)
parser.add_argument(
    "--include-custom-only",
    action="store_true",
    help="Include only custom fields (__c).",
)
parser.add_argument(
    "--include-required-only",
    action="store_true",
    help="Include only required fields (nillable=False).",
)
args = parser.parse_args()

# Normalize object list (split on commas and strip whitespace)
object_list = []
for entry in args.object:
    for o in entry.split(","):
        name = o.strip()
        if name:
            object_list.append(name)
if not object_list:
    print("❌ No valid object names provided.")
    sys.exit(1)

# ---------------------------------------------------------------------
# Connect to Salesforce
# ---------------------------------------------------------------------
try:
    print("🔌 Connecting to Salesforce...")
    sf = Salesforce(
        username=sf_username,
        password=sf_password,
        security_token=sf_token,
        domain=sf_domain,
    )
    print(f"✅ Logged in to Salesforce instance: {sf.sf_instance}")
except Exception as e:
    print("❌ Error logging in to Salesforce:", e)
    sys.exit(1)


# ---------------------------------------------------------------------
# Fetch fields via standard describe() call
# ---------------------------------------------------------------------
def get_fields_for_object(sf, object_api_name):
    """
    Return a list of fields for a given object using the standard REST describe() call.
    Includes both standard and custom (__c) fields.
    """
    print(f"\n📡 Fetching field metadata for {object_api_name}...")
    try:
        desc = sf.__getattr__(object_api_name).describe()
        fields = desc.get("fields", [])
        print(f"✅ Retrieved {len(fields)} total fields from describe()")
    except Exception as e:
        print(f"⚠️ Error describing {object_api_name}: {e}")
        return []

    filtered = []
    for f in fields:
        api_name = f.get("name", "")
        is_custom = api_name.endswith("__c")
        nillable = f.get("nillable", True)
        calculated = f.get("calculated", False)
        has_namespace = "__" in api_name and not is_custom and api_name.count("__") >= 2

        # Apply filters
        if not args.include_managed and has_namespace:
            continue
        if args.include_custom_only and not is_custom:
            continue
        if args.include_required_only and nillable:
            continue

        filtered.append(
            {
                "Object API Name": object_api_name,
                "Field API Name": api_name,
                "Field Type": f.get("type", ""),
                "Field Label": f.get("label", ""),
                "Length": f.get("length", ""),
                "Is Nillable": nillable,
                "Is Calculated": calculated,
                "Custom": is_custom,
            }
        )

    print(f"✅ Filtered down to {len(filtered)} fields after applying flags")
    return filtered


# ---------------------------------------------------------------------
# Process multiple objects
# ---------------------------------------------------------------------
dataframes = {}

for obj_name in object_list:
    fields = get_fields_for_object(sf, obj_name)
    if not fields:
        print(f"⚠️ Skipping {obj_name}: no fields found.")
        continue

    df = pd.DataFrame(fields)
    dataframes[obj_name] = df

# ---------------------------------------------------------------------
# Write CSV for each object (faster than Excel for intermediate files)
# ---------------------------------------------------------------------
import os

os.makedirs(args.output_dir, exist_ok=True)

for obj_name, df in dataframes.items():
    filename = os.path.join(args.output_dir, f"bbf_{obj_name}_fields.csv")
    df.to_csv(filename, index=False)
    print(f"📄 CSV export complete → {filename}")

print("\n🏁 All objects processed successfully.")
