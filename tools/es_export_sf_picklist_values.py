#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from simple_salesforce import Salesforce

# ---------------------------------------------------------------------
# Salesforce credentials
# ---------------------------------------------------------------------
sf_username = "sfdcapi@everstream.net"
sf_password = "pV4CAxns8DQtJsBq!"
sf_token = "r1uoYiusK19RbrflARydi86TA"
sf_domain = "login"  # or 'test' for sandbox

# ---------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Export detailed picklist values for Salesforce objects."
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
    "--include-inactive",
    action="store_true",
    help="Include inactive picklist values (default: active only).",
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
# Fetch picklist values
# ---------------------------------------------------------------------
def get_picklist_values_for_object(sf, object_api_name):
    """
    Return a list of all picklist values for all picklist fields in an object.
    Each row represents one picklist value.
    """
    print(f"\n📡 Fetching picklist values for {object_api_name}...")
    try:
        desc = sf.__getattr__(object_api_name).describe()
        fields = desc.get("fields", [])
        print(f"✅ Retrieved {len(fields)} total fields from describe()")
    except Exception as e:
        print(f"⚠️ Error describing {object_api_name}: {e}")
        return []

    picklist_data = []
    picklist_count = 0

    for f in fields:
        api_name = f.get("name", "")
        field_type = f.get("type", "")
        is_custom = api_name.endswith("__c")
        has_namespace = "__" in api_name and not is_custom and api_name.count("__") >= 2

        # Skip non-picklist fields
        if field_type not in ["picklist", "multipicklist"]:
            continue

        # Apply filters
        if not args.include_managed and has_namespace:
            continue
        if args.include_custom_only and not is_custom:
            continue

        picklist_count += 1

        # Extract picklist values
        picklist_values_list = f.get("picklistValues", [])

        if not picklist_values_list:
            # Add a row even if no values exist
            picklist_data.append(
                {
                    "Object API Name": object_api_name,
                    "Field API Name": api_name,
                    "Field Label": f.get("label", ""),
                    "Field Type": field_type,
                    "Picklist Value": "",
                    "Is Active": "",
                    "Is Default": "",
                    "Label": "",
                    "Total Values": 0,
                }
            )
        else:
            for idx, pv in enumerate(picklist_values_list, 1):
                value = pv.get("value", "")
                is_active = pv.get("active", False)
                is_default = pv.get("defaultValue", False)
                label = pv.get("label", value)  # Label might be different from value

                # Skip inactive values unless flag is set
                if not is_active and not args.include_inactive:
                    continue

                picklist_data.append(
                    {
                        "Object API Name": object_api_name,
                        "Field API Name": api_name,
                        "Field Label": f.get("label", ""),
                        "Field Type": field_type,
                        "Picklist Value": value,
                        "Is Active": is_active,
                        "Is Default": is_default,
                        "Label": label,
                        "Value Order": idx,
                        "Total Values": len(picklist_values_list),
                    }
                )

    print(
        f"✅ Found {picklist_count} picklist fields with {len(picklist_data)} total values"
    )
    return picklist_data


# ---------------------------------------------------------------------
# Process multiple objects
# ---------------------------------------------------------------------
dataframes = {}

for obj_name in object_list:
    picklist_data = get_picklist_values_for_object(sf, obj_name)
    if not picklist_data:
        print(f"⚠️ Skipping {obj_name}: no picklist values found.")
        continue

    df = pd.DataFrame(picklist_data)
    dataframes[obj_name] = df

# ---------------------------------------------------------------------
# Write CSV for each object (faster than Excel for intermediate files)
# ---------------------------------------------------------------------
import os
os.makedirs(args.output_dir, exist_ok=True)

for obj_name, df in dataframes.items():
    filename = os.path.join(args.output_dir, f"es_{obj_name}_picklist_values.csv")
    df.to_csv(filename, index=False)
    print(f"📄 CSV export complete → {filename}")

print("\n✨ All objects processed successfully.")
