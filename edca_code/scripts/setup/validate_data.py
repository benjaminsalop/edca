from edca_code.scripts.setup.loaders import (
    load_catalog_data,
    validate_all_relationships,
)


data = load_catalog_data(regenerate_systems=True)

relationship_errors = validate_all_relationships(
    data.materials,
    data.floor_families,
    data.floor_variants,
    data.beam_families,
    data.beam_variants,
    data.column_families,
    data.column_variants,
    data.lateral_families,
    data.lateral_variants,
)

all_errors = (
    data.material_errors
    + data.floor_family_errors
    + data.floor_variant_errors
    + data.beam_family_errors
    + data.beam_variant_errors
    + data.column_family_errors
    + data.column_variant_errors
    + data.lateral_family_errors
    + data.lateral_variant_errors
    + relationship_errors
)

if all_errors:
    print("DATA VALIDATION ERRORS:")
    for err in all_errors:
        print(" -", err)
    raise SystemExit(1)

print(
    "✅ Data validation passed "
    f"(materials={len(data.materials)}, "
    f"floor_families={len(data.floor_families)}, floor_variants={len(data.floor_variants)}, "
    f"beam_families={len(data.beam_families)}, beam_variants={len(data.beam_variants)}, "
    f"column_families={len(data.column_families)}, column_variants={len(data.column_variants)}, "
    f"lateral_families={len(data.lateral_families)}, lateral_variants={len(data.lateral_variants)})"
)
