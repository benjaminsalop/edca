from src.edtool.db.loaders import (
    load_materials,
    load_system_families,
    load_system_variants,
    validate_relationships,
)

materials, m_errs = load_materials()
families, f_errs = load_system_families()
variants, v_errs = load_system_variants()

rel_errs = validate_relationships(materials, families, variants)

all_errors = m_errs + f_errs + v_errs + rel_errs

if all_errors:
    print("DATA VALIDATION ERRORS:")
    for e in all_errors:
        print(" -", e)
    raise SystemExit(1)

print("✅ Data validation passed")