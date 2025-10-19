#!/usr/bin/env python3
import os
import sys
from pathlib import Path

print("=" * 80)
print("AI Tag Enrichment Batch - Setup Validation")
print("=" * 80)

errors = []
warnings = []

print("\n📋 Checking environment configuration...")

env_file = Path(".env")
if not env_file.exists():
    warnings.append("⚠️  No .env file found. Copy .env.example to .env and configure it.")
else:
    print("✅ .env file exists")

required_vars = ["DATABASE_URL", "OPENAI_API_KEY", "TAGS_CSV_PATH", "OUTPUT_CSV_PATH"]
for var in required_vars:
    value = os.getenv(var, "")
    if not value:
        errors.append(f"❌ Missing required environment variable: {var}")
    elif var == "DATABASE_URL" and value == "postgresql://user:password@localhost:5432/your_database":
        warnings.append(f"⚠️  {var} is still set to example value")
    elif var == "OPENAI_API_KEY" and value.startswith("sk-..."):
        warnings.append(f"⚠️  {var} is still set to example value")
    else:
        print(f"✅ {var} is set")

print("\n📂 Checking file structure...")

src_files = [
    "src/config.py",
    "src/database.py",
    "src/tags_loader.py",
    "src/embeddings.py",
    "src/prototype_knn.py",
    "src/llm_arbiter.py",
    "src/scorer.py",
    "src/output.py",
    "src/main.py"
]

for file in src_files:
    if Path(file).exists():
        print(f"✅ {file}")
    else:
        errors.append(f"❌ Missing file: {file}")

tags_csv = os.getenv("TAGS_CSV_PATH", "data/tags.csv")
if not Path(tags_csv).exists():
    warnings.append(f"⚠️  Tags CSV not found at {tags_csv}. See data/tags_example.csv for format.")
else:
    print(f"✅ Tags CSV exists at {tags_csv}")

output_dir = Path(os.getenv("OUTPUT_CSV_PATH", "output/tag_suggestions.csv")).parent
if not output_dir.exists():
    print(f"📁 Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

print("\n📦 Checking Python dependencies...")
required_packages = [
    "openai",
    "psycopg2",
    "pandas",
    "numpy",
    "sklearn",
    "dotenv"
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"✅ {package}")
    except ImportError:
        errors.append(f"❌ Missing package: {package}")

print("\n" + "=" * 80)

if errors:
    print("❌ ERRORS FOUND:")
    for error in errors:
        print(f"   {error}")

if warnings:
    print("\n⚠️  WARNINGS:")
    for warning in warnings:
        print(f"   {warning}")

if not errors and not warnings:
    print("✅ All checks passed! Ready to run the batch.")
    print("\nTo run the batch:")
    print("   python src/main.py")
elif not errors:
    print("\n⚠️  Setup is mostly complete but has warnings.")
    print("Review the warnings above, then run:")
    print("   python src/main.py")
else:
    print("\n❌ Please fix the errors above before running the batch.")
    sys.exit(1)

print("=" * 80)
