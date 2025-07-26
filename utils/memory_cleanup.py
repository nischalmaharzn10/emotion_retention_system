import json
from typing import List, Dict


def clean_memory_file(path: str) -> None:
    """Cleans a memory JSON file by removing entries that are not from 'user' or 'ai',
    or have empty content."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data: List[Dict] = json.load(f)

        cleaned = [
            item for item in data
            if item.get("role") in {"user", "ai"} and item.get("content", "").strip()
        ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2)

        removed_count = len(data) - len(cleaned)
        print(f"✅ Cleaned {removed_count} empty or irrelevant messages from memory")

    except FileNotFoundError:
        print(f"❌ File not found: {path}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {path}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
