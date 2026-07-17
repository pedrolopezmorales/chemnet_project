from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


CANONICAL_CLASSIFICATIONS = {
    "government": "Government",
    "gov": "Government",
    "company": "Company",
    "corporate": "Company",
    "corp": "Company",
    "university": "University",
    "uni": "University",
    "foundation": "Foundation",
    "nonprofit": "Foundation",
    "non-profit": "Foundation",
    "charity": "Foundation",
    "unknown": "Unknown",
    "other": "Unknown",
    "not found": "Unknown",
}

JSON_STYLE_PAIR = re.compile(
    r"^[\'\"](?P<name>.+?)[\'\"]\s*:\s*[\'\"](?P<classification>.+?)[\'\"]$"
)
UNQUOTED_PAIR = re.compile(
    r"^(?P<name>[^:=\t|]+?)\s*(?::|=>|=|\t|\|)\s*(?P<classification>.+)$"
)


class Command(BaseCommand):
    help = (
        "Update data/company_classifications.json from a text file of mappings. "
        "Supports lines like: \"Name\": \"Government\", or Name => Government."
    )

    def add_arguments(self, parser):
        default_updates = Path(settings.BASE_DIR) / "data" / "company_classification_updates.txt"
        default_json = Path(settings.BASE_DIR) / "data" / "company_classifications.json"

        parser.add_argument(
            "--updates-file",
            default=str(default_updates),
            help="Path to the text file containing updates.",
        )
        parser.add_argument(
            "--classifications-json",
            default=str(default_json),
            help="Path to company classifications JSON.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse and report what would change without writing files.",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Fail if any update line cannot be parsed.",
        )
        parser.add_argument(
            "--no-backup",
            action="store_true",
            help="Do not create a .bak copy before writing JSON.",
        )
        parser.add_argument(
            "--create-template",
            action="store_true",
            help="Create a template updates file at --updates-file and exit.",
        )

    def handle(self, *args, **options):
        updates_file = Path(options["updates_file"])
        classifications_json = Path(options["classifications_json"])
        dry_run = options["dry_run"]
        strict = options["strict"]
        no_backup = options["no_backup"]
        create_template = options["create_template"]

        if create_template:
            self._write_template(updates_file)
            self.stdout.write(self.style.SUCCESS(f"Template created: {updates_file}"))
            return

        if not updates_file.exists():
            raise CommandError(
                f"Updates file not found: {updates_file}. "
                f"Use --create-template to generate one."
            )

        if not classifications_json.exists():
            raise CommandError(f"Classifications JSON not found: {classifications_json}")

        updates_text = updates_file.read_text(encoding="utf-8")
        updates, errors = self._parse_updates(updates_text)

        if not updates:
            raise CommandError("No valid updates found in updates file.")

        if errors and strict:
            preview = "\n".join(errors[:10])
            raise CommandError(
                "Unparsable lines found while running in --strict mode:\n"
                f"{preview}"
            )

        with classifications_json.open("r", encoding="utf-8") as f:
            classifications = json.load(f)

        if not isinstance(classifications, dict):
            raise CommandError("Classifications JSON must be a single JSON object.")

        added = 0
        changed = 0
        unchanged = 0

        for name, new_classification in updates.items():
            old_classification = classifications.get(name)
            if old_classification is None:
                added += 1
            elif old_classification == new_classification:
                unchanged += 1
            else:
                changed += 1
            classifications[name] = new_classification

        self.stdout.write(
            "Parsed updates: "
            f"{len(updates)} valid, {len(errors)} skipped, "
            f"{added} added, {changed} changed, {unchanged} unchanged."
        )

        if errors:
            self.stdout.write(self.style.WARNING("Skipped lines:"))
            for line in errors[:20]:
                self.stdout.write(self.style.WARNING(f"  - {line}"))
            if len(errors) > 20:
                self.stdout.write(self.style.WARNING("  - ..."))

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry run complete. No files written."))
            return

        if not no_backup:
            backup_path = classifications_json.with_suffix(".json.bak")
            shutil.copy2(classifications_json, backup_path)
            self.stdout.write(f"Backup written: {backup_path}")

        with classifications_json.open("w", encoding="utf-8") as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)
            f.write("\n")

        self.stdout.write(self.style.SUCCESS(f"Updated classifications: {classifications_json}"))

    def _normalize_classification(self, raw: str) -> str:
        normalized = raw.strip().strip(",").strip().strip("\"'")
        key = normalized.lower()
        canonical = CANONICAL_CLASSIFICATIONS.get(key)
        if canonical:
            return canonical
        raise ValueError(
            f"Invalid classification '{raw}'. Use one of: "
            "Government, Company, University, Foundation, Unknown"
        )

    def _parse_updates(self, text: str) -> Tuple[Dict[str, str], List[str]]:
        updates: Dict[str, str] = {}
        errors: List[str] = []

        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed_json = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_json = None

            if isinstance(parsed_json, dict):
                for raw_name, raw_classification in parsed_json.items():
                    if not isinstance(raw_name, str):
                        errors.append(f"JSON key is not a string: {raw_name}")
                        continue
                    try:
                        updates[raw_name.strip()] = self._normalize_classification(
                            str(raw_classification)
                        )
                    except ValueError as exc:
                        errors.append(str(exc))
                return updates, errors

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            if line in {"{", "}"}:
                continue

            line = line.rstrip(",").strip()

            match = JSON_STYLE_PAIR.match(line)
            if not match:
                match = UNQUOTED_PAIR.match(line)

            if not match:
                errors.append(f"line {line_number}: {raw_line}")
                continue

            name = match.group("name").strip().strip("\"'")
            raw_classification = match.group("classification").strip()

            if not name:
                errors.append(f"line {line_number}: empty organization name")
                continue

            try:
                classification = self._normalize_classification(raw_classification)
            except ValueError as exc:
                errors.append(f"line {line_number}: {exc}")
                continue

            updates[name] = classification

        return updates, errors

    def _write_template(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        template = """# Add one update per line. Supported formats:
#   \"Organization Name\": \"Government\",
#   Organization Name => Company
#   Organization Name: University
#
# Allowed classifications:
#   Government, Company, University, Foundation, Unknown

\"OpenAI\": \"Company\",
\"MIT\": \"University\",
\"EPA\": \"Government\",
\"Wellcome Trust\": \"Foundation\",
\"Unclear Source\": \"Unknown\"
"""
        path.write_text(template, encoding="utf-8")