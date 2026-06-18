from __future__ import annotations

import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


DEFAULT_MAIN_URL = "https://ucsf.box.com/shared/static/nzp30ijxahusyl73ety1l52utphnleg9.csv"
WIKIPEDIA_HEADERS = {
    "User-Agent": "ChemNetResearchBot/1.0 (local data repair; contact: local)",
    "Accept": "application/json",
}
OPENALEX_HEADERS = {
    "User-Agent": "ChemNetResearchBot/1.0 (local data repair; contact: local)",
    "Accept": "application/json",
}
CROSSREF_HEADERS = {
    "User-Agent": "ChemNetResearchBot/1.0 (local data repair; contact: local)",
    "Accept": "application/json",
}


class Command(BaseCommand):
    help = (
        "Find funding source names containing '?', suggest best replacements, "
        "and optionally apply high-confidence fixes."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--input-csv",
            default=DEFAULT_MAIN_URL,
            help="Input CSV path or URL. Defaults to current main dataset URL.",
        )
        parser.add_argument(
            "--output-csv",
            default=None,
            help="Output CSV path for repaired data. Defaults to data/esandt_papers_main_repaired.csv",
        )
        parser.add_argument(
            "--report-csv",
            default=None,
            help="Report CSV path. Defaults to data/funding_source_repair_report.csv",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.88,
            help="Minimum similarity score for auto-apply (0 to 1). Default: 0.88",
        )
        parser.add_argument(
            "--min-gap",
            type=float,
            default=0.08,
            help="Required score gap between best and second-best candidates. Default: 0.08",
        )
        parser.add_argument(
            "--online-lookup",
            action="store_true",
            help="Use online funder/entity lookup as an extra signal for suggestions.",
        )
        parser.add_argument(
            "--online-provider",
            choices=["all", "wikipedia", "openalex", "crossref"],
            default="all",
            help="Provider(s) for online lookup. Default: all",
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Apply auto-approved replacements and write output CSV.",
        )
        parser.add_argument(
            "--manual-approve-csv",
            default=None,
            help=(
                "Optional CSV with columns 'original' and 'apply'. "
                "Rows marked apply=1/true/yes will be written even if they do not pass the auto threshold."
            ),
        )
        parser.add_argument(
            "--interactive-review",
            action="store_true",
            help=(
                "Review near-threshold suggestions in terminal and manually approve/skip/edit them "
                "during execution."
            ),
        )
        parser.add_argument(
            "--review-margin",
            type=float,
            default=0.08,
            help=(
                "When interactive review is enabled, include suggestions with chosen_score within this "
                "distance below threshold. Default: 0.08"
            ),
        )

    def handle(self, *args, **options):
        input_csv = options["input_csv"]
        threshold = options["threshold"]
        min_gap = options["min_gap"]
        use_online = options["online_lookup"]
        online_provider = options["online_provider"]
        apply_changes = options["apply"]
        manual_approve_csv = options["manual_approve_csv"]
        interactive_review = options["interactive_review"]
        review_margin = options["review_margin"]

        if threshold < 0 or threshold > 1:
            raise CommandError("--threshold must be between 0 and 1")
        if min_gap < 0 or min_gap > 1:
            raise CommandError("--min-gap must be between 0 and 1")
        if review_margin < 0 or review_margin > 1:
            raise CommandError("--review-margin must be between 0 and 1")

        if interactive_review and not sys.stdin.isatty():
            self.stdout.write(
                self.style.WARNING(
                    "Interactive review requested but no TTY detected; continuing without prompts."
                )
            )
            interactive_review = False

        output_csv = options["output_csv"] or str(
            Path(settings.BASE_DIR) / "data" / "esandt_papers_main_repaired.csv"
        )
        report_csv = options["report_csv"] or str(
            Path(settings.BASE_DIR) / "data" / "funding_source_repair_report.csv"
        )

        self.stdout.write(f"Loading dataset from: {input_csv}")
        df = pd.read_csv(input_csv)

        if "Funding Sources" not in df.columns:
            raise CommandError("Input CSV does not contain 'Funding Sources' column")

        all_sources = self._extract_sources(df["Funding Sources"])
        bad_sources = sorted({s for s in all_sources if "?" in s})
        good_sources = sorted({s for s in all_sources if s and "?" not in s})

        if not bad_sources:
            self.stdout.write(self.style.SUCCESS("No funding sources with '?' were found."))
            return

        self.stdout.write(
            f"Found {len(bad_sources)} unique corrupted funding source names and {len(good_sources)} clean candidates."
        )

        replacements: Dict[str, str] = {}
        interactive_replacements: Dict[str, str] = {}
        report_rows: List[dict] = []
        interactive_enabled = interactive_review

        for bad in bad_sources:
            best, best_score, second_score = self._best_match(bad, good_sources)
            gap = best_score - second_score

            online_title = ""
            online_best = ""
            online_score = 0.0
            online_gap = 0.0
            online_provider_name = ""
            provider_hits: Dict[str, List[str]] = {}
            all_online_titles: List[str] = []
            direct_online_title = ""
            direct_online_score = 0.0
            direct_online_second = 0.0

            chosen = best
            chosen_score = best_score
            chosen_gap = gap
            source = "local"

            if use_online:
                for provider in self._online_providers(online_provider):
                    titles = self._provider_titles(provider, bad)
                    if titles:
                        provider_hits[provider] = titles
                        all_online_titles.extend(titles)

                # Keep insertion order and remove duplicates.
                all_online_titles = list(dict.fromkeys(all_online_titles))

                for title in all_online_titles:
                    candidate_best, candidate_score, candidate_second = self._best_match(title, good_sources)
                    candidate_gap = candidate_score - candidate_second
                    if candidate_score > online_score:
                        online_title = title
                        online_best = candidate_best
                        online_score = candidate_score
                        online_gap = candidate_gap
                        online_provider_name = self._title_provider(title, provider_hits)

                    # Also compare online title directly to corrupted text.
                    direct_score = self._score(bad, title)
                    if direct_score > direct_online_score:
                        direct_online_second = direct_online_score
                        direct_online_score = direct_score
                        direct_online_title = title
                    elif direct_score > direct_online_second:
                        direct_online_second = direct_score

                if online_score > chosen_score:
                    chosen = online_best
                    chosen_score = online_score
                    chosen_gap = online_gap
                    source = "online"

                # Fallback for cases where clean candidate pool lacks the exact corrected name.
                if direct_online_score > chosen_score:
                    chosen = direct_online_title
                    chosen_score = direct_online_score
                    chosen_gap = direct_online_score - direct_online_second
                    source = "online_direct"

            auto_apply = bool(chosen and chosen_score >= threshold and chosen_gap >= min_gap)
            if auto_apply:
                replacements[bad] = chosen

            interactive_apply = False
            interactive_choice = ""
            near_threshold = bool(chosen and not auto_apply and chosen_score >= max(0.0, threshold - review_margin))
            if interactive_enabled and near_threshold:
                interactive_apply, interactive_choice, interactive_enabled = self._interactive_decision(
                    original=bad,
                    suggestion=chosen,
                    score=chosen_score,
                    threshold=threshold,
                    source=source,
                )
                if interactive_apply and interactive_choice:
                    interactive_replacements[bad] = interactive_choice

            report_rows.append(
                {
                    "original": bad,
                    "local_best": best,
                    "local_score": round(best_score, 4),
                    "local_second_score": round(second_score, 4),
                    "score_gap": round(gap, 4),
                    "online_title": online_title,
                    "online_best": online_best,
                    "online_score": round(online_score, 4),
                    "online_provider": online_provider_name,
                    "direct_online_title": direct_online_title,
                    "direct_online_score": round(direct_online_score, 4),
                    "wiki_candidates": " | ".join(provider_hits.get("wikipedia", [])),
                    "openalex_candidates": " | ".join(provider_hits.get("openalex", [])),
                    "crossref_candidates": " | ".join(provider_hits.get("crossref", [])),
                    "chosen": chosen,
                    "chosen_score": round(chosen_score, 4),
                    "source": source,
                    "auto_apply": auto_apply,
                    "interactive_apply": interactive_apply,
                    "interactive_choice": interactive_choice,
                }
            )

        report_df = pd.DataFrame(report_rows).sort_values(
            by=["auto_apply", "chosen_score"], ascending=[False, False]
        )
        Path(report_csv).parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_csv, index=False)

        if manual_approve_csv:
            manual_df = self._load_manual_approvals(manual_approve_csv)
            if not manual_df.empty:
                report_df = report_df.merge(manual_df, on="original", how="left")
                report_df["manual_apply"] = report_df["manual_apply"].fillna(False)
                report_df["final_apply"] = report_df["auto_apply"] | report_df["manual_apply"]
                report_df.to_csv(report_csv, index=False)

                approved_manual = report_df.loc[report_df["manual_apply"], "original"].tolist()
                if approved_manual:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Manual approvals loaded for {len(approved_manual)} originals: {len(approved_manual)}"
                        )
                    )

        self.stdout.write(self.style.SUCCESS(f"Wrote repair report: {report_csv}"))
        self.stdout.write(f"Auto-approved replacements: {len(replacements)}")
        if interactive_replacements:
            self.stdout.write(f"Interactive approvals: {len(interactive_replacements)}")

        if not apply_changes:
            self.stdout.write(
                self.style.WARNING(
                    "Dry run mode: no dataset changes written. Use --apply to write repaired CSV."
                )
            )
            return

        final_replacements = dict(replacements)
        final_replacements.update(interactive_replacements)
        if manual_approve_csv:
            manual_df = self._load_manual_approvals(manual_approve_csv)
            for _, row in manual_df.iterrows():
                if bool(row.get("manual_apply", False)) and row["original"] not in final_replacements:
                    chosen = row.get("chosen", "")
                    if isinstance(chosen, str) and chosen.strip():
                        final_replacements[row["original"]] = chosen.strip()

        repaired_df = df.copy()
        repaired_df["Funding Sources"] = repaired_df["Funding Sources"].apply(
            lambda v: self._replace_sources_in_cell(v, final_replacements)
        )

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        repaired_df.to_csv(output_csv, index=False)
        self.stdout.write(self.style.SUCCESS(f"Wrote repaired dataset: {output_csv}"))

    def _interactive_decision(
        self,
        original: str,
        suggestion: str,
        score: float,
        threshold: float,
        source: str,
    ) -> Tuple[bool, str, bool]:
        self.stdout.write("\n--- Interactive Review ---")
        self.stdout.write(f"Original:   {original}")
        self.stdout.write(f"Suggested:  {suggestion}")
        self.stdout.write(f"Source:     {source}")
        self.stdout.write(f"Score:      {score:.4f} (threshold {threshold:.4f})")
        self.stdout.write("Approve suggestion? [y]es / [n]o / [e]dit / [q]uit review")

        try:
            answer = input("> ").strip().lower()
        except EOFError:
            return False, "", False

        if answer in {"q", "quit"}:
            self.stdout.write("Interactive review stopped for remaining rows.")
            return False, "", False
        if answer in {"y", "yes"}:
            return True, suggestion, True
        if answer in {"e", "edit"}:
            try:
                custom = input("Enter replacement text: ").strip()
            except EOFError:
                return False, "", True
            if custom:
                return True, custom, True
            return False, "", True

        return False, "", True

    def _extract_sources(self, series: pd.Series) -> List[str]:
        sources: List[str] = []
        for cell in series.dropna():
            text = str(cell)
            parts = [p.strip() for p in text.split(";") if p.strip()]
            sources.extend(parts)
        return sources

    def _replace_sources_in_cell(self, value: object, replacements: Dict[str, str]) -> str:
        if pd.isna(value):
            return ""
        parts = [p.strip() for p in str(value).split(";") if p.strip()]
        fixed = [replacements.get(part, part) for part in parts]
        return "; ".join(fixed)

    def _load_manual_approvals(self, path: str) -> pd.DataFrame:
        manual_df = pd.read_csv(path)
        has_apply = "apply" in manual_df.columns
        has_auto_apply = "auto_apply" in manual_df.columns

        if "original" not in manual_df.columns or (not has_apply and not has_auto_apply):
            expected = ["original", "apply"]
            raise CommandError(
                "Manual approvals CSV must contain 'original' and one of 'apply' or 'auto_apply'. "
                f"Expected at least: {expected}"
            )

        def to_bool(value: object) -> bool:
            if pd.isna(value):
                return False
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            return text in {"1", "true", "yes", "y", "apply", "approve"}

        manual_df = manual_df.copy()
        manual_df["original"] = manual_df["original"].astype(str).str.strip()
        apply_col = "apply" if has_apply else "auto_apply"
        manual_df["manual_apply"] = manual_df[apply_col].apply(to_bool)
        if "chosen" not in manual_df.columns:
            manual_df["chosen"] = ""
        return manual_df[["original", "manual_apply", "chosen"]]

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\?+", " ", text)
        text = re.sub(r"[^a-z0-9\s&\-.,()]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _query_variants_for_score(self, query: str) -> List[str]:
        variants = [query]

        # Variant requested by user: compare against a form where '?' is removed.
        removed_q = re.sub(r"\?+", "", query).strip()
        if removed_q and removed_q not in variants:
            variants.append(removed_q)

        return variants

    def _score(self, a: str, b: str) -> float:
        best = 0.0
        normalized_b = self._normalize(b)
        for a_variant in self._query_variants_for_score(a):
            score = SequenceMatcher(None, self._normalize(a_variant), normalized_b).ratio()
            if score > best:
                best = score
        return best

    def _best_match(self, query: str, candidates: Iterable[str]) -> Tuple[str, float, float]:
        best = ""
        best_score = 0.0
        second = 0.0

        for cand in candidates:
            score = self._score(query, cand)
            if score > best_score:
                second = best_score
                best_score = score
                best = cand
            elif score > second:
                second = score

        return best, best_score, second

    def _wikipedia_title(self, query: str) -> str:
        titles = self._wikipedia_titles(query)
        return titles[0] if titles else ""

    def _online_providers(self, selected: str) -> List[str]:
        if selected == "all":
            return ["wikipedia", "openalex", "crossref"]
        return [selected]

    def _provider_titles(self, provider: str, query: str) -> List[str]:
        if provider == "wikipedia":
            return self._wikipedia_titles(query)
        if provider == "openalex":
            return self._openalex_titles(query)
        if provider == "crossref":
            return self._crossref_titles(query)
        return []

    def _title_provider(self, title: str, provider_hits: Dict[str, List[str]]) -> str:
        for provider, titles in provider_hits.items():
            if title in titles:
                return provider
        return ""

    def _wikipedia_titles(self, query: str) -> List[str]:
        try:
            cleaned = self._normalize_lookup_query(query)
            if not cleaned:
                return []

            titles: List[str] = []
            for search_text in self._wikipedia_search_variants(cleaned):
                resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": search_text,
                        "srlimit": 5,
                        "format": "json",
                    },
                    headers=WIKIPEDIA_HEADERS,
                    timeout=8,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    search_results = payload.get("query", {}).get("search", [])
                    if search_results:
                        titles.extend(
                            str(result.get("title", "")).strip()
                            for result in search_results
                            if str(result.get("title", "")).strip()
                        )

                resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": search_text,
                        "limit": 1,
                        "namespace": 0,
                        "format": "json",
                    },
                    headers=WIKIPEDIA_HEADERS,
                    timeout=8,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    if len(payload) >= 2 and payload[1]:
                        titles.extend(
                            str(title).strip()
                            for title in payload[1]
                            if str(title).strip()
                        )
        except Exception:
            return []

        return list(dict.fromkeys(titles))

    def _normalize_lookup_query(self, text: str) -> str:
        text = re.sub(r"\?+", "", text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"[^\w\s&\-.,()]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _wikipedia_search_variants(self, text: str) -> List[str]:
        variants = [text]
        ascii_text = text.encode("ascii", errors="ignore").decode("ascii").strip()
        if ascii_text and ascii_text != text:
            variants.append(ascii_text)
        return list(dict.fromkeys(variant for variant in variants if variant))

    def _openalex_titles(self, query: str) -> List[str]:
        try:
            cleaned = self._normalize_lookup_query(query)
            if not cleaned:
                return []

            titles: List[str] = []
            for search_text in self._wikipedia_search_variants(cleaned):
                resp = requests.get(
                    "https://api.openalex.org/funders",
                    params={"search": search_text, "per-page": 5},
                    headers=OPENALEX_HEADERS,
                    timeout=8,
                )
                if resp.status_code != 200:
                    continue

                payload = resp.json()
                for result in payload.get("results", []):
                    name = str(result.get("display_name", "")).strip()
                    if name:
                        titles.append(name)
            return list(dict.fromkeys(titles))
        except Exception:
            return []

    def _crossref_titles(self, query: str) -> List[str]:
        try:
            cleaned = self._normalize_lookup_query(query)
            if not cleaned:
                return []

            titles: List[str] = []
            for search_text in self._wikipedia_search_variants(cleaned):
                resp = requests.get(
                    "https://api.crossref.org/funders",
                    params={"query": search_text, "rows": 5},
                    headers=CROSSREF_HEADERS,
                    timeout=8,
                )
                if resp.status_code != 200:
                    continue

                payload = resp.json()
                items = payload.get("message", {}).get("items", [])
                for item in items:
                    name = str(item.get("name", "")).strip()
                    if name:
                        titles.append(name)
            return list(dict.fromkeys(titles))
        except Exception:
            return []
