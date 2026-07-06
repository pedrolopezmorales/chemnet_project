from django.core.management.base import BaseCommand, CommandError

from networkviewer.network_functions import create_funding_source_category_dataframes


class Command(BaseCommand):
    help = "Build per-category funding source table CSVs (all/government/university/foundation/company/unknown)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--top-n",
            type=int,
            default=50,
            help="Number of rows per category CSV. Default: 50",
        )

    def handle(self, *args, **options):
        top_n = options["top_n"]
        if top_n < 1:
            raise CommandError("--top-n must be >= 1")

        created = create_funding_source_category_dataframes(top_n=top_n)
        self.stdout.write(self.style.SUCCESS("Created funding source category CSVs:"))
        for category, path in created.items():
            self.stdout.write(f"- {category}: {path}")
