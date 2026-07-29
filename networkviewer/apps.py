import os
import shutil

from django.apps import AppConfig


class NetworkviewerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "networkviewer"


    # def ready(self):
    #     from .network_functions import create_funding_source_category_dataframes
    #     from django.conf import settings
    
    #     try:
    #         print("🔄 Building funding source tables (all + categories)...")
    #         created_files = create_funding_source_category_dataframes(top_n=50)

    #         # Keep the legacy all-table filename in sync for existing references.
    #         all_table_path = created_files.get('all')
    #         legacy_path = os.path.join(settings.BASE_DIR, 'data', 'funding_source_table_df.csv')
    #         if all_table_path and os.path.exists(all_table_path):
    #             shutil.copy2(all_table_path, legacy_path)

    #         print("✓ Funding source tables created:")
    #         for category, path in created_files.items():
    #             print(f"  {category}: {path}")
    #         print(f"✓ Legacy all-table copy saved to {legacy_path}")

    #     except Exception as e:
    #         print(f"⚠ Error creating funding source tables: {e}")
    #         import traceback
    #         traceback.print_exc()

    # def ready(self):
    #     import os
    #     if os.environ.get("RUN_MAIN") != "true":
    #         return

    #     try:
    #         from .dataframes_creation import create_company_assoc_dataframe
    #         create_company_assoc_dataframe()

    #     except Exception as e:
    #         print(f"Error creating company assoc dataframe: {e}")


    # def ready(self):
    #     import os
    #     if os.environ.get("RUN_MAIN") != "true":
    #         return

    #     try:
    #         from .dataframes_creation import create_main_dataframe, create_filtered_main_dataframe
    #         # create_main_dataframe()
    #         create_filtered_main_dataframe()
    #     except Exception as e:
    #         print(f"Error creating main dataframe: {e}")

    
    # def ready(self):
    #     import os
    #     if os.environ.get("RUN_MAIN") != "true":
    #         return

    #     try:
    #         from .dataframes_creation import create_company_classifications
    #         create_company_classifications()
    #     except Exception as e:
    #         print(f"Error creating main dataframe: {e}")