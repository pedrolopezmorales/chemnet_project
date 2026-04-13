import os

from django.apps import AppConfig


class NetworkviewerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "networkviewer"


    # def ready(self):
    #     from .network_functions import create_funding_source_dataframe
    #     import os 
    #     from django.conf import settings
    
    #     try:
    #         print("🔄 Building funding source enriched dataframe...")
    #         funding_source_table_df = create_funding_source_dataframe(chem_limit=5)
    #         csv_path = os.path.join(settings.BASE_DIR, 'data', 'funding_source_table_df.csv')
    #         funding_source_table_df.to_csv(csv_path, index=False)
    #         print(f"✓ Funding dataframe saved to {csv_path}")
    #         print(f"  Shape: {funding_source_table_df.shape}")

    #     except Exception as e:
    #         print(f"⚠ Error creating funding source dataframe: {e}")
    #         import traceback
    #         traceback.print_exc()

    # def ready(self):
    #     import os 
    #     if os.environ.get("RUN_MAIN") != "true":
    #         return

    #     try:
    #         from . import dataframes_creation
    #         from django.conf import settings

    #         csv_path = os.path.join(settings.BASE_DIR, "data", "comparing_fundingsources.csv")
    #         dataframes_creation.company_assoc.to_csv(csv_path, index=False)
    #         print(f"Saved company assoc dataframe to {csv_path}")

    #     except Exception as e:
    #         print(f"Error creating company assoc dataframe: {e}")