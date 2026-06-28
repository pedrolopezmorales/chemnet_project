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
    #         #create_main_dataframe()
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