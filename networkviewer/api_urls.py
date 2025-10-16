from django.urls import path
from .api_views import (
    ChemicalSearchAPI,
    CompanySearchAPI,
    UniversitySearchAPI,
    ResearcherSearchAPI
)

urlpatterns = [
    path('chemicals/', ChemicalSearchAPI.as_view(), name='chemical-search-api'),
    path('companies/', CompanySearchAPI.as_view(), name='company-search-api'),
    path('universities/', UniversitySearchAPI.as_view(), name='university-search-api'),
    path('researchers/', ResearcherSearchAPI.as_view(), name='researcher-search-api'),
]