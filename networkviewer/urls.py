from django.urls import path
from . import views

urlpatterns = [
    path('chemical/', views.chemical_view, name='chemical_view'),
    path('company/',  views.company_view, name='company_view'),
    path('university/', views.university_view, name='university_view'),
    path('researcher/', views.researcher_view, name='researcher_view'),
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('data/', views.data_view, name='data'),    
    path('contact/', views.contact_view, name='contact')
]