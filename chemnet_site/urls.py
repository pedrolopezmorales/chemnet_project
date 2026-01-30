"""
URL configuration for chemnet_site project.

"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('networkviewer.api_urls')),  # API endpoints
    path('', include('networkviewer.urls')),  # Original HTML views
]

# Serve static files during development
# Note: In production, web server (Apache/Nginx) should serve static files
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)