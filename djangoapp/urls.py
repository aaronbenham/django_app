from django.contrib import admin
from django.urls import path
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.home),
    # path("test/", TemplateView.as_view(template_name="index.html"))
    # path("test/", views.test, name="web_cam")
]