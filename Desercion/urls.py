"""Desercion URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from gestion_estudiante import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('crear/', views.create),
    path('redes/', views.redes),
    path('inicio/', views.inicio),
    path('acerca/', views.acerca),
    path('index/', views.IndexView.as_view(), name = 'usuario_listar'),
    path('editar/<int:pk>/', views.Actualizar.as_view(), name= 'usuario_editar'),
    path('eliminar/<int:pk>/', views.Eliminar.as_view(), name= 'usuario_eliminar'),
]