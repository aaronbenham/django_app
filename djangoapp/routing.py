from django.urls import path

from .consumers import WSConsumer

ws_urlpatterns = [
    path('ws/online_adversarial_detection/', WSConsumer.as_asgi())
]