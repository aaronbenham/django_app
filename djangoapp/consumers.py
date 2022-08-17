from channels.generic.websocket import WebsocketConsumer

import json
from random import randint
from time import sleep

from .views import standard_scores

#
#
# class WSConsumer(WebsocketConsumer):
#     def connect(self):
#         self.accept()
#
#         for i in range(1000):
#             arr = update()
#             self.send(json.dumps({'message': arr}))
#             sleep(1)