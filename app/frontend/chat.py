#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import tornado.websocket

class ChatHandler(tornado.websocket.WebSocketHandler):

    waiters = set()
    messages = []
    def check_origin(self, origin):
        return True
    def open(self, *args, **kwargs):
        self.waiters.add(self)
        self.write_message({'messages': self.messages})

    def on_message(self, message):
        message = json.loads(message)
        self.messages.append(message)
        for waiter in self.waiters:
            if waiter == self:
                continue
            waiter.write_message({'img_path': message['img_path'], 'message': message['message']})

    def on_close(self):
        self.waiters.remove(self)
