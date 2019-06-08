#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tornado.ioloop
import tornado.options
import tornado.web
import chat
import index
import imageclass
import imagelist

base_dir = os.path.dirname(__file__)
handlers = [
    (r"/", index.IndexHandler),
    (r"/imageclass", imageclass.IndexHandler),
    (r"/chat", chat.ChatHandler),
    (r"/imagelist", imagelist.IndexHandler),
]

tornado.options.define(
    "port", default=8000, type=int, metavar="PORT", help="ポート番号")
tornado.options.define(
    "debug", default=False, type=bool, metavar="DEBUG", help="デバッグ")
tornado.options.define(
    "config", type=str, metavar="PATH", help="設定ファイルのパス",
    callback=lambda path: tornado.options.parse_config_file(path, final=False))

settings = {
    "template_path": os.path.join(base_dir, "templates"),
    "static_path": os.path.join(base_dir, "static"),
    "xsrf_cookies": True
}

def main():
    options = tornado.options.options
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers, **settings, debug=options.debug)
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()
