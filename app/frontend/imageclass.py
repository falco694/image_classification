#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import uuid
import tornado.web
import prediction

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("imageclass.html", pictureid="", result="")

    def post(self):
        try:
            files = self.request.files
            file = files['file'][0]['body']
            mime = files['file'][0]['content_type']
            pictureid = str(uuid.uuid4())
            picturefullpath = "static/uploaddata/" + pictureid + ".jpeg"
            with open(picturefullpath, 'wb') as f:
                f.write(file)
            # クラス判定
            result = prediction.main(picturefullpath)
            # 結果を保存
            with open('database.tsv', 'a') as f:
                f.write(pictureid + "\t" + str(result) + "\n")
            self.render("imageclass.html", pictureid=pictureid, result=result)
            # self.write("ファイルを保存しました")
        except KeyError:
            self.render("imageclass.html", pictureid="", result="")
