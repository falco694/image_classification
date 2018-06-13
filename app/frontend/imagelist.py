#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tornado.web
import os
import base64
import csv

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        imagelist = []
        with open('database.tsv', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                imagelist.append(row)
        imagelist.reverse()
        self.render("imagelist.html", imagelist=imagelist)