import requests
from bs4 import BeautifulSoup
import os
import sys
import re
import csv
import json
import datetime
import calendar
import time

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWebKit import *
from PyQt5.QtWebKitWidgets import *


#Take this class for granted.Just use result of rendering.
class Render(QWebPage):
  def __init__(self, url):
    self.app = QApplication(sys.argv)
    QWebPage.__init__(self)
    self.loadFinished.connect(self._loadFinished)
    self.mainFrame().load(QUrl(url))
    self.app.exec_()

  def _loadFinished(self, result):
    self.frame = self.mainFrame()
    self.app.quit()


match_template = 'https://www.over.gg/{}/iddqd-vs-c9-one-nation-of-gamers-invitational-onog-invitational-ub-r2'

def parse_page(id):
    page = requests.get(match_template.format(id))

    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup)
    r = Render(match_template.format(id))
    result = r.frame.toHtml()
    print(result)
    #print(soup.find_all('div'))

def scrape_data():
    for i in range(513,514):
        parse_page(i)

if __name__ == '__main__':
    scrape_data()

