from googleapiclient.discovery import build
import json
import urllib2
import os
import pprint
import shutil

cx = '003798233075176445716:gkxn-iv6be0'
apiKey='AIzaSyCSp5VA5cjQTRKDvPjOTNfFvJtVofvGcNA'

src_directory = "./src"

q = "cute kitten"

def download(link, mime):
      try:
            req = urllib2.Request(link)
            raw_img = urllib2.urlopen(req).read()
            image_type = "img"
            cntr = len([i for i in os.listdir(src_directory) if image_type in i]) + 1
            print cntr
            if mime == "image/jpeg":
                  f = open(os.path.join(src_directory , image_type + "_"+ str(cntr)+".jpg"), 'wb')
            else:
                  assert "Mimetype %s not handled" % mime
            f.write(raw_img)
            f.close()
      except Exception as e:
            print "could not load : "+link
            print e

def main():
      shutil.rmtree(src_directory)
      os.makedirs(src_directory)

      service = build("customsearch", "v1", developerKey=apiKey)
      res = service.cse().list(q=q, cx=cx, searchType="image", fileType="jpg",).execute()
      for item in res['items']:
            mime = item['mime']
            link = item['link']
            download(link, mime)
            
if __name__ == '__main__':
      main()
