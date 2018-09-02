import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import re
from pathlib import Path
import os

# this has to be run outside of the script
# import pymultitor
# pymultitor.run(listen_port=1337,on_count=3) # pymultitor -lp=1337 --on-count=3


folder = 'midis/'

# activate tor proxy
# urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({'http': '127.0.0.1:1337'})))

# regex to extract filename from url
get_filename_fuge = re.compile('=.*')
get_filename_else = re.compile('\/[^\/]*')
for url in open("midi_links.txt"):
    # find all ".mid"-links in web pages
    page = BeautifulSoup(urllib.request.urlopen(url).read().decode('ISO-8859-1'))
    for link in page.find_all('a', href=True):
        if '.mid' in link['href'] or '.MID' in link['href']:
            # get midi file url
            midi_url = 'http://' + urllib.parse.urlsplit(url).netloc + '/' + link['href']
            midi_url = midi_url.replace('..', '')
            # get midi file name
            if 'kunstderfuge.com' in url:
                # special treatment for kunstderfuge.com
                filename = get_filename_fuge.findall(midi_url)
                path = folder + filename[0][1:].replace('/', '_')
            else:
                filename = get_filename_else.findall(midi_url)
                path = folder + filename[-1][1:]
            if not Path(path).is_file():
                # download midi
                print("Downloading:", midi_url)
                path, headers = urllib.request.urlretrieve(url=midi_url, filename=path)
                print("Saved:", path)
                file = open(path)
                # test if that has really been a midi file
                try:
                    for x in file:
                        if 'DOCTYPE HTML PUBLIC' in x:
                            print('ERROR: GOT HTML FILE.')
                            file.close()
                            os.remove(path)
                            print('REMOVED.')
                            # exit(400)
                        else:
                            break
                except Exception:
                    pass
