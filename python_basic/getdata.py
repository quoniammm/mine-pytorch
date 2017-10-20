from urllib.request import urlopen

with urlopen('http://localhost:3000/comment/music?id=186016&limit=1') as pinglun:
    print(type(pinglun))