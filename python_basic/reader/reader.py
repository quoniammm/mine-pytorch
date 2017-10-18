import os

from compressed import bzipped, gzipped

extension_map = {
    '.bz2':
    'gz':
}

class Reader:
    def __init__(self, filename):
        self.filename = filename
        self.f = open(self.filename, 'r')

    def read(self):
        return self.f.read()

    def close(self):
        self.f.close()