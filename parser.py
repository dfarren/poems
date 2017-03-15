from HTMLParser import HTMLParser
import cPickle
import re
import gzip

class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.poems = {}
        self.poem = False
        self.author = False
        self.title = False
        self.poem_line = False
        self.entity = False
        self.author_name = ''
        self.title_name = ''
        
    def handle_starttag(self, tag, attrs):

        if tag=='poem':
            self.poem = True
        elif self.poem:
            if tag=='a1':
                self.author = True
            elif tag=='t1':
                self.title = True
            elif tag=='l':
                for attr in attrs:
                    if attr[0]=='ln':
                        self.poem_line = True

    def handle_data(self, data):

        if self.poem:
            pattern = re.compile('[\W_]+')
            data = pattern.sub('', data)

            if self.author and data not in self.poems:
                # if there is &something in the data string
                if self.entity:
                    del self.poems[self.author_name]
                    self.author_name += data
                else:
                    self.author_name = data
                self.poems[self.author_name] = {}

            elif self.title and data not in self.poems[self.author_name]:

                # if there is &something in the data string
                if self.entity:
                    del self.poems[self.author_name][self.title_name]
                    self.title_name += data
                else:
                    self.title_name = data
                self.poems[self.author_name][self.title_name] = ''

            elif self.poem_line:
                self.poems[self.author_name][self.title_name] += data + ' '

    # TODO: fix this so that entities are ignored and don't split titles
    # def handle_entityref(self, name):
    #     if (self.author and self.author_name != '') or (self.title and self.title_name != ''):
    #         self.entity = True

    def handle_endtag(self, tag):
        if tag=='poem':
            self.poem = False
        elif self.poem:
            if tag=='a1':
                self.author = False
                self.entity = False
            elif tag=='t1':
                self.title = False
                self.entity = False
            elif tag=='l' and self.poem_line:
                self.poem_line = False


def parse(poems_path):
    from os import walk

    parser = MyHTMLParser()
    poems_full = {}

    for (dirpath, dirnames, filenames) in walk(mypath):
        print "processing %s" % dirpath
        for file in filenames:
            with open(dirpath+'/'+file, 'r') as f:
                parser.feed(f.read())
                poems_full.update(parser.poems)

    return poems_full


# def add_labels(poems_file_loc):
#     survival = {}
#
#     with open(poems_file_loc, 'rb') as f:
#         poems = cPickle.load(f)
#
#     for author, author_dct in poems.iteritems():
#         for title in author_dct.iterkeys():
#             #first key without permutations
#             try:
#                 response = get_etexts('title', title)
#             except:
#                 response = frozenset([])
#
#             if response != frozenset([]):
#                 survival[title] = None
#                 continue
#
#             #use a bag of words approach because some titles have strange symbols and are slightly different than the original titles
#             for permutation in permute_words(title):
#                 try:
#                     response = get_etexts('title', permutation)
#                 except:
#                     response = frozenset([])
#
#                 if response != frozenset([]):
#                     survival[title] = None
#                     break
#
#     return survival

def match_poems_with_gutenberg(poems_file_loc, gutenberg_file_loc):
    jaccard_similarity = {}

    with open(poems_file_loc, 'rb') as f:
        poems = cPickle.load(f)

    with gzip.open(gutenberg_file_loc, 'rb') as f:
        gutenberg_titles = cPickle.load(f)

    for author, author_dct in poems.iteritems():
        for poem_title in author_dct.iterkeys():

            jaccard_similarity[poem_title] = 0.0
            for gutenberg_title in gutenberg_titles:
                jaccard_similarity[poem_title] = max(calculate_jaccard_similarity(frozenset(poem_title.split()), gutenberg_title), jaccard_similarity[poem_title])

    return jaccard_similarity


def calculate_jaccard_similarity(set_a, set_b):
    return float(len(set_a.intersection(set_b))) / len(set_a.union(set_b))


def count_poems(poems):
    cnt = 0
    for author, author_dct in poems.iteritems():
        cnt += len(author_dct)

    return cnt


if __name__=='__main__':

    mypath = 'chadh-poetry/'
    poems_path = 'poems.pkl'
    labels_path = 'labels.pkl'
    PICKLEFILE = '/tmp/md.pickle.gz'

    print "parsing poems corpus"
    with open(poems_path, 'wb') as f:
       cPickle.dump(parse(mypath), f)

    print "matching poems with gutenberg"
    with open(labels_path, 'wb') as f:
        cPickle.dump(match_poems_with_gutenberg(poems_path, PICKLEFILE), f)

    print "done"

    #parser = MyHTMLParser()
    #parser.feed('<poem><a1>pepito</a1><T1> From &ldquo;The Minstrel Girl.&rdquo;&mdash;James G.Whittier. </T1></poem>')
    #print parser.poems
