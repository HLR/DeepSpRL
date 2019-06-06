'''
Name: Chen Zheng
Institution: Tulane University
python file: data_helpers.py
Purpose: 1. decode xml file of our sprl dataset
         2. merge the data
'''

from xml.dom import minidom
import xml.sax


class SprlHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        # self.SENTENCE = ""
        self.CONTENT = ""
        self.TRAJECTOR = ""
        self.LANDMARK = ""
        self.SPATIAL_INDICATOR = ""
        self.RELATION = ""

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "SENTENCE":
            print("*****SENTENCE*****")
            id = attributes["id"]
            print("id:", str(id))

    def endElement(self, tag):
        if self.CurrentData == "CONTENT":
            print("CONTENT:", str(self.CONTENT))
        elif self.CurrentData == "TRAJECTOR":
            print("TRAJECTOR:", str(self.TRAJECTOR))
        elif self.CurrentData == "LANDMARK":
            print("LANDMARK:", str(self.LANDMARK))
        elif self.CurrentData == "SPATIAL_INDICATOR":
            print("SPATIAL_INDICATOR:", str(self.SPATIAL_INDICATOR))
        elif self.CurrentData == "RELATION":

            print("RELATION:", str(self.RELATION))
        self.CurrentData = ""

    def characters(self, content):
        if self.CurrentData == "CONTENT":
            self.CONTENT = content
        elif self.CurrentData == "TRAJECTOR":
            self.TRAJECTOR = content
        elif self.CurrentData == "LANDMARK":
            self.LANDMARK = content
        elif self.CurrentData == "SPATIAL_INDICATOR":
            self.SPATIAL_INDICATOR = content
        elif self.CurrentData == "RELATION":
            self.RELATION = content



# # create XMLReader
# parser = xml.sax.make_parser()
# # turn off namepsaces
# parser.setFeature(xml.sax.handler.feature_namespaces, 0)
#
# # re-written ContextHandler
# Handler = SprlHandler()
# parser.setContentHandler(Handler)
#
# parser.parse('data/SpRL-2012-Gold.xml')
'''
http://www.runoob.com/python/python-xml.html
'''





def decode_xml_SENTENCE(file_name):
    xmldoc = minidom.parse(file_name)
    itemlist = xmldoc.getElementsByTagName("SENTENCE")
    print(len(itemlist))
    print(itemlist[0].attributes['id'].value)
    for s in itemlist:
        print(s.attributes['id'].value)

def decode_xml_CONTENT(file_name):
    xmldoc = minidom.parse(file_name)
    itemlist = xmldoc.getElementsByTagName("CONTENT")
    print(len(itemlist))
    print(itemlist[0].attributes['id'].value)
    for s in itemlist:
        print(s.attributes['id'].value)

def decode_xml_TRAJECTOR(file_name):
    xmldoc = minidom.parse(file_name)
    itemlist = xmldoc.getElementsByTagName("TRAJECTOR")
    print(len(itemlist))
    print(itemlist[0].attributes['id'].value)
    for s in itemlist:
        print(s.attributes['id'].value)

def decode_xml_LANDMARK(file_name):
    xmldoc = minidom.parse(file_name)
    itemlist = xmldoc.getElementsByTagName("LANDMARK")
    print(len(itemlist))
    print(itemlist[0].attributes['id'].value)
    for s in itemlist:
        print(s.attributes['id'].value)

# decode_xml_CONTENT('data/SpRL-2012-Gold.xml')



class extract_content(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        # self.SENTENCE = ""
        self.CONTENT = ""
        self.TRAJECTOR = ""
        self.LANDMARK = ""
        self.SPATIAL_INDICATOR = ""
        self.RELATION = ""

    def startElement(self, tag, attributes):
        self.CurrentData = tag

    def endElement(self, tag):
        if self.CurrentData == "CONTENT":
            print(str(self.CONTENT))
        self.CurrentData = ""

    def characters(self, content):
        if self.CurrentData == "CONTENT":
            self.CONTENT = content

# create XMLReader
parser = xml.sax.make_parser()
# turn off namepsaces
parser.setFeature(xml.sax.handler.feature_namespaces, 0)

# re-written ContextHandler
Handler = extract_content()
parser.setContentHandler(Handler)

parser.parse('data/SpRL-2012-Train.xml')
