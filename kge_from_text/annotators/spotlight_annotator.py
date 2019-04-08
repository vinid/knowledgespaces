import re
import requests, json
import spotlight
import os, random


from kge_from_text.annotators import annotators


class DBpediaSpotlightAnnotator(annotators.Annotator):

    def __init__(self, endpoint_url):
        super().__init__()
        self.endpoint_url = endpoint_url

    def annotate(self,  string):
        annotations = []
        try:
            annotations = spotlight.annotate(self.endpoint_url,
            string,
            confidence = 0.5, support = 0)
        except Exception as e:
            print(e)
            pass
        return annotations

    def extract_surfaces_and_entities(self, string):
        annotation_output = self.annotate(string)
        list_of_surfaces_and_entities = []
        for k in annotation_output:
            list_of_surfaces_and_entities.append((k["surfaceForm"], k["URI"].replace("http://dbpedia.org/resource/", "")))

            #whole_string = (" ".join(k["surfaceForm"].split()) + "\t" + k["URI"].replace("http://dbpedia.org/resource/",
            #                                                                             "") + "\n").encode("utf-8")
        return list_of_surfaces_and_entities



if __name__ == "__main__":

    dbp = DBpediaSpotlightAnnotator("http://model.dbpedia-spotlight.org/en/annotate")
    string = "Chișinău also known as Kishinev (Russian: Кишинёв, tr. Kishinyov), is the capital and largest city of the Republic of Moldova. The city is Moldova's main industrial and commercial center, and is located in the middle of the country, on the river Bîc. According to the results of the 2014 census, the city proper had a population"
    print(dbp.annotate(string))
    print(dbp.extract_surfaces_and_entities(string))
