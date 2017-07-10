# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 16:35:43 2017

@author: Praveen Balasubramanian
"""

'''Import python packages'''
import pandas as pd
import sys , traceback
import datetime

"""Get all patent_ids that has the substring 'embod' in any of its image descriptions.
"""
def get_substring_patent_count(patents,substr_list):
    result = {}
    words_thatappear = {}
    sets = set(substr_list)
    
    for ids in patents:
        c=0
        s = []
        for image_desc in patents[ids]:
            if any(word in image_desc for word in substr_list): 
                c+=1
                s.extend(list(sets.intersection(set(image_desc.split()))))   
        if c != 0: 
            result[ids] = c
            words_thatappear[ids] = list(set(s))
    return result,words_thatappear
    
"""Function loops through embod dictionary keys (all patent_ids where 'embod' has appeared atleast
in any of its image descriptions) and checks if 'embod' substring is followed by the keyword 'invention'.
"""
def get_embod_inv_patent_count(patents,embod,substr_list,words_thatappear):
    result = {}
    for ids in embod:
        c=0
        for image_desc in patents[ids]:
            spl = image_desc.split()
            em = words_thatappear[ids]
            common_list = list(set(spl).intersection(set(em)))
            if common_list: c+= sum([check_invention(spl,common_word) for common_word in common_list])
        if c != 0: result[ids] = c
    return result
    
"""Function splits the image description list based on index to check if 'invention' appears 
after 'embod' substring/word. 
"""
def check_invention(image_desc_list,substr):
    c = 0
    try:
        ind = image_desc_list.index(substr)
        if 'invention' in image_desc_list[ind+1:]: c+=1
    except Exception:
        pass
    return c
    
"""Returns all words that contain a particular substring on it"""
def check_substr_words(set_word_list,substr):
    return [word for word in set_word_list if substr in word]
    
"""Builds a list of all words combining every image description"""
def get_words_list(patent_dict):
    words_list = []
    for ids in patent_dict:
        for itag in patent_dict[ids]:
            words_list.extend(itag.split())
    return words_list
    
"""Function returns a dictionary that has patent_ids as keys and the image descriptions as a list value
- ex., '4490921': ['the single figure of the drawing illustrates a preferred embodiment of the invention', 
                'brief description of the drawing']
"""
def get_data(dataframe):
    patents = {}
    # Loop through dataframe and store contents into dictionary.
    for x in range(len(dataframe)):
        currentid = dataframe.iloc[x,1]
        currentvalue = dataframe.iloc[x,2]
        patents.setdefault(currentid, [])
        patents[currentid].append(currentvalue.lower())
    return patents
        
def main():
    filepath = str(raw_input('Enter patent_drawing.csv filepath:'))
    #sample_filepath = 'C:\Users\Praveen\Desktop\DataScience\L3 test\patent_drawing.csv'
    
    try:
        #Read csv file using pandas library -> stores output as a dataframe
        df = pd.read_csv(filepath)
        
        #Parse the dataframe and load data into a dictionary with patent_ids as keys
        #and a list of lower cased image descriptions.
        patents = get_data(df)
        
        #Outputs a list containing all the words used in the image descriptions of 
        #all patents.
        words_list = get_words_list(patents)
        
        #Get all words that contains 'embod' as a substring
        embod_substr_list = check_substr_words(list(set(words_list)),'embod')
        
        #Outputs two dictionaries:
        # embod - > keys: patent_ids, value: integer (total no. of image descriptions in which the substr has appeared atleast once)
        # word -> keys: patent_ids, value: a list of all substrings that appear in any of its image descriptions) 
        # If our substring list that we passed had 6 words and if two words out of 6 appeared in 4 image descriptions atleast once for
        # patent_ids 123456 then -> word[123456] = [word1,word2]
        embod,word = get_substring_patent_count(patents,embod_substr_list)
        
        #Outputs a single dictionary:
        # embod_inv -> keys: patent_ids, value: integer (total no. of image descriptions where any word containing 'embod' is followed
        # by 'invention' keyword.)
        embod_inv = get_embod_inv_patent_count(patents,embod,embod_substr_list,word)
        
        print "Embod stats: \n"
        print "Total no. of patents that had the substring ‘embod’ with any ending: "+str(len(embod))
        print "Total no. of distinct image descriptions that had ‘embod’ with any ending : "+str(sum(embod.values()))+"\n"
        
        print "Embod and Invention co-occurrence stats: \n"
        print "Total no. of patents that had the substring ‘embod’ with any ending and also followed by the keyword ‘invention’ : "+str(len(embod_inv))
        print "Total no. of distinct image descriptions that had ‘embod’ word with any ending and followed by keyword ‘invention’ : "+str(sum(embod_inv.values()))
    
    except Exception:
        tb = sys.exc_info()[2]
        tbinfo = traceback.format_tb(tb)[0]
        pymsg = "PYTHON ERRORS:\nTraceback info:\n" + tbinfo + "\nError Info:\n" + str(sys.exc_info()[1])
        d = datetime.datetime.now()
        log = open('ERROR_Log.txt',"w")
        log.write("\n")
        log.write("__________________________________________")
        log.write("ERROR LOGS ")
        log.write("__________________________________________")
        log.write("\n")
        log.write("Log: " + str(d) + "\n")
        log.write("" + pymsg + "\n")
        log.close()
        print 'Unexpected exit !!! Check log file (ERROR_Log.txt)'
        

if __name__ == '__main__':
    main()