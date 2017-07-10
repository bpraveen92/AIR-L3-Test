# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 22:34:05 2017

@author: Praveen
"""

from itertools import izip

def hamming_distance(w1,w2):
    cnt=0
    for str1,str2 in zip(w1.split(),w2.split()):
        for i,(c1, c2) in enumerate(izip(str1, str2)):
            if c1.lower() in ('s','z') and c2.lower() in ('s', 'z'): pass
            elif i == 0 and  c1.lower() == c2.lower() and (c1.isupper() or c2.isupper()): pass
            elif i == 0 and c1.lower() != c2.lower(): cnt+=1
            else:
                if c1.lower() != c2.lower() and ((c1.isupper() or c2.isupper())): cnt+= 1.5
                elif c1.lower() != c2.lower(): cnt += 1
                elif c1.lower() == c2.lower() and ((c1.isupper() or c2.isupper())): cnt+= 0.5
                    
                
    return cnt
    
print hamming_distance('make', 'Mage')
print hamming_distance('MaiSY', 'MaiZy')
print hamming_distance('Eagle', 'Eager')
print hamming_distance('Sentences work too.', 'Sentences wAke too.')