import os
import gzip
import json

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

fileName = 'Electronics'
inputFileName  = fileName + '.json.gz'
if not os.path.exists(fileName): os.makedirs(fileName)

count = 0
ii = 0
data = {}
asin = '0'
for review in parse(inputFileName):
    if review['asin'] != asin:
        outputFileName = fileName + r'/' + asin + '.json'
        outputFile = open(outputFileName, 'w', newline='')
        json.dump(data, outputFile)
        outputFile.close
        ii = 0
        data = {}
        asin = review['asin']
    try: helpfulness = review['vote']
    except: helpfulness = '0'
    try:
        data[ii] = {
            'helpfulness': helpfulness,
            'rating': review['overall'],
            'text': review['reviewText']}
        ii += 1
    except: pass
    count += 1
    if count % 1e5 == 0: print(count)