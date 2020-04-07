import pandas as pd

fileName = open('Electronics/B0002SQ2P2.json')

data = pd.read_json(fileName)

print(data)
