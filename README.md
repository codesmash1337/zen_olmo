# ZEN OLMO
Taking the open source OLMO model and finetuning it to output zen quotes

## Data Preparation

### ```fetch_parse_texts.py```
Fetches from web + parses a given list of texts into something consumable, with some aggressive parsing to dodge ugly artifacts of txt.

#### Usage
```bash
# 1. Process URLs from default file (high_quality_texts.txt) and append to output
python fetch_parse_texts.py

# 2. Overwrite the output file instead of appending
python fetch_parse_texts.py --overwrite

# 3. Use a different input file
python fetch_parse_texts.py --file texts.txt

# 4. Specify a different output file
python fetch_parse_texts.py --output yoda.txt
```

### ```high_quality_texts.txt```
Contains links to seminal works of zen buddhism

### ```texts.txt```
Contains links to random zen buddhist texts

## Finetuning strategy
To start we are going to do some simple PEFT tuning on the high quality datasets
