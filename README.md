# ZEN OLMO
Taking the open source OLMO model and finetuning it to output zen quotes

## Data Preparation

### fetch_parse_texts.py
Fetches from web + parses a given list of texts into something consumable, with some aggressive parsing to dodge ugly artifacts of txt.

### high_quality_texts.txt
Contains links to seminal works of zen buddhism

### texts.txt
Contains links to random zen buddhist texts

## Finetuning strategy
To start we are going to do some simple PEFT tuning on the high quality datasets
