import re
import sys
import os
import nltk
from nltk.corpus import stopwords

class KeywordExtractor:
    def __init__(self):
        self.setup_nltk()
        self.stop_words = set(stopwords.words('english'))

    def setup_nltk(self):
        nltk_data_dir = os.path.expanduser('~/.cache/nltk')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)

        try:
            stopwords.words('english')
        except LookupError:
            print("Downloading NLTK stopwords...", file=sys.stderr)
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

    def extract_keywords(self, text):
        # Convert to lowercase and remove punctuation
        words = re.findall(r'\w+', text.lower())
        
        # Remove stop words and create a set of unique words
        keywords = [word for word in words if word not in self.stop_words]
        
        return list(set(keywords))

def main():
    extractor = KeywordExtractor()

        # Process stdin line by line
    for line in sys.stdin:
        # Strip whitespace and skip empty lines
        line = line.strip()
        if not line:
            continue
        
        # Extract keywords
        keywords = extractor.extract_keywords(line)
        
        # Write keywords to stdout, all on one line
        print(" ".join(keywords))

if __name__ == "__main__":
    main()

