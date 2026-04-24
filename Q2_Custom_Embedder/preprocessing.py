import os
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd
from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.strip()
    return text


def extract_pdf_data(pdf_path: str) -> Dict:
    reader = PdfReader(pdf_path)
    pages_text = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned = clean_text(text)
            pages_text.append({
                'page_number': i + 1,
                'text': cleaned,
                'source': os.path.basename(pdf_path)
            })
    
    return {
        'source': os.path.basename(pdf_path),
        'pages': pages_text,
        'num_pages': len(pages_text)
    }


def extract_all_pdfs(data_dir: str) -> pd.DataFrame:
    all_pages = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        try:
            data = extract_pdf_data(pdf_path)
            all_pages.extend(data['pages'])
            print(f"Extracted {data['num_pages']} pages from {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    df = pd.DataFrame(all_pages)
    return df


def tokenize_text(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens


def create_corpus(df: pd.DataFrame) -> List[str]:
    corpus = df['text'].tolist()
    return corpus


def main():
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent
    
    print("Extracting text from PDFs...")
    df = extract_all_pdfs(str(data_dir))
    
    corpus_path = output_dir / "corpus.csv"
    df.to_csv(corpus_path, index=False)
    print(f"Saved {len(df)} pages to {corpus_path}")
    
    print("\nCorpus statistics:")
    print(f"Total pages: {len(df)}")
    print(f"Documents: {df['source'].nunique()}")
    print(f"Average text length: {df['text'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()