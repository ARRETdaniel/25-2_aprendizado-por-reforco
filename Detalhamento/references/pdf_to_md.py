import sys
from pathlib import Path
import PyPDF2

def pdf_to_md(pdf_path, md_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(f"# Extracted from: {pdf_path}\n\n")
        md_file.write(text)

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "contextual"
    output_dir.mkdir(exist_ok=True)
    # If no arguments, process all .pdf files in the current directory
    if len(sys.argv) < 2:
        pdf_files = list(Path(__file__).parent.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in the current directory.")
            sys.exit(0)
        print(f"No arguments provided. Converting all PDFs in {Path(__file__).parent}.")
    else:
        pdf_files = [Path(arg) for arg in sys.argv[1:]]

    for pdf_path in pdf_files:
        if pdf_path.suffix.lower() != ".pdf" or not pdf_path.exists():
            print(f"Skipping {pdf_path}: not a valid PDF file.")
            continue
        md_name = pdf_path.stem + ".md"
        md_path = output_dir / md_name
        pdf_to_md(pdf_path, md_path)
        print(f"Converted {pdf_path} -> {md_path}")
