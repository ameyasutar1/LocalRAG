import fitz
import zipfile
import shutil
import pathlib
import re
import threading
import os
import base64
import json
from typing import Optional, Tuple, List, Dict, Any

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# RAG specific imports
import chromadb
# Removed 'ollama' import as we are no longer using it for embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

# ------------------------------------------------------------
# (A) IMAGE‐EXTRACTION FUNCTIONS for different file types
# ------------------------------------------------------------

def extract_images_from_pdf(pdf_path: str, output_folder: str) -> None:
    pdf = fitz.open(pdf_path)
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    img_count = 0
    for page in pdf:
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            img_dict = pdf.extract_image(xref)
            img_bytes = img_dict["image"]
            ext = img_dict["ext"]
            img_count += 1
            img_filename = f"image{img_count}.{ext}"
            img_path = output_folder / img_filename
            with open(img_path, "wb") as f_img:
                f_img.write(img_bytes)

    print(f"Extracted {img_count} images from PDF → {output_folder}")


def extract_images_from_docx(docx_path: str, output_folder: str) -> None:
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx_path, "r") as zf:
        all_entries = zf.namelist()
        media_files = [e for e in all_entries if e.startswith("word/media/")]

        if not media_files:
            print(f"⚠ No images found in {docx_path}")
            return

        for idx, media_rel in enumerate(media_files, start=1):
            ext = pathlib.Path(media_rel).suffix
            img_filename = f"image{idx}{ext}"
            img_path = output_folder / img_filename # This creates the full path

            with zf.open(media_rel) as embedded_f:
                data = embedded_f.read()
                with open(img_path, "wb") as out_f:
                    out_f.write(data)

        print(f"Extracted {len(media_files)} images from DOCX → {output_folder}")


def extract_images_from_pptx(pptx_path: str, output_folder: str) -> None:
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(pptx_path, "r") as zf:
        all_entries = zf.namelist()
        media_files = [e for e in all_entries if e.startswith("ppt/media/")]

        if not media_files:
            print(f"⚠ No images found in {pptx_path}")
            return

        for idx, media_rel in enumerate(media_files, start=1):
            ext = pathlib.Path(media_rel).suffix
            img_filename = f"image{idx}{ext}"
            img_path = output_folder / img_filename # This creates the full path

            with zf.open(media_rel) as embedded_f:
                data = embedded_f.read()
                with open(img_path, "wb") as out_f:
                    out_f.write(data)

        print(f"Extracted {len(media_files)} images from PPTX → {output_folder}")


def extract_images_from_xlsx(xlsx_path: str, output_folder: str) -> None:
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(xlsx_path, "r") as zf:
        all_entries = zf.namelist()
        media_files = [e for e in all_entries if e.startswith("xl/media/")]

        if not media_files:
            print(f"⚠ No images found in {xlsx_path}")
            return

        for idx, media_rel in enumerate(media_files, start=1):
            ext = pathlib.Path(media_rel).suffix
            img_filename = f"image{idx}{ext}"
            img_path = output_folder / img_filename # This creates the full path

            with zf.open(media_rel) as embedded_f:
                data = embedded_f.read()
                with open(img_path, "wb") as out_f:
                    out_f.write(data)

        print(f"Extracted {len(media_files)} images from XLSX → {output_folder}")


def extract_image_file(image_path: str, output_folder: str) -> None:
    output_folder = pathlib.Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    src = pathlib.Path(image_path)
    ext = src.suffix
    img_path = output_folder / f"image1{ext}"
    shutil.copy2(src, img_path)
    print(f"Copied standalone image → {img_path}")


def extract_images_generic(input_path: str, output_folder: str) -> None:
    suffix = pathlib.Path(input_path).suffix.lower()

    if suffix == ".pdf":
        extract_images_from_pdf(input_path, output_folder)

    elif suffix == ".docx":
        extract_images_from_docx(input_path, output_folder)

    elif suffix == ".pptx":
        extract_images_from_pptx(input_path, output_folder)

    elif suffix == ".xlsx":
        extract_images_from_xlsx(input_path, output_folder)

    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        extract_image_file(input_path, output_folder)

    elif suffix == ".csv":
        # CSVs do not bundle images—skip extraction
        print(f"⚠ Skipping image extraction for CSV: {input_path}")

    else:
        print(f"⚠ Unsupported extension '{suffix}' for image extraction: {input_path}")


# ------------------------------------------------------------
# (B) DOC‐TO‐MARKDOWN (OCR + text + tables, ignoring images)
# ------------------------------------------------------------

def convert_to_markdown_any(
    input_path: str,
    output_dir: str,
    converter: Optional[DocumentConverter] = None
) -> None:
    if converter is None:
        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = True
        pipeline_opts.do_table_structure = True
        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.IMAGE,
                InputFormat.XLSX,
                InputFormat.CSV,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            }
        )

    result = converter.convert(input_path)
    stem = pathlib.Path(input_path).stem
    out_md_path = pathlib.Path(output_dir) / f"{stem}.md"

    raw_md = result.document.export_to_markdown()
    with open(out_md_path, "w", encoding="utf-8") as f_md:
        f_md.write(raw_md)

    print(f"Docling OCR → Markdown: {out_md_path}")


# ------------------------------------------------------------
# (C) PATCH MARKDOWN PLACEHOLDERS → ACTUAL ![…](…) TAGS
# ------------------------------------------------------------

def patch_markdown_with_images(md_path: str, images_folder: str) -> None:
    md_path = pathlib.Path(md_path)
    base_dir = md_path.parent
    folder = pathlib.Path(images_folder)

    extracted = sorted(folder.iterdir(), key=lambda p: p.name)
    if not extracted:
        print(f"⚠ No images found in {images_folder}. Nothing to patch in {md_path}.")
        return

    text = md_path.read_text(encoding="utf-8")

    def replacement_fn(match_obj):
        if not extracted:
            return "" 

        img_file = extracted.pop(0)
        rel = img_file.relative_to(base_dir)
        fig_num = img_file.stem.replace("image", "")
        return f"![Figure {fig_num}]({rel})"

    patched = re.sub(r'', replacement_fn, text)
    md_path.write_text(patched, encoding="utf-8")
    print(f"Patched Markdown → {md_path}")


# ------------------------------------------------------------
# (D) UTILITY FUNCTION: PROCESS A SINGLE FILE IN PARALLEL FOR
#    IMAGE EXTRACTION + DOCLING CONVERSION
# ------------------------------------------------------------

def process_file(input_path: str, output_base: str
                ) -> Tuple[str, str]:
    """
    Given any supported file (PDF, DOCX, PPTX, IMAGE, XLSX, CSV), this function:
      1) Extracts embedded images (or copies the standalone image) into
         <output_base>/<stem>_images/
      2) Runs Docling → Markdown (text + tables) into <output_base>/<stem>.md
      3) Patches the generated Markdown so each becomes
         a proper Markdown tag pointing at the extracted images

    Returns:
        (md_filepath, images_folderpath)
    """
    stem = pathlib.Path(input_path).stem
    img_folder = pathlib.Path(output_base) / f"{stem}_images"
    md_path = pathlib.Path(output_base) / f"{stem}.md"

    # 1) Make sure the output base folder exists
    pathlib.Path(output_base).mkdir(parents=True, exist_ok=True)

    # 2) Build a single DocumentConverter once, to reuse for Markdown conversion
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = True
    pipeline_opts.do_table_structure = True

    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.IMAGE,
            InputFormat.XLSX,
            InputFormat.CSV,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
        }
    )

    # 3) Run image extraction + Markdown generation in parallel threads
    t1 = threading.Thread(
        target=extract_images_generic,
        args=(input_path, str(img_folder))
    )
    t2 = threading.Thread(
        target=convert_to_markdown_any,
        args=(input_path, output_base, converter)
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 4) Once both threads are done, patch the Markdown
    patch_markdown_with_images(str(md_path), str(img_folder))

    # 5) Return the paths so caller knows where to look
    return (str(md_path), str(img_folder))


# ===============================================================
# NEW RAG PIPELINE COMPONENTS (with unified embedder)
# ===============================================================

class UnifiedEmbedder:
    """
    A wrapper class to handle embeddings using a single model for both text and images.
    """
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for SentenceTransformer")
        try:
            self.embedder = SentenceTransformer(model_name, device=self.device)
            print(f"SentenceTransformer model '{model_name}' loaded successfully for both text and images.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            print("Please ensure the model name is correct and you have an internet connection for the first download.")
            raise

    def get_text_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text using SentenceTransformer."""
        try:
            embedding = self.embedder.encode(text).tolist()
            return embedding
        except Exception as e:
            print(f"Error generating text embedding with SentenceTransformer: {e}")
            return []

    def get_image_embedding(self, image_path: str) -> List[float]:
        """
        Generates an embedding for the given image using SentenceTransformer.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            embedding = self.embedder.encode(img).tolist()
            return embedding
        except Exception as e:
            print(f"Error generating embedding for image {image_path} with SentenceTransformer: {e}")
            return [] # Return empty list on error

class ChromaDBManager:
    """
    Manages interactions with a ChromaDB collection.
    """
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "document_embeddings"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Initialize UnifiedEmbedder to use a single model for both text and images
        self.unified_embedder = UnifiedEmbedder(model_name="sentence-transformers/clip-ViT-B-32") 
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # No need to explicitly set embedding_function here. ChromaDB infers it
            # from the first insertion. By ensuring UnifiedEmbedder uses only one
            # type of embedding, the dimension will be consistent.
        )
        print(f"Connected to ChromaDB collection: '{collection_name}' at '{persist_directory}'")

    def add_document_chunk(self,
                           text_chunk: str,
                           filename: str,
                           db_fileid: str,
                           chunk_id: int,
                           source_type: str = "text"
                          ) -> None:
        """Adds a text chunk and its embedding to ChromaDB."""
        embedding = self.unified_embedder.get_text_embedding(text_chunk)
        if not embedding:
            print(f"Skipping empty embedding for text chunk {chunk_id} in {filename}")
            return

        doc_id = f"{db_fileid}_text_chunk_{chunk_id}"
        metadata = {
            "filename": filename,
            "db_fileid": db_fileid,
            "chunk_id": chunk_id,
            "source_type": source_type,
            "content_type": "text"
        }
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[text_chunk],
                metadatas=[metadata],
                ids=[doc_id]
            )
            # print(f"Added text chunk {chunk_id} from '{filename}' to ChromaDB.")
        except Exception as e:
            print(f"Error adding text chunk {chunk_id} from '{filename}' to ChromaDB: {e}")

    def add_image_document(self,
                           image_path: str,
                           filename: str,
                           db_fileid: str,
                           image_id: int,
                           source_type: str = "image"
                          ) -> None:
        """Adds an image embedding to ChromaDB."""
        embedding = self.unified_embedder.get_image_embedding(image_path)
        if not embedding:
            print(f"Skipping empty embedding for image {image_path} in {filename}")
            return

        doc_id = f"{db_fileid}_image_{image_id}"
        metadata = {
            "filename": filename,
            "db_fileid": db_fileid,
            "image_path": image_path,
            "image_id": image_id,
            "source_type": source_type,
            "content_type": "image"
        }
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[f"Image: {os.path.basename(image_path)}"], # Store a simple description
                metadatas=[metadata],
                ids=[doc_id]
            )
            print(f"Added image '{os.path.basename(image_path)}' from '{filename}' to ChromaDB.")
        except Exception as e:
            print(f"Error adding image '{os.path.basename(image_path)}' from '{filename}' to ChromaDB: {e}")

    def query_documents(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Queries the ChromaDB collection with a text query."""
        query_embedding = self.unified_embedder.get_text_embedding(query_text)
        if not query_embedding:
            print("Could not generate embedding for query text.")
            return {}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return results

    def query_images(self, image_path: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Queries the ChromaDB collection with an image query."""
        query_embedding = self.unified_embedder.get_image_embedding(image_path)
        if not query_embedding:
            print("Could not generate embedding for query image.")
            return {}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"content_type": "image"}, # Only query image embeddings
            include=['documents', 'metadatas', 'distances']
        )
        return results


def process_and_embed_file(
    md_filepath: str,
    images_folderpath: str,
    filename: str,
    db_fileid: str,
    chroma_manager: ChromaDBManager
) -> None:
    """
    Takes the processed markdown and image folder, generates embeddings,
    and stores them in ChromaDB.
    """
    print(f"\n--- Processing for embedding: {filename} (ID: {db_fileid}) ---")

    # 1. Text Splitting and Embedding
    try:
        md_content = pathlib.Path(md_filepath).read_text(encoding="utf-8")
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        text_chunks = text_splitter.split_text(md_content)
        print(f"Split '{filename}' into {len(text_chunks)} text chunks.")

        for i, chunk in enumerate(text_chunks):
            chroma_manager.add_document_chunk(chunk, filename, db_fileid, i)
    except Exception as e:
        print(f"Error processing text for {filename}: {e}")

    # 2. Image Embedding
    try:
        image_files = sorted(pathlib.Path(images_folderpath).iterdir(), key=lambda p: p.name)
        if not image_files:
            print(f"No images found in '{images_folderpath}' for embedding.")
            return

        print(f"Found {len(image_files)} images in '{images_folderpath}'.")
        for i, img_path in enumerate(image_files):
            chroma_manager.add_image_document(str(img_path), filename, db_fileid, i)
    except Exception as e:
        print(f"Error processing images for {filename}: {e}")

    print(f"--- Finished embedding for: {filename} ---")


# ------------------------------------------------------------
# (E) EXAMPLE: HOW YOU MIGHT CALL process_file(...) FOR EACH INPUT
# ------------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    # List of files to process (replace with your actual files)
    files_to_process = [
        r"C:\Users\ameya\Downloads\NIPS-2017-attention-is-all-you-need-Paper.pdf",
        # Add more files here if needed
        # r"C:\path\to\your\document.docx",
        # r"C:\path\to\your\image.png"
    ]
    output_base_dir = "converted_docs_and_images" # Directory to save MD and extracted images
    chroma_db_persist_dir = "./rag_chroma_db" # Directory to store ChromaDB

    # Initialize ChromaDB Manager
    # This will now use SentenceTransformer for all embeddings.
    chroma_manager = ChromaDBManager(persist_directory=chroma_db_persist_dir)

    processed_results = []
    for fpath in files_to_process:
        # Generate a unique ID for the file in the database
        # You can use a more robust UUID generation in a real application
        db_fileid = f"doc_{pathlib.Path(fpath).stem}_{os.urandom(4).hex()}"
        filename = pathlib.Path(fpath).name # Original filename for metadata

        # Step 1: Process file with Docling to get Markdown and extracted images
        md_output_path, images_output_folder = process_file(fpath, output_base_dir)
        processed_results.append((fpath, md_output_path, images_output_folder, filename, db_fileid))

        # Step 2: Generate embeddings and store in ChromaDB
        process_and_embed_file(
            md_output_path,
            images_output_folder,
            filename,
            db_fileid,
            chroma_manager
        )

    print("\n All files processed and embeddings generated. Summary:")
    for inp, mdp, imd, fn, fid in processed_results:
        print(f"   • Original: '{inp}'\n    → Markdown: '{mdp}'\n    → Images folder: '{imd}'\n    → DB ID: '{fid}'")

    # --- Example Querying the ChromaDB ---
    print("\n--- Example Querying ---")
    
    # Query for text
    query_text = "What is self attention ?"
    print(f"\nQuerying with text: '{query_text}'")
    text_query_results = chroma_manager.query_documents(query_text, n_results=3)
    if text_query_results and text_query_results['documents']:
        for i in range(len(text_query_results['documents'][0])):
            doc_content = text_query_results['documents'][0][i]
            metadata = text_query_results['metadatas'][0][i]
            distance = text_query_results['distances'][0][i]
            print(f"   Result {i+1} (Distance: {distance:.4f}):")
            print(f"     Type: {metadata.get('content_type')}, File: {metadata.get('filename')}, Chunk ID: {metadata.get('chunk_id')}")
            print(f"     Content: '{doc_content}...'") # Print first 100 chars
    else:
        print("No text results found.")

    # Query for an image (using one of the extracted images)
    if processed_results:
        first_file_images_folder = processed_results[0][2]
        image_files_in_folder = list(pathlib.Path(first_file_images_folder).iterdir())
        if image_files_in_folder:
            query_image_path = str(image_files_in_folder[0])
            print(f"\nQuerying with image: '{query_image_path}'")
            image_query_results = chroma_manager.query_images(query_image_path, n_results=3)
            if image_query_results and image_query_results['documents']:
                for i in range(len(image_query_results['documents'][0])):
                    doc_content = image_query_results['documents'][0][i]
                    metadata = image_query_results['metadatas'][0][i]
                    distance = image_query_results['distances'][0][i]
                    print(f"   Result {i+1} (Distance: {distance:.4f}):")
                    print(f"     Type: {metadata.get('content_type')}, File: {metadata.get('filename')}, Original Path: {metadata.get('image_path')}")
                    print(f"     Content: '{doc_content}'")
            else:
                print("No image results found.")
        else:
            print("\nNo images extracted from the first processed file to use for image query example.")
    else:
        print("\nNo files were processed to demonstrate image querying.")


    # Optional: Clean up generated directories
    # shutil.rmtree(output_base_dir, ignore_errors=True)
    # print(f"\nCleaned up '{output_base_dir}' directory.")
    # shutil.rmtree(chroma_db_persist_dir, ignore_errors=True)
    # print(f"Cleaned up '{chroma_db_persist_dir}' directory.")