import fitz              
import zipfile
import shutil
import pathlib
import re
import threading
from typing import Optional, Tuple

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ------------------------------------------------------------
# (A) IMAGE‐EXTRACTION FUNCTIONS for different file types
# (… your existing extract_images_from_pdf, extract_images_from_docx, etc. …)
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
            img_path = output_folder / img_filename

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
            img_path = output_folder / img_filename

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
            img_path = output_folder / img_filename

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
            return "<!-- no‐more‐images -->"

        img_file = extracted.pop(0)
        rel = img_file.relative_to(base_dir)
        fig_num = img_file.stem.replace("image", "")
        return f"![Figure {fig_num}]({rel})"

    patched = re.sub(r'<!--\s*image\s*-->', replacement_fn, text)
    md_path.write_text(patched, encoding="utf-8")
    print(f"Patched Markdown → {md_path}")


# ------------------------------------------------------------
# (D) UTILITY FUNCTION: PROCESS A SINGLE FILE IN PARALLEL FOR 
#                  IMAGE EXTRACTION + DOCLING CONVERSION
# ------------------------------------------------------------

def process_file(input_path: str, output_base: str
                 ) -> Tuple[str, str]:
    """
    Given any supported file (PDF, DOCX, PPTX, IMAGE, XLSX, CSV), this function:
      1) Extracts embedded images (or copies the standalone image) into
         <output_base>/<stem>_images/
      2) Runs Docling → Markdown (text + tables) into <output_base>/<stem>.md
      3) Patches the generated Markdown so each <!-- image --> becomes
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


# ------------------------------------------------------------
# (E) EXAMPLE: HOW YOU MIGHT CALL process_file(...) FOR EACH INPUT
# ------------------------------------------------------------
if __name__ == "__main__":
    files = [
        r"C:\Users\ameya\OneDrive\Documents\Ameya Sutar Cover Letter JP morgen.pdf"
    ]
    output_base = "converted_md"

    results = []
    for fpath in files:
        md_out, img_out = process_file(fpath, output_base)
        results.append((fpath, md_out, img_out))
        print(f"Processed '{fpath}' → MD: {md_out}, Images: {img_out}")

    print("\n✅ All files processed. Summary:")
    for inp, mdp, imd in results:
        print(f"  • {inp} → Markdown: {mdp}  |  Images folder: {imd}")