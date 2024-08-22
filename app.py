import gc
import warnings
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import fitz
from io import BytesIO
import time
from PIL import Image
import pytesseract
import pypdfium2 as pdfium

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
st.title("Text Extraction App")

# Allow the user to select a library for extraction
library = st.selectbox(
    "Select Library",
    ["PyMuPDF", "OCR Combo"],
)

# Allow the user to upload files
uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    st.write(f"Number of uploaded files: {num_files}")


def convert_pdf_to_images(doc_buffer, scale=300 / 72):
    pdf_file = pdfium.PdfDocument(doc_buffer)
    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    list_final_images = []

    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format="jpeg", optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append({i: image_byte_array})

    return list_final_images


def process_page_batch_tesseract(image_batch):
    batch_text = []
    for image_dict in image_batch:
        for page_number, image_bytes in image_dict.items():
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            batch_text.append(text)

            image.close()

        gc.collect()

    return batch_text


def extract_text_ocr_combo(doc_buffer, batch_size=50):
    # Convert PDF to images
    images = convert_pdf_to_images(doc_buffer)
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

    extracted_text = []

    with ThreadPoolExecutor() as executor:
        future_to_batch = {
            executor.submit(process_page_batch_tesseract, batch): batch
            for batch in batches
        }

        # Collect the results as they complete
        for future in future_to_batch:
            try:
                extracted_text.extend(future.result())
            except Exception as e:
                extracted_text.append(f"Error processing batch: {str(e)}")

    gc.collect()
    return "\n".join(extracted_text)


def process_page_batch_pymupdf(document, page_numbers):
    extracted_text = ""
    for page_number in page_numbers:
        page = document.load_page(page_number)
        extracted_text += page.get_text("text")
        del page

    gc.collect()
    return extracted_text


def extract_text_pymupdf(doc_buffer, batch_size=50):
    document = fitz.open(stream=doc_buffer, filetype="pdf")
    total_pages = document.page_count

    batches = [range(i, min(i + batch_size, total_pages)) for i in range(0, total_pages, batch_size)]

    extracted_text = []

    with ThreadPoolExecutor() as executor:
        future_to_batch = {
            executor.submit(process_page_batch_pymupdf, document, batch): batch
            for batch in batches
        }

        for future in future_to_batch:
            try:
                extracted_text.append(future.result())
            except Exception as e:
                extracted_text.append(f"Error processing batch: {str(e)}")

    document.close()
    gc.collect()

    return "\n".join(extracted_text)


if st.button("Extract"):
    start_time = time.time()
    if uploaded_files:
        for file in uploaded_files:
            pdf_buffer = BytesIO(file.getbuffer())  # Load the file into a BytesIO buffer

            if library == "PyMuPDF":
                result = extract_text_pymupdf(pdf_buffer)
                st.write("PyMuPDF Extracted Text:")
                st.write(result)

            elif library == "OCR Combo":
                result = extract_text_ocr_combo(pdf_buffer)
                st.write("OCR Combo Extracted Text:")
                st.write(result)

        end_time = time.time()
        st.text(f"Total processing time: {end_time - start_time:.2f} seconds")

    else:
        st.error("Please upload files first")
