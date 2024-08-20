import gc
import concurrent.futures
import streamlit as st
import fitz
from io import BytesIO
import time
from PIL import Image
import pytesseract
import pypdfium2 as pdfium

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


def convert_pdf_to_images(pdf_buffer, scale=300 / 72):
    pdf_file = pdfium.PdfDocument(pdf_buffer)
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


def extract_text_ocr_combo(pdf_buffer):
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_buffer)

    extracted_text = []

    for image_dict in images:
        for page_number, image_bytes in image_dict.items():
            # Convert byte array back to an image
            image = Image.open(BytesIO(image_bytes))

            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(image)
            extracted_text.append(f"Page {page_number + 1}:\n{text}")

    # Return the extracted text as a single string
    return "\n".join(extracted_text)


def extract_text_pymupdf_concurrent(pdf_buffer, batch_size):
    text = ""
    try:
        doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
        total_pages = len(doc)

        # Helper function to process a batch of pages concurrently
        def process_page_batch(start_page, end_page):
            batch_text = ""
            for page_num in range(start_page, end_page):
                page = doc.load_page(page_num)
                batch_text += page.get_text("text")
            return batch_text

        # Create a thread pool executor to handle the concurrent tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                futures.append(executor.submit(process_page_batch, start_page, end_page))
                # st.write(f"Scheduled pages {start_page + 1} to {end_page} for processing")

            # Collect the results as they complete
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            text = "".join(results)

        # Perform garbage collection after processing all batches
        gc.collect()

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    return text


def extract_text_pymupdf_optimized(pdf_buffer, batch_size=50):
    return extract_text_pymupdf_concurrent(pdf_buffer, batch_size)


# Button to trigger extraction
if st.button("Extract"):
    start_time = time.time()
    if uploaded_files:
        for file in uploaded_files:
            pdf_buffer = BytesIO(file.getbuffer())  # Load the file into a BytesIO buffer

            if library == "PyMuPDF":
                result = extract_text_pymupdf_optimized(pdf_buffer)
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
