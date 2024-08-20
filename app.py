import gc
import concurrent.futures
import os
import tempfile
import time
import streamlit as st
import fitz

st.title("Text Extraction App")

# Allow the user to select a library for extraction
library = st.selectbox(
    "Select Library",
    ["PyMuPDF"],
)

# Allow the user to upload files
uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    st.write(f"Number of uploaded files: {num_files}")


# Optimized PyMuPDF extraction function with concurrency and garbage collection
def extract_text_pymupdf_concurrent(file_path, batch_size=50):
    text = ""
    try:
        with fitz.open(file_path) as doc:
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
                    st.write(f"Scheduled pages {start_page + 1} to {end_page} for processing")

                # Collect the results as they complete
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                text = "".join(results)

        # Perform garbage collection after processing all batches
        gc.collect()

    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return None

    return text

def extract_text_pymupdf_optimized(file_path, batch_size=30):
    return extract_text_pymupdf_concurrent(file_path, batch_size)

# def extract_text_pymupdf_optimized(file_path):
#     with fitz.open(file_path) as doc:
#         text = ""
#         for page in doc:
#             text += page.get_text("text")
#
#     gc.collect()
#     return text
#

start_time = time.time()
# Button to trigger extraction
if st.button("Extract"):
    start_time = time.time()
    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for file in uploaded_files:
                file_path = os.path.join(tmp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                if library == "PyMuPDF":
                    result = extract_text_pymupdf_optimized(file_path)
                    st.write("PyMuPDF Extracted Text:")
                    st.write(result)
    else:
        st.error("Please upload files first")

end_time = time.time()
time_difference = end_time - start_time
st.write(f"Time taken: {time_difference}")