import streamlit as st
import pandas as pd
import requests
import os
import gzip
import tarfile
from pathlib import Path
import subprocess
import time
from urllib.parse import urljoin, unquote
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import logging
import ftplib
import re
from bs4 import BeautifulSoup


class GEODownloader:
    """Automated GEO dataset downloader with progress tracking"""

    def __init__(self, base_dir: str = "data/raw/single_cell_rnaseq"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # GEO FTP base URLs
        self.geo_ftp_base = "https://ftp.ncbi.nlm.nih.gov/geo/series"
        self.sra_base = "https://www.ncbi.nlm.nih.gov/sra"

        # Setup logging - suppress Streamlit warnings but keep our logs
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Suppress specific Streamlit warnings
        logging.getLogger('streamlit').setLevel(logging.ERROR)
        logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
        logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

        self.logger = logging.getLogger(__name__)

    def search_geo_datasets(self, search_term: str, max_results: int = 20) -> List[Dict]:
        """Search GEO for datasets matching a term like 'Perturb-Seq'"""
        try:
            # Search GDS records (this matches the browser search)
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                'db': 'gds',
                'term': search_term,  # Simplified - no [All Fields] wrapper
                'retmax': max_results,
                'retmode': 'xml'
            }

            self.logger.info(f"Searching with URL: {esearch_url}")
            self.logger.info(f"Search params: {search_params}")

            response = requests.get(esearch_url, params=search_params, timeout=30)
            self.logger.info(f"Response status: {response.status_code}")
            self.logger.info(f"Response URL: {response.url}")

            if response.status_code != 200:
                self.logger.error(f"Search failed: HTTP {response.status_code}")
                self.logger.error(f"Response content: {response.text[:500]}")
                return []

            # Parse search results
            self.logger.info(f"Response content (first 1000 chars): {response.text[:1000]}")
            root = ET.fromstring(response.content)

            # Check the full XML structure
            count_elem = root.find('.//Count')
            if count_elem is not None:
                self.logger.info(f"Total count found: {count_elem.text}")

            id_list = root.find('.//IdList')

            if id_list is None:
                self.logger.warning("No IdList found in response")
                return []

            ids = [id_elem.text for id_elem in id_list]
            self.logger.info(f"Found {len(ids)} IDs: {ids}")

            if len(ids) == 0:
                self.logger.info(f"No results found for '{search_term}'")
                return []

            # Fetch summaries for all results
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {
                'db': 'gds',
                'id': ','.join(ids),
                'retmode': 'xml'
            }

            self.logger.info(f"Fetching summaries for IDs: {','.join(ids)}")
            summary_response = requests.get(esummary_url, params=summary_params, timeout=30)

            if summary_response.status_code != 200:
                self.logger.error(f"Summary fetch failed: HTTP {summary_response.status_code}")
                return []

            self.logger.info(f"Summary response (first 1000 chars): {summary_response.text[:1000]}")

            # Parse summaries
            summary_root = ET.fromstring(summary_response.content)
            results = []

            # Find all DocSum elements (not DocumentSummary)
            doc_sums = summary_root.findall('.//DocSum')
            self.logger.info(f"Found {len(doc_sums)} DocSum elements")

            for i, doc_sum in enumerate(doc_sums):
                try:
                    # Extract information - get all Item elements
                    items = doc_sum.findall('.//Item')
                    available_fields = {item.get('Name'): item.text for item in items if item.get('Name')}

                    # Log available fields for first result only (to avoid spam)
                    if i == 0:  # Fixed: use index instead of len(results)
                        self.logger.info(f"Available fields in first result: {list(available_fields.keys())}")

                    # Extract the fields we can find
                    accession = available_fields.get('Accession', '')
                    gse = available_fields.get('GSE', '')
                    title = available_fields.get('title', 'No title')
                    summary_text = available_fields.get('summary', 'No summary')

                    # For organism, date, samples
                    organism = available_fields.get('taxon', 'Unknown')
                    pub_date = available_fields.get('PDAT', 'Unknown')
                    n_samples = available_fields.get('n_samples', 'Unknown')

                    # Construct the full GSE accession
                    final_accession = None
                    if gse and gse.strip():
                        # GSE field contains just the number, add 'GSE' prefix
                        final_accession = f"GSE{gse}"
                    elif accession and accession.startswith('GSE'):
                        # Accession field already has GSE prefix
                        final_accession = accession

                    # Debug logging for first few records
                    if i < 3:
                        self.logger.info(
                            f"Record {i + 1}: Accession='{accession}', GSE='{gse}', final='{final_accession}'")

                    # Accept if we have a valid GSE accession
                    if final_accession and final_accession.startswith('GSE'):
                        result = {
                            'accession': final_accession,
                            'title': title if title else "No title",
                            'summary': summary_text if summary_text else "No summary",
                            'organism': organism,
                            'date': pub_date,
                            'n_samples': n_samples,
                            'search_term': search_term
                        }
                        results.append(result)

                        if len(results) <= 3:  # Log first few results
                            self.logger.info(f"Added result {len(results)}: {final_accession} - {title[:50]}...")

                except Exception as e:
                    self.logger.error(f"Error parsing DocSum {i}: {e}")
                    continue

            self.logger.info(f"Final results: Found {len(results)} GSE datasets for '{search_term}'")
            return results

        except Exception as e:
            self.logger.error(f"Error searching for '{search_term}': {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def get_geo_info(self, geo_accession: str) -> Dict:
        """Fetch GEO dataset metadata"""
        try:
            # Query GEO for dataset info
            esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'gds',
                'term': geo_accession,
                'retmode': 'xml'
            }

            response = requests.get(esearch_url, params=params)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                id_list = root.find('.//IdList')
                if id_list is not None and len(id_list) > 0:
                    geo_id = id_list[0].text

                    # Get detailed info
                    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        'db': 'gds',
                        'id': geo_id,
                        'retmode': 'xml'
                    }

                    summary_response = requests.get(esummary_url, params=summary_params)
                    summary_root = ET.fromstring(summary_response.content)

                    title = summary_root.find('.//Item[@Name="title"]')
                    summary = summary_root.find('.//Item[@Name="summary"]')

                    return {
                        'accession': geo_accession,
                        'title': title.text if title is not None else "Unknown",
                        'summary': summary.text if summary is not None else "No summary available",
                        'status': 'Found'
                    }

            return {
                'accession': geo_accession,
                'title': 'Not found',
                'summary': 'Dataset not found in GEO',
                'status': 'Not Found'
            }

        except Exception as e:
            self.logger.error(f"Error fetching GEO info for {geo_accession}: {e}")
            return {
                'accession': geo_accession,
                'title': 'Error',
                'summary': str(e),
                'status': 'Error'
            }

    def get_geo_file_list(self, geo_accession: str) -> List[Dict]:
        """Get actual list of available files for a GEO dataset by scanning FTP directory"""
        try:
            # Parse series number (e.g., GSE264667 -> GSE264nnn, GSE303901 -> GSE303nnn)
            series_num = geo_accession[3:6] + "nnn"  # GSE264nnn

            # Try FTP first (more reliable)
            files_info = self._scan_ftp_directory(geo_accession, series_num)

            # If FTP fails, try HTTP directory listing
            if not files_info:
                files_info = self._scan_http_directory(geo_accession, series_num)

            return files_info

        except Exception as e:
            self.logger.error(f"Error getting file list for {geo_accession}: {e}")
            return []

    def _scan_ftp_directory(self, geo_accession: str, series_num: str) -> List[Dict]:
        """Scan FTP directory for actual files"""
        files_info = []
        try:
            ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
            ftp.login()

            ftp_path = f"/geo/series/GSE{series_num}/{geo_accession}/suppl/"

            # List directory contents
            file_list = []
            try:
                ftp.retrlines(f'LIST {ftp_path}', file_list.append)
            except ftplib.error_perm as e:
                if "550" in str(e):  # Directory not found
                    self.logger.warning(f"Directory not found: {ftp_path}")
                    ftp.quit()
                    return []
                else:
                    raise

            ftp.quit()

            # Parse FTP listing
            for line in file_list:
                parts = line.split()
                if len(parts) >= 9 and not line.startswith('d'):  # Not a directory
                    filename = ' '.join(parts[8:])  # Handle filenames with spaces
                    try:
                        file_size = int(parts[4])
                    except (ValueError, IndexError):
                        file_size = 0

                    # Construct download URL (HTTP, not FTP for easier downloading)
                    file_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{series_num}/{geo_accession}/suppl/{filename}"

                    files_info.append({
                        'filename': filename,
                        'url': file_url,
                        'type': self._classify_file_type(filename),
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2) if file_size > 0 else 0
                    })

            self.logger.info(f"Found {len(files_info)} files via FTP for {geo_accession}")

        except Exception as e:
            self.logger.warning(f"FTP scan failed for {geo_accession}: {e}")

        return files_info

    def _scan_http_directory(self, geo_accession: str, series_num: str) -> List[Dict]:
        """Fallback: scan HTTP directory listing"""
        files_info = []
        try:
            http_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{series_num}/{geo_accession}/suppl/"

            response = requests.get(http_url, timeout=30)
            if response.status_code == 200:
                # Parse HTML directory listing
                soup = BeautifulSoup(response.content, 'html.parser')

                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and not href.startswith('..') and not href.endswith('/'):
                        # Decode URL-encoded filename
                        filename = unquote(href)

                        # Skip parent directory links
                        if filename in ['..', '.']:
                            continue

                        file_url = urljoin(http_url, href)

                        # Try to get file size from the page text (this is often unreliable)
                        file_size = 0
                        size_mb = 0

                        # Look for size information in nearby text
                        parent = link.parent
                        if parent:
                            size_text = parent.get_text()
                            # Look for patterns like "123M" or "456K" or "789 bytes"
                            size_match = re.search(r'(\d+\.?\d*)\s*([KMG]?B?)\b', size_text)
                            if size_match:
                                size_value = float(size_match.group(1))
                                size_unit = size_match.group(2).upper()

                                if 'K' in size_unit:
                                    file_size = int(size_value * 1024)
                                elif 'M' in size_unit:
                                    file_size = int(size_value * 1024 * 1024)
                                elif 'G' in size_unit:
                                    file_size = int(size_value * 1024 * 1024 * 1024)
                                else:
                                    file_size = int(size_value)

                                size_mb = round(file_size / (1024 * 1024), 2)

                        files_info.append({
                            'filename': filename,
                            'url': file_url,
                            'type': self._classify_file_type(filename),
                            'size_bytes': file_size,
                            'size_mb': size_mb if size_mb > 0 else 0
                        })

                self.logger.info(f"Found {len(files_info)} files via HTTP for {geo_accession}")

        except Exception as e:
            self.logger.warning(f"HTTP scan failed for {geo_accession}: {e}")

        return files_info

    def _classify_file_type(self, filename: str) -> str:
        """Classify file type based on filename"""
        filename_lower = filename.lower()

        # More specific classification based on real GEO file patterns
        if 'filtered_feature_bc_matrix' in filename_lower:
            return '10X Filtered Matrix'
        elif 'raw_feature_bc_matrix' in filename_lower:
            return '10X Raw Matrix'
        elif 'matrix.mtx' in filename_lower:
            return 'Expression Matrix (MTX)'
        elif 'barcodes.tsv' in filename_lower:
            return 'Cell Barcodes'
        elif 'features.tsv' in filename_lower or 'genes.tsv' in filename_lower:
            return 'Gene Features'
        elif filename_lower.endswith('.h5ad'):
            return 'AnnData (H5AD)'
        elif filename_lower.endswith('.h5'):
            return 'HDF5 Matrix'
        elif 'metadata' in filename_lower or 'sample' in filename_lower:
            return 'Metadata'
        elif filename_lower.endswith('.txt.gz') or filename_lower.endswith('.csv.gz'):
            return 'Compressed Text/CSV'
        elif filename_lower.endswith('.tar.gz'):
            return 'Archive (TAR.GZ)'
        elif filename_lower.endswith('.gz'):
            return 'Compressed File'
        else:
            return 'Other'

    def download_file(self, url: str, local_path: Path, progress_callback=None) -> bool:
        """Download a file with progress tracking"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))

                with open(local_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if progress_callback and total_size > 0:
                                progress = downloaded / total_size
                                progress_callback(progress, downloaded, total_size)

                return True
            else:
                self.logger.error(f"Failed to download {url}: HTTP {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract tar.gz or gz files"""
        try:
            if archive_path.suffix == '.gz':
                if archive_path.stem.endswith('.tar'):
                    # tar.gz file
                    with tarfile.open(archive_path, 'r:gz') as tar:
                        tar.extractall(extract_to)
                else:
                    # Regular gz file
                    with gzip.open(archive_path, 'rb') as gz_file:
                        with open(extract_to / archive_path.stem, 'wb') as out_file:
                            out_file.write(gz_file.read())

            return True

        except Exception as e:
            self.logger.error(f"Error extracting {archive_path}: {e}")
            return False


# Streamlit UI
def main():
    st.set_page_config(page_title="GEO Dataset Downloader", layout="wide")

    st.title("🧬 GEO Dataset Download Automation")
    st.markdown("Streamlined downloading and processing of Gene Expression Omnibus datasets")

    # Initialize downloader
    if 'downloader' not in st.session_state:
        st.session_state.downloader = GEODownloader()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        base_dir = st.text_input("Base Directory", value="data/raw/single_cell_rnaseq")
        if st.button("Update Directory"):
            st.session_state.downloader = GEODownloader(base_dir)
            st.success("Directory updated!")

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Dataset Search")

        # New: Keyword Search
        st.subheader("🔍 Search by Keywords")
        search_terms = [
            "",
            "Perturb-Seq",
            "CRISPR screen",
            "single cell perturbation",
            "guide RNA",
            "CRISPRi",
            "CRISPRa",
            "perturbation screen"
        ]

        col1a, col1b = st.columns([2, 1])
        with col1a:
            selected_search = st.selectbox("Common search terms:", search_terms)
            custom_search = st.text_input("Or enter custom search term:")

        with col1b:
            max_results = st.number_input("Max results:", min_value=5, max_value=50, value=20)

        search_term = custom_search or selected_search

        if search_term and st.button("🔍 Search GEO"):
            with st.spinner(f"Searching GEO for '{search_term}'..."):
                search_results = st.session_state.downloader.search_geo_datasets(search_term, max_results)
                st.session_state.search_results = search_results
                if search_results:
                    st.success(f"Found {len(search_results)} datasets!")
                else:
                    st.warning("No datasets found. Try different keywords.")

        # Display search results
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.subheader("Search Results")

            # Create a compact display of results
            results_data = []
            for i, result in enumerate(st.session_state.search_results):
                results_data.append({
                    'GSE': result['accession'],
                    'Title': result['title'][:60] + "..." if len(result['title']) > 60 else result['title'],
                    'Organism': result['organism'],
                    'Samples': result['n_samples'],
                    'Date': result['date']
                })

            results_df = pd.DataFrame(results_data)

            # Let user select from search results
            selected_idx = st.selectbox(
                "Select a dataset:",
                options=range(len(st.session_state.search_results)),
                format_func=lambda
                    x: f"{st.session_state.search_results[x]['accession']} - {st.session_state.search_results[x]['title'][:50]}..."
            )

            if st.button("📥 Load Selected Dataset"):
                selected_result = st.session_state.search_results[selected_idx]
                st.session_state.current_dataset = {
                    'accession': selected_result['accession'],
                    'title': selected_result['title'],
                    'summary': selected_result['summary'],
                    'status': 'Found'
                }
                st.success(f"Loaded {selected_result['accession']}")

        st.divider()

        # Quick access to your known datasets
        st.subheader("Your Project Datasets")
        known_datasets = [
            "GSE264667",  # HepG2 and Jurkat
            "GSE274751",  # TF knockout
            "GSE303901",  # PerturbSeq DC-TAP
        ]

        selected_known = st.selectbox("Select known dataset:", [""] + known_datasets)

        st.subheader("Direct GSE Lookup")
        custom_geo = st.text_input("Enter GEO accession (e.g., GSE264667):")

        # Use selected or custom input
        geo_accession = selected_known or custom_geo

        if geo_accession and st.button("Search Dataset"):
            with st.spinner("Fetching dataset information..."):
                dataset_info = st.session_state.downloader.get_geo_info(geo_accession)
                st.session_state.current_dataset = dataset_info

    with col2:
        st.header("Dataset Information")

        if hasattr(st.session_state, 'current_dataset'):
            dataset = st.session_state.current_dataset

            # Display dataset info
            st.subheader(f"📊 {dataset['accession']}")
            st.write(f"**Title:** {dataset['title']}")
            st.write(f"**Status:** {dataset['status']}")

            with st.expander("Summary"):
                st.write(dataset['summary'])

            if dataset['status'] == 'Found':
                st.subheader("Available Files")

                if st.button("Scan for Files"):
                    with st.spinner("Scanning for available files..."):
                        files = st.session_state.downloader.get_geo_file_list(dataset['accession'])
                        st.session_state.available_files = files

                if hasattr(st.session_state, 'available_files') and st.session_state.available_files:
                    files_df = pd.DataFrame(st.session_state.available_files)

                    # Display files in a nice table - handle missing columns gracefully
                    display_columns = ['filename', 'type']
                    if 'size_mb' in files_df.columns:
                        display_columns.append('size_mb')
                        display_df = files_df[display_columns].copy()
                        # Format size column
                        display_df['size_mb'] = display_df['size_mb'].apply(
                            lambda x: f"{x} MB" if isinstance(x, (int, float)) and x > 0 else "Unknown"
                        )
                    else:
                        display_df = files_df[display_columns].copy()

                    st.dataframe(display_df, use_container_width=True)

                    # File selection with proper formatting
                    def format_file_option(x):
                        file_info = files_df.iloc[x]
                        base_text = f"{file_info['filename']} ({file_info['type']})"

                        # Add size info if available
                        if 'size_mb' in file_info:
                            size_mb = file_info['size_mb']
                            if isinstance(size_mb, (int, float)) and size_mb > 0:
                                return f"{base_text} - {size_mb} MB"

                        return base_text

                    selected_files = st.multiselect(
                        "Select files to download:",
                        options=range(len(files_df)),
                        format_func=format_file_option
                    )

                    if selected_files and st.button("Download Selected Files"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, file_idx in enumerate(selected_files):
                            file_info = files_df.iloc[file_idx]
                            filename = file_info['filename']
                            url = file_info['url']

                            # Create local path
                            dataset_dir = Path(base_dir) / dataset['accession']
                            dataset_dir.mkdir(parents=True, exist_ok=True)
                            local_path = dataset_dir / filename

                            status_text.text(f"Downloading {filename}...")

                            def update_progress(progress, downloaded, total):
                                overall_progress = (i + progress) / len(selected_files)
                                progress_bar.progress(overall_progress)
                                size_mb = downloaded / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                status_text.text(
                                    f"Downloading {filename}: {size_mb:.1f}/{total_mb:.1f} MB ({overall_progress * 100:.1f}% overall)")

                            success = st.session_state.downloader.download_file(
                                url, local_path, update_progress
                            )

                            if success:
                                st.success(f"✅ Downloaded {filename}")

                                # Auto-extract if it's an archive
                                if filename.endswith('.tar.gz') or filename.endswith('.gz'):
                                    extract_checkbox_key = f"extract_{filename}_{i}"
                                    if st.checkbox(f"Extract {filename}?", value=True, key=extract_checkbox_key):
                                        extract_dir = dataset_dir / "extracted"
                                        extract_dir.mkdir(exist_ok=True)

                                        if st.session_state.downloader.extract_archive(local_path, extract_dir):
                                            st.success(f"✅ Extracted {filename}")
                                        else:
                                            st.error(f"❌ Failed to extract {filename}")
                            else:
                                st.error(f"❌ Failed to download {filename}")

                        progress_bar.progress(1.0)
                        status_text.text("Download complete!")

                elif hasattr(st.session_state, 'available_files'):
                    st.warning(
                        "No files found for this dataset. The dataset may not have supplementary files or may use a different structure.")

    # Dataset management section
    st.header("📁 Local Dataset Management")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Downloaded Datasets")
        if os.path.exists(base_dir):
            datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if datasets:
                for dataset in datasets:
                    dataset_path = os.path.join(base_dir, dataset)
                    files = os.listdir(dataset_path)
                    file_count = len(files)
                    total_size = sum(os.path.getsize(os.path.join(dataset_path, f))
                                     for f in files if os.path.isfile(os.path.join(dataset_path, f)))
                    size_mb = total_size / (1024 * 1024)

                    st.write(f"📦 **{dataset}** - {file_count} files ({size_mb:.1f} MB)")
            else:
                st.info("No datasets downloaded yet")

    with col4:
        st.subheader("Quick Actions")

        # Preprocessing pipeline integration
        if st.button("🔄 Run Preprocessing Pipeline"):
            st.info("This would integrate with your existing preprocessing scripts")
            st.code("""
            # Example integration:
            python data/scrna_normalization.py --dataset {selected_dataset}
            python standardized_normalizer.py --input {dataset_path}
            """)

        if st.button("📊 Generate Dataset Report"):
            st.info("This would create a summary report of all downloaded datasets")

        if st.button("🧹 Cleanup Temporary Files"):
            st.info("This would clean up extracted archives and temporary files")


if __name__ == "__main__":
    main()