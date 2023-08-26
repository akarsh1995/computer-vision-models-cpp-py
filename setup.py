TORCH_CPU_LIB_LINK = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip"

from subprocess import check_call
import requests
import os
from tqdm import tqdm
from pathlib import Path
import zipfile
from urllib import parse

# URL of the ZIP file to download
zip_url = TORCH_CPU_LIB_LINK


def download_zip_file(url: str, out_dir: Path):
    # Create the destination directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    parsed_url = parse.urlparse(url)
    zip_filename = os.path.basename(parsed_url.path)

    # Create the path to save the downloaded ZIP file
    zip_file_path = Path(out_dir).joinpath(zip_filename)

    # Send an HTTP GET request to download the file
    response = requests.get(url, stream=True)

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Get the total file size from the response headers
        total_size = int(response.headers.get("content-length", 0))

        print(f"Downloading {zip_filename} to {out_dir}")
        # Create a tqdm progress bar
        with tqdm(
            total=total_size, unit="B", unit_scale=True, miniters=1, desc="Downloading"
        ) as progress_bar:
            # Open the destination file for writing in binary mode
            with open(zip_file_path, "wb") as zip_file:
                # Iterate through the response content and write it to the file while updating the progress bar
                for data in response.iter_content(chunk_size=1024):
                    zip_file.write(data)
                    progress_bar.update(len(data))

        print(f"ZIP file downloaded to {zip_file_path}")

        # Count the total number of files in the ZIP archive
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            total_files = len(zip_ref.infolist())

        # Initialize the tqdm progress bar
        with tqdm(
            total=total_files, unit="file", desc="Extracting", ncols=100
        ) as progress_bar:
            # Extract the contents of the ZIP file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    # Update the progress bar for each file
                    progress_bar.update(1)
                    # Extract the file
                    zip_ref.extract(file_info, out_dir)
        print(f"ZIP file extracted to {out_dir}")
    else:
        print("Failed to download the ZIP file.")


def check_cmake():
    import subprocess

    # Run the "cmake --version" command to check for CMake
    try:
        output = subprocess.check_output(
            ["cmake", "--version"], stderr=subprocess.STDOUT, universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        # The "cmake" command returned a non-zero exit code, indicating it's not on the PATH
        print("CMake is not found on the PATH.")
        print("Please install CMake")
        print(e.output)
    except FileNotFoundError:
        # The "cmake" command was not found
        print("CMake is not installed or not on the PATH.")


def make_build_dir():
    Path("build").mkdir(exist_ok=True)


def main():
    download_zip_file(TORCH_CPU_LIB_LINK, Path("lib"))
    make_build_dir()
    check_cmake()


if __name__ == "__main__":
    main()
