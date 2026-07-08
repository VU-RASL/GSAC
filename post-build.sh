#!/bin/bash

GSAC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "--- GSAC Post-Build ---"
echo $GSAC_DIR

cd ${GSAC_DIR}/Avatar

# These packages are already installed inside the docker image.
# pip install --no-build-isolation /mounted/home/dresden/repositories/GSAC/Avatar/3rd_party/diff_gaussian_rasterization-alphadep
# pip install git+https://github.com/facebookresearch/pytorch3d.git

cd ${GSAC_DIR}/Preprocessor

if [ ! -d "common/utils" ]; then
    echo "Error: /common/utils directory does not exist."
    exit 1
fi

if [ ! -d "tools/" ]; then
    echo "Downloading TOOLS ."
    wget -O /tmp/human_model_files.tar \
    "https://huggingface.co/RendongZhang/GSAC-Dependencies/resolve/main/human_model_files.tar"
    tar -xf /tmp/human_model_files.tar -C common/utils
    rm /tmp/human_model_files.tar

    # It's tools.zip not .tar
    wget -O /tmp/tools.zip \
        "https://huggingface.co/RendongZhang/GSAC-Dependencies/resolve/main/tools.zip"

    # Unizip for unzpping
    unzip /tmp/tools.zip -d .
    rm /tmp/tools.zip

    wget -O /tmp/sapiens.tar \
        "https://huggingface.co/RendongZhang/GSAC-Dependencies/resolve/main/sapiens.tar"
    tar -xf /tmp/sapiens.tar -C tools/
    rm /tmp/sapiens.tar
else
    echo "* Assets already exist in Preprocessor/tools/  Skipping downloads"
fi


# Get the installation path of torchgeometry
TORCHGEOMETRY_PATH=$(pip show torchgeometry | grep Location | awk '{print $2}')

# Ensure torchgeometry is installed
if [ -z "$TORCHGEOMETRY_PATH" ]; then
    echo "Error: torchgeometry is not installed!"
    exit 1
fi

# Define the target directory
TARGET_DIR="${TORCHGEOMETRY_PATH}/torchgeometry/core/"

# Ensure the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR does not exist!"
    exit 1
fi

echo "* Copying conversions.py"
echo "  From: $(pwd)/conversions.py"
echo "  To:   ${TARGET_DIR}conversions.py"

cp conversions.py "${TARGET_DIR}conversions.py"

echo "--- Done Downloading Tools inside Preprocessor and copying conversions.py inside the environment---"