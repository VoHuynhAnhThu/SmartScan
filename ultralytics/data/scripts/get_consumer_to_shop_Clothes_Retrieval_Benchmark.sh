#!/bin/bash
# Download Consumer-to-shop Clothes Retrieval Benchmark dataset from Google Drive
# Example usage: bash data/scripts/get_consumer_to_shop_Clothes_Retrieval_Benchmark.sh

d='../datasets' # unzip directory
file_id='1evyngAeBekkP4MpPPsQKCmx_nnNLBQxr'
url="https://drive.google.com/uc?export=download&id=${file_id}"
f='consumer_to_shop_Clothes_Retrieval_Benchmark.zip'

echo "Downloading ${f} from Google Drive..."

# Download from Google Drive with proper error handling
curl -L "$url" -o "$f" -# || {
  echo "Direct download failed, trying with confirmation..."
  curl -c cookies.txt -s -L "$url" > /dev/null
  confirm_url=$(curl -s -b cookies.txt -L "$url" | grep -o 'confirm=[^&]*' | head -1)
  if [ -n "$confirm_url" ]; then
    curl -b cookies.txt -L "${url}&${confirm_url}" -o "$f" -#
  else
    echo "Could not find confirmation token. Download may have failed."
  fi
  rm -f cookies.txt
}

# Check if download was successful
if [ ! -f "$f" ]; then
  echo "Error: Download failed - file not found"
  exit 1
fi

# Check if file is actually a zip file
file_type=$(file "$f")
echo "File type: $file_type"

if [[ "$file_type" == *"HTML"* ]] || [[ "$file_type" == *"text"* ]]; then
  echo "Error: Downloaded file is not a ZIP file (probably HTML error page)"
  echo "First few lines of downloaded file:"
  head -5 "$f"
  rm -f "$f"
  exit 1
fi

# Create directory if it doesn't exist
mkdir -p "$d"

# Unzip and cleanup
echo "Extracting to $d..."
unzip -q "$f" -d "$d" && rm "$f" || {
  echo "Error: Failed to extract $f"
  echo "File size: $(ls -lh "$f" | awk '{print $5}')"
  exit 1
}

echo "Download and extraction completed successfully!"
wait
