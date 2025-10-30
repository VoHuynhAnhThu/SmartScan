#!/bin/bash
# Download Consumer-to-shop Clothes Retrieval Benchmark dataset (Evaluation + Images)
# Improved: handles large files, checks permission/HTML responses, supports resume, optional gdown fallback
# Usage: bash get_consumer_to_shop_Clothes_Retrieval_Benchmark.sh

set -eu

d='datasets'

file_eval_id='1evyngAeBekkP4MpPPsQKCmx_nnNLBQxr'
file_img_highres_id='1w1V8ZBUo3o9WYugaCQMI6xYi73--DobJ'
file_img_id='1fAKz7944axwrlt-yTZmiF0etwky-V60R'

f_eval='consumer_to_shop_Clothes_Retrieval_Benchmark.zip'
f_img_highres='img_highres.zip'
f_img='img.zip'

_extract_confirm() {
    local html_file="$1"
    local confirm
    if grep -oP '(?<=name="confirm" value=")[^"]+' "$html_file" >/dev/null 2>&1; then
        confirm=$(grep -oP '(?<=name="confirm" value=")[^"]+' "$html_file" | head -n 1)
    else
        # sed fallback
        confirm=$(sed -n 's/.*name="confirm" value="\([^"]*\)".*/\1/p' "$html_file" | head -n 1 || true)
    fi
    printf '%s' "$confirm"
}

_extract_action() {
    local html_file="$1"
    local action
    if grep -oP '(?<=action=")https://[^"]+' "$html_file" >/dev/null 2>&1; then
        action=$(grep -oP '(?<=action=")https://[^"]+' "$html_file" | head -n 1)
    else
        action=$(sed -n 's/.*action="\([^"]*https:[^"]*\)".*/\1/p' "$html_file" | head -n 1 || true)
    fi
    printf '%s' "$action"
}

_check_access_errors() {
    local html_file="$1"
    if grep -q -i "You need access" "$html_file" 2>/dev/null || grep -q -i "sign in" "$html_file" 2>/dev/null || grep -q -i "login" "$html_file" 2>/dev/null || grep -q -i "Sign in" "$html_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

download_from_gdrive() {
    local file_id=$1
    local filename=$2
    local base_url="https://drive.google.com/uc?export=download&id=${file_id}"

    echo
    echo "------------------------------------------------------------"
    echo "Downloading ${filename} from Google Drive (id=${file_id})..."
    echo "Temp files: cookies.txt / /tmp/gdrive_tmp.html"
    echo "If this fails with 'need access', make sure the file is shared (Anyone with the link)."
    echo "------------------------------------------------------------"

    if command -v gdown >/dev/null 2>&1; then
        echo "gdown found — trying gdown as first attempt (handles confirm tokens automatically)..."
        if gdown --id "$file_id" -O "$filename" -c; then
            echo "gdown succeeded for $filename"
        else
            echo "gdown failed; falling back to curl method..."
        fi
    fi

    if [ -f "$filename" ]; then
        if file "$filename" | grep -q -i "zip archive"; then
            echo "$filename already exists and looks like a zip — skipping download."
            return 0
        else
            echo "$filename exists but is not a zip — will attempt re-download."
            rm -f "$filename"
        fi
    fi

    curl -c cookies.txt -s -L "$base_url" -o /tmp/gdrive_tmp.html

    if _check_access_errors /tmp/gdrive_tmp.html; then
        echo "Access problem detected for file id=${file_id}."
        echo "Hint: Open https://drive.google.com/file/d/${file_id}/view in a browser."
        echo "If you see 'You need access', set the file to 'Anyone with the link can view' or provide an accessible link."
        echo "First lines of HTML response (for debugging):"
        sed -n '1,20p' /tmp/gdrive_tmp.html
        rm -f cookies.txt /tmp/gdrive_tmp.html
        exit 2
    fi

    confirm=$(_extract_confirm /tmp/gdrive_tmp.html || true)
    real_url=$(_extract_action /tmp/gdrive_tmp.html || true)

    if [ -n "$real_url" ]; then
        echo "Found action URL: $real_url"
    fi
    if [ -n "$confirm" ]; then
        echo "Found confirm token: $confirm"
    fi

    if [ -z "$real_url" ]; then
        real_url="https://drive.google.com/uc?export=download"
    fi

    echo "Starting curl download from: ${real_url}?confirm=${confirm}&id=${file_id}"
    curl -Lb cookies.txt --fail --show-error --location "${real_url}?confirm=${confirm}&id=${file_id}" -o "$filename" --continue-at - --progress-bar || {
        echo "curl download failed for ${filename}. Saving debug HTML for inspection..."

        file_type=$(file "$filename" 2>/dev/null || true)
        echo "Downloaded file type: ${file_type:-unknown}"
        echo "First 10 lines of downloaded file:"
        head -n 10 "$filename" || true
        rm -f cookies.txt /tmp/gdrive_tmp.html
        exit 3
    }

    rm -f cookies.txt /tmp/gdrive_tmp.html

    file_type=$(file "$filename" || true)
    echo "File type: $file_type"
    if echo "$file_type" | grep -qiE 'html|text'; then
        echo "Error: Downloaded file is not a ZIP file (probably HTML page)."
        echo "First few lines of the downloaded file for debugging:"
        head -n 20 "$filename" || true
        rm -f "$filename"
        exit 4
    fi

    echo "Download finished and looks like a binary archive."
}

mkdir -p "$d"

download_from_gdrive "$file_eval_id" "$f_eval"
download_from_gdrive "$file_img_highres_id" "$f_img_highres"
download_from_gdrive "$file_img_id" "$f_img"

echo
echo "Extracting datasets to $d..."
unzip -q "$f_eval" -d "$d" && rm -f "$f_eval"
unzip -q "$f_img_highres" -d "$d" && rm -f "$f_img_highres"
unzip -q "$f_img" -d "$d" && rm -f "$f_img"

echo
echo "Download and extraction completed successfully!"
