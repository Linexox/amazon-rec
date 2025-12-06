@echo off
echo ===================================
echo Multi-Modal Embeddings Extraction
echo ===================================
echo.

echo [1/2] Extracting text embeddings with BERT...
python "%~dp0txt_embs_extract.py"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Text embeddings extraction failed!
    pause
    exit /b %ERRORLEVEL%
)
echo Text embeddings extraction completed!
echo.

echo [2/2] Extracting image embeddings with ViT...
python "%~dp0img_embs_extract.py"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Image embeddings extraction failed!
    pause
    exit /b %ERRORLEVEL%
)
echo Image embeddings extraction completed!
echo.

echo ===================================
echo All embeddings extracted successfully!
echo ===================================
pause
