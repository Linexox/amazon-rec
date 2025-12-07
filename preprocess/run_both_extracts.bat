@echo off
echo Starting embeddings extraction...
echo.

echo [1/2] Extracting text embeddings...
python d:\.Workspace\数据分析大作业\preprocess\txt_embs_extract.py
if %errorlevel% neq 0 (
    echo Error: Text embeddings extraction failed!
    exit /b %errorlevel%
)
echo Text embeddings extraction completed successfully!
echo.

echo [2/2] Extracting image embeddings...
python d:\.Workspace\数据分析大作业\preprocess\img_embs_extract.py
if %errorlevel% neq 0 (
    echo Error: Image embeddings extraction failed!
    exit /b %errorlevel%
)
echo Image embeddings extraction completed successfully!
echo.

echo All embeddings extracted successfully!
