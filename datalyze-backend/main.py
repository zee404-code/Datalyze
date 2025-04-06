from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import chardet
import io
import asyncio
import json
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

from dataprocessing import *  # custom functions
# NOTE: `select_persona`, `persona`, `detect_data_domain`, etc. must be defined in dataprocessing.py

# Initialize app
app = FastAPI()

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp memory store for uploaded files
uploaded_files = {}

# ========== Models ==========
class RoleInput(BaseModel):
    role: str

# ========== Routes ==========
@app.post("/submit-role")
async def submit_role(data: RoleInput):
    select_persona(data.role)
    return {"message": f"Received role: {data.role}"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        uploaded_files[file.filename] = content
        return {"message": "✅ File uploaded successfully", "filename": file.filename}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get-insights-progress/{filename}")
async def get_insights_progress(filename: str, request: Request):
    async def event_generator():
        try:
            content = uploaded_files.get(filename)
            if not content:
                yield f"data: {json.dumps({'error': '❌ File not found'})}\n\n"
                return

            # Step 0: Load file
            if filename.lower().endswith(".csv"):
                encoding = chardet.detect(content[:10000])['encoding']
                df = pd.read_csv(io.StringIO(content.decode(encoding)), low_memory=False)
            elif filename.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            else:
                yield f"data: {json.dumps({'error': '❌ Unsupported file format'})}\n\n"
                return

            df = df.dropna(axis=1, how='all')
            explanation_log = []
            stats_log = []

            # Step 1
            yield f"data: {json.dumps({'step': '🔍 Detecting data domain...', 'progress': 10})}\n\n"
            insights = detect_data_domain(df)
            await asyncio.sleep(0.5)

            # Step 2
            yield f"data: {json.dumps({'step': '🧼 Handling missing values...', 'progress': 30})}\n\n"
            df, _, _ = handle_missing_values_DA_BO(df, explanation_log, stats_log)
            await asyncio.sleep(0.5)

            # Step 3
            yield f"data: {json.dumps({'step': '📉 Dropping extreme outliers...', 'progress': 50})}\n\n"
            df_cleaned = drop_extreme_outliers(df, explanation_log, stats_log)
            await asyncio.sleep(0.5)

            # Step 4
            yield f"data: {json.dumps({'step': '📊 Performing EDA & stat tests...', 'progress': 70})}\n\n"
            df, _, _ = perform_eda_and_stat_tests(df_cleaned, explanation_log, stats_log)
            await asyncio.sleep(0.5)

            # Step 5
            yield f"data: {json.dumps({'step': '📝 Generating report...', 'progress': 85})}\n\n"
            if persona == 'DA':
                report = generate_bi_report(df, stats_log, insights)
                report_text = f"{insights}\n\n{report}\n\n" + "\n".join(explanation_log)
            else:
                report = generate_bi_report(df, explanation_log, insights)
                report_text = f"{insights}\n\n{report}\n\n" + "\n".join(explanation_log)
            await asyncio.sleep(0.5)

            # Step 6
            yield f"data: {json.dumps({'step': '💾 Saving report to PDF...', 'progress': 95})}\n\n"
            unique_path = f"/tmp/BI_Report_{uuid.uuid4().hex}.pdf"
            save_report_to_pdf(report_text, filename=unique_path)
            await asyncio.sleep(0.5)

            filename_only = os.path.basename(unique_path)
            yield f"data: {json.dumps({'step': '✅ Done!', 'progress': 100, 'filename': filename_only})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': f'❌ Error: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download-report/{filename}")
async def download_report(filename: str):
    # Check if the file exists on the server
    filepath = f"/tmp/{filename}"
    if not os.path.exists(filepath):
        return JSONResponse(content={"error": "❌ Report file not found"}, status_code=404)

    # Serve the file to the client for download
    return FileResponse(path=filepath, filename=filename, media_type='application/pdf')
