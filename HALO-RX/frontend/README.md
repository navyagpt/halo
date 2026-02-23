# HALO-RX Frontend

Web UI for nurses to:
- upload prescription images
- parse medicine details from table-like prescriptions
- run slash-command workflows for timetable/administer actions

## From Scratch Setup

### 1) Install system dependency (Tesseract)

macOS:

```bash
brew install tesseract
```

### 2) Install `uv` (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3) Create environment and install dependencies

From repository root (`HALO-RX/`):

```bash
cd HALO-RX
uv sync
```

## Run the Web UI

```bash
cd HALO-RX
uv run python -m streamlit run frontend/app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Slash Commands

- `/medicine-timetable`
  - generates timetable from parsed prescription
  - generated times stay within 07:00 to 19:00

- `/override-medicine-timetable`
  - opens editable timetable mode in UI
  - lets you update time slots manually

- `/medicine-administer <medicine-name>`
  - warns if medicine is not in parsed list
  - warns if current time is off-schedule
  - gives continue/cancel option
  - shows `med.png` on execution

- `/medicine-info <medicine-name>`
  - shows parsed row details for the medicine

- `/auto-administer`
  - same safety behavior as `/medicine-administer`
  - picks the closest scheduled medicine automatically
  - shows `med.png` on execution

- `/wrong-medicine`
  - available only after administer flow has been triggered
  - shows `med.png`

## OCR Output

Parsed prescription CSV is saved as:

`<uploaded_image_name>.csv`

## Optional CLI Use

If you want only CSV extraction without UI:

```bash
uv run python extract_table_to_csv.py presImg.png
```

## Troubleshooting

If Tesseract cannot find language data:
```bash
export TESSDATA_PREFIX="$(brew --prefix)/share/tessdata"
```

If OpenCV has an architecture mismatch such as `have 'arm64', need 'x86_64'`, HALO-RX falls back to a non-OpenCV OCR parser so the UI keeps working. For best table-extraction accuracy, rebuild the environment:
```bash
rm -rf .venv
uv venv --python 3.11
uv sync
uv run python -m streamlit run app.py
```

If `/medicine-administer` reports missing pipeline dependencies, run from repository root:
```bash
cd HALO-RX
uv sync
uv run python -m streamlit run frontend/app.py
```

If you place images in `image-input/` (repository root), HALO-RX prioritizes the newest image from that folder for pipeline runs.

If pipeline execution says `No crops found ...`, the detection model did not find instrument regions in the selected image. Re-upload the prescription image and rerun parsing so `/medicine-administer` uses the latest uploaded image.
