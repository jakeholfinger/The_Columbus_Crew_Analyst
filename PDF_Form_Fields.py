import io
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader, PdfWriter

# Matches the figure size PdfPages pages are created at in Pre_Match_Report.py
# (plt.figure(figsize=(8.5, 11))) — reportlab's 'letter' page size is the same
# 612x792pt, so a matplotlib add_axes bbox ([x0, y0, width, height], figure-fraction,
# bottom-left origin) converts to reportlab points with a straight multiply, no axis
# flip needed.
PAGE_WIDTH_PT, PAGE_HEIGHT_PT = letter

# The rest of the report (every page.text() call in Pre_Match_Report.py/
# Tactical_Visualizations.py) uses matplotlib's default font, DejaVu Sans. reportlab's
# AcroForm text fields can't match that — AcroForm.makeFont only accepts one of the
# standard 14 base PDF fonts (confirmed: it hard-rejects anything outside that fixed
# dict, and there's no supported way to embed a custom TTF into a form field). Of those
# 14, Helvetica is the closest visual match to DejaVu Sans (both plain sans-serif), so
# it's used explicitly here rather than left as an implicit default.
NOTES_FONT_NAME = 'Helvetica'


def add_fillable_text_fields(pdf_path, page_index, fields):
    '''Overlays real, clickable AcroForm text fields onto an already-rendered PDF at
    pdf_path. matplotlib's PdfPages only ever produces flattened, static vector content
    — there's no way to create an interactive form field from within matplotlib itself
    — so this is a separate post-processing pass: draw the fields on a blank reportlab
    page, then merge that page onto the real one with pypdf.

    fields: list of (name, [x0, y0, width, height]) bboxes, in the same figure-fraction,
    bottom-left-origin convention used by page.add_axes() throughout this project.
    '''
    overlay_buffer = io.BytesIO()
    overlay_canvas = canvas.Canvas(overlay_buffer, pagesize=letter)
    form = overlay_canvas.acroForm

    for name, bbox in fields:
        x0, y0, width, height = bbox
        form.textfield(
            name=name,
            tooltip='Scouting notes',
            x=x0 * PAGE_WIDTH_PT,
            y=y0 * PAGE_HEIGHT_PT,
            width=width * PAGE_WIDTH_PT,
            height=height * PAGE_HEIGHT_PT,
            fontName=NOTES_FONT_NAME,
            fontSize=10,
            fillColor=None,
            borderColor=None,
            borderWidth=0,
            forceBorder=False,
            fieldFlags='multiline',
            maxlen=2000,
        )

    overlay_canvas.showPage()
    overlay_canvas.save()
    overlay_buffer.seek(0)
    overlay_page = PdfReader(overlay_buffer).pages[0]

    writer = PdfWriter(clone_from=pdf_path)
    writer.pages[page_index].merge_page(overlay_page)

    # merge_page carries the widget annotations onto the page, but the document-level
    # /AcroForm /Fields array doesn't automatically pick them up afterward — a known
    # pypdf gap with a purpose-built fix.
    writer.reattach_fields()

    # Write to a temp path first: pdf_path is still open for reading (writer was cloned
    # from it) at this point, so overwriting it directly would corrupt the read.
    temp_path = pdf_path + '.tmp'
    with open(temp_path, 'wb') as f:
        writer.write(f)
    os.replace(temp_path, pdf_path)