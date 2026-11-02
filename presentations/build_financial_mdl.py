"""Build an editable research deck from frozen result CSVs; never fit a model."""

import argparse
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[1]
FONT = "Arial"
INK = "19283D"
MUTED = "526178"
ACCENT = "CA2854"
GRID = "DFE4EB"
COLORS = ["440154", "443983", "31688E", "21918C", "35B779", "90D743", "FDE725"]


def px(value):
    return round(value * 9525)


def rgb(value):
    return RGBColor.from_string(value)


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def synchronize_chart_caches(chart):
    """Use the embedded worksheet's serialization for the chart's numeric cache.

    Excel workbook writers and chart XML writers can serialize the same float with
    different last digits. Matching their strings preserves native editing and avoids
    a spurious change when PowerPoint refreshes a chart from its workbook.
    """
    with zipfile.ZipFile(io.BytesIO(chart.part.chart_workbook.xlsx_part.blob)) as workbook:
        sheets = [name for name in workbook.namelist() if name.startswith("xl/worksheets/sheet")]
        assert sheets == ["xl/worksheets/sheet1.xml"]
        root = ET.fromstring(workbook.read(sheets[0]))
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        cells = {
            cell.attrib["r"]: cell.find("x:v", ns).text
            for cell in root.findall(".//x:c", ns)
            if cell.find("x:v", ns) is not None
        }
    for reference in chart._chartSpace.xpath(".//c:numRef"):
        namespace = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
        address = reference.find(f"{namespace}f").text.split("!")[1].replace("$", "")
        start = re.fullmatch(r"([A-Z]+)(\d+)", address.split(":")[0])
        assert start is not None
        column, first_row = start.group(1), int(start.group(2))
        for point in reference.findall(f"{namespace}numCache/{namespace}pt"):
            value = point.find(f"{namespace}v")
            actual = cells[f"{column}{first_row + int(point.attrib['idx'])}"]
            assert abs(float(value.text) - float(actual)) < 1e-10
            value.text = actual


def text(slide, value, left, top, width, height, size=22, color=INK, bold=False, align=None):
    shape = slide.shapes.add_textbox(px(left), px(top), px(width), px(height))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for index, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = rgb(color)
        paragraph.space_after = Pt(6)
        if align is not None:
            paragraph.alignment = align
    return shape


def table(slide, values, top, height, widths, *, left=72, font_size=19, highlight=None):
    native = slide.shapes.add_table(
        len(values), len(values[0]), px(left), px(top), px(sum(widths)), px(height)
    ).table
    for attribute in ("firstRow", "firstCol", "lastRow", "lastCol", "bandRow", "bandCol"):
        native._tbl.tblPr.set(attribute, "0")
    for column, width in zip(native.columns, widths, strict=True):
        column.width = px(width)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            cell = native.cell(i, j)
            cell.text = str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = px(13)
            cell.margin_top = cell.margin_bottom = px(5)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(
                INK if i == 0 else "FFF2F5" if i == highlight else "FFFFFF"
            )
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = FONT
                paragraph.font.size = Pt(font_size if i else font_size - 1)
                paragraph.font.bold = i == 0 or i == highlight
                paragraph.font.color.rgb = rgb("FFFFFF" if i == 0 else INK)
                paragraph.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            tc_pr = cell._tc.get_or_add_tcPr()
            for edge_index, edge in enumerate(("lnL", "lnR", "lnT", "lnB")):
                old = tc_pr.find(f"{{http://schemas.openxmlformats.org/drawingml/2006/main}}{edge}")
                if old is not None:
                    tc_pr.remove(old)
                border = OxmlElement(f"a:{edge}")
                border.set("w", "6350")
                fill = OxmlElement("a:solidFill")
                color = OxmlElement("a:srgbClr")
                color.set("val", GRID)
                fill.append(color)
                border.append(fill)
                tc_pr.insert(edge_index, border)
    return native


def chart_fonts(chart):
    for axis in (chart.category_axis, chart.value_axis):
        axis.tick_labels.font.name = FONT
        axis.tick_labels.font.size = Pt(13)
        axis.format.line.color.rgb = rgb(GRID)
        if axis.has_title:
            for paragraph in axis.axis_title.text_frame.paragraphs:
                paragraph.font.name = FONT
                paragraph.font.size = Pt(15)
                paragraph.font.color.rgb = rgb(MUTED)
    if chart.has_legend:
        chart.legend.font.name = FONT
        chart.legend.font.size = Pt(12)
    chart.font.name = FONT
    chart.font.size = Pt(14)


def scatter(slide, rows, *, zoom, position):
    data = XyChartData()
    for count in range(7):
        series = data.add_series(f"k={count}")
        for row in rows:
            if int(row["n_features"]) == count:
                series.add_data_point(
                    float(row["mean_total_bits"]), float(row["mean_heldout_rmse"])
                )
    full = next(row for row in rows if row["is_full"] == "True")
    selected = next(row for row in rows if int(row["selection_count"]) == 5)
    for label, row in (("Complet", full), ("MDL", selected)):
        data.add_series(label).add_data_point(
            float(row["mean_total_bits"]), float(row["mean_heldout_rmse"])
        )
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER, *(px(value) for value in position), data
    ).chart
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = True
    for index, series in enumerate(chart.series):
        marker = series.marker
        marker.style = XL_MARKER_STYLE.CIRCLE
        marker.size = 6
        marker.format.fill.solid()
        marker.format.fill.fore_color.rgb = rgb(
            COLORS[index] if index < 7 else INK if index == 7 else ACCENT
        )
        marker.format.line.fill.background()
        series.format.line.fill.background()
        if index == 7:
            marker.style, marker.size = XL_MARKER_STYLE.SQUARE, 11
            marker.format.fill.fore_color.rgb = rgb("FFFFFF")
            marker.format.line.color.rgb = rgb(INK)
            marker.format.line.width = Pt(1.5)
        elif index == 8:
            marker.style, marker.size = XL_MARKER_STYLE.SQUARE, 12
            marker.format.line.color.rgb = rgb(ACCENT)
            marker.format.line.width = Pt(1.2)
    horizontal, vertical = chart.category_axis, chart.value_axis
    horizontal.has_title = vertical.has_title = True
    horizontal.axis_title.text_frame.text = "Longueur MDL moyenne (bits)"
    vertical.axis_title.text_frame.text = "RMSE hors entraînement"
    horizontal.tick_labels.number_format = "0"
    vertical.tick_labels.number_format = "0.0000" if zoom else "0.0"
    horizontal.has_major_gridlines = True
    vertical.has_major_gridlines = True
    for axis in (horizontal, vertical):
        axis.major_gridlines.format.line.color.rgb = rgb(GRID)
        axis.major_gridlines.format.line.width = Pt(0.5)
    if zoom:
        limit = float(full["mean_heldout_rmse"]) * 1.10
        visible = [row for row in rows if float(row["mean_heldout_rmse"]) <= limit]
        xs = [float(row["mean_total_bits"]) for row in visible]
        ys = [float(row["mean_heldout_rmse"]) for row in visible]
        x_padding = max(1, (max(xs) - min(xs)) * 0.08)
        y_padding = max((max(ys) - min(ys)) * 0.2, float(full["mean_heldout_rmse"]) * 0.0005)
        horizontal.minimum_scale, horizontal.maximum_scale = (
            min(xs) - x_padding,
            max(xs) + x_padding,
        )
        vertical.minimum_scale, vertical.maximum_scale = min(ys) - y_padding, max(ys) + y_padding
        vertical.major_unit = 0.0005
        horizontal.major_unit = 100
    else:
        horizontal.minimum_scale, horizontal.maximum_scale = 8600, 10500
        vertical.minimum_scale, vertical.maximum_scale = 0.3, 2.8
        vertical.major_unit = 0.5
    chart_fonts(chart)
    return chart


def financial_chart(slide, rows):
    ids = ["full", "mdl", "persistence-20"]
    selected = {row["candidate"]: row for row in rows if row["phase"] == "test"}
    data = CategoryChartData()
    data.categories = ["Ridge\ncomplet", "Ridge\n+ MDL", "Persistance\n20 séances"]
    data.add_series("RMSE test", [float(selected[name]["rmse"]) for name in ids])
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, px(64), px(172), px(770), px(405), data
    ).chart
    chart.has_title = chart.has_legend = False
    chart.value_axis.minimum_scale, chart.value_axis.maximum_scale = 0, 0.12
    chart.value_axis.major_unit = 0.02
    chart.value_axis.tick_labels.number_format = "0.00"
    chart.value_axis.has_title = True
    chart.value_axis.axis_title.text_frame.text = "RMSE sur le test 2024–2025"
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = rgb(GRID)
    chart.plots[0].gap_width = 65
    chart.plots[0].has_data_labels = True
    labels = chart.plots[0].data_labels
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.number_format = "0.000000"
    labels.font.name, labels.font.size = FONT, Pt(16)
    for point, color in zip(chart.series[0].points, ("586E85", ACCENT, "208D93"), strict=True):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = rgb(color)
        point.format.line.fill.background()
    chart_fonts(chart)
    return chart


def build(output):
    content_path = ROOT / "presentations/financial-mdl/content.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    synthetic = read_csv("docs/results/mdl-frontier-v1-summary.csv")
    signal = [row for row in synthetic if row["scenario"] == "signal"]
    financial = read_csv("docs/results/spy-final-test-v1-summary.csv")
    sensitivity = read_csv("docs/results/mdl-sensitivity-v1-summary.csv")
    test = {row["candidate"]: row for row in financial if row["phase"] == "test"}
    delta = 100 * (float(test["mdl"]["rmse"]) / float(test["full"]["rmse"]) - 1)
    assert len(signal) == 64
    assert sum(slide.get("duration_seconds", 0) for slide in content["slides"]) == 420
    assert len(content["slides"]) == 10
    assert abs(delta - 0.7708528863568) < 1e-10
    deck = Presentation()
    deck.slide_width, deck.slide_height = 12192000, 6858000
    deck.core_properties.title = content["title"]
    deck.core_properties.subject = "Projet 5AI25 : MDL et prévision de volatilité"
    deck.core_properties.author = ""
    deck.core_properties.last_modified_by = ""
    deck.core_properties.created = datetime.now(UTC).replace(tzinfo=None)
    deck.core_properties.modified = deck.core_properties.created
    deck.core_properties.keywords = "MDL, Ridge, sélection de variables, volatilité, SPY"
    for index, entry in enumerate(content["slides"]):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb("FFFFFF")
        if entry.get("hidden"):
            slide._element.set("show", "0")
        if index:
            text(slide, entry["title"], 72, 48, 1136, 70, size=32, bold=True)
        text(
            slide,
            str(index + 1) if index < 6 else f"A{index - 5}",
            1150,
            676,
            58,
            22,
            size=12,
            color=MUTED,
            align=PP_ALIGN.RIGHT,
        )
        sources = "\n".join(
            f"https://github.com/MipeyV/financial-mdl/blob/336dc09/{source}"
            if source.startswith(("docs/", "src/", "protocols/", "experiments/"))
            else source
            for source in entry["sources"]
        )
        timing = (
            f"Durée indicative : {entry['duration_seconds']} secondes\n\n"
            if "duration_seconds" in entry
            else "Annexe pour les questions\n\n"
        )
        slide.notes_slide.notes_text_frame.text = (
            timing + entry["notes"] + "\n\nSources\n" + sources
        )
        kind = entry["id"]
        if kind == "question":
            text(slide, entry["title"], 72, 154, 1100, 165, size=46, bold=True)
            text(
                slide,
                "Une représentation plus courte\nconserve-t-elle la qualité prédictive ?",
                74,
                349,
                1090,
                130,
                size=29,
                color=ACCENT,
            )
            text(slide, "Prévision de volatilité sur SPY", 74, 535, 1050, 48, size=22, color=MUTED)
            text(slide, "Projet 5AI25, Télécom Paris", 74, 636, 900, 35, size=17, color=MUTED)
        elif kind == "criterion":
            text(
                slide,
                "L(S) + L(θ sachant S, X) + L(y sachant θ, S, X)",
                72,
                158,
                1136,
                60,
                size=27,
                bold=True,
            )
            text(
                slide,
                "Sous-ensemble, paramètres et cibles quantifiées, en bits",
                72,
                233,
                1136,
                45,
                size=21,
                color=MUTED,
            )
            table(
                slide,
                [
                    ["Modèle", "Identif.", "Paramètres", "Cibles", "Total"],
                    ["Ridge complet\n5 variables", "3", "448", "19 706", "20 157"],
                    ["Ridge + MDL\n3 variables", "7", "320", "19 718", "20 045"],
                ],
                320,
                216,
                [300, 160, 210, 230, 236],
                highlight=2,
            )
            text(
                slide,
                "112 bits économisés sur l’entraînement",
                72,
                574,
                1136,
                48,
                size=25,
                bold=True,
                color=ACCENT,
            )
            text(
                slide,
                "X connu. La RMSE hors entraînement ne sélectionne pas les variables.",
                72,
                635,
                1090,
                36,
                size=17,
                color=MUTED,
            )
        elif kind == "protocol":
            text(
                slide,
                "SPY quotidien : volatilité des cinq séances suivantes",
                72,
                153,
                1136,
                48,
                size=25,
                bold=True,
            )
            text(
                slide,
                "Même snapshot, mêmes dates éligibles et même cible pour tous les candidats",
                72,
                219,
                1136,
                65,
                size=21,
                color=MUTED,
            )
            table(
                slide,
                [
                    ["Entraînement", "Validation", "Test final"],
                    [
                        "2015–2021\n1 738 origines",
                        "2022–2023\n496 origines",
                        "2024–2025\n497 origines",
                    ],
                    [
                        "Ajustement et\nsélection MDL",
                        "Comparaison\ndes approches",
                        "Candidats\nfigés",
                    ],
                ],
                319,
                218,
                [379, 378, 379],
                font_size=20,
            )
            text(
                slide,
                "Tolérance exploratoire déclarée : RMSE MDL / RMSE complet ≤ 1,02",
                72,
                585,
                1136,
                74,
                size=22,
                color=ACCENT,
            )
        elif kind == "synthetic":
            text(
                slide,
                "600 observations d’entraînement, 300 d’évaluation, cinq graines",
                72,
                131,
                1136,
                38,
                size=19,
                color=MUTED,
            )
            scatter(slide, signal, zoom=True, position=(40, 190, 858, 405))
            text(slide, "Signaux forts", 920, 211, 300, 42, size=21, bold=True)
            text(
                slide,
                "2 variables sur 6\nSignaux conservés",
                920,
                259,
                300,
                92,
                size=21,
                color=ACCENT,
            )
            text(slide, "Bruit seul", 920, 381, 300, 38, size=21, bold=True)
            text(slide, "Modèle constant", 920, 427, 300, 42, size=19, color=MUTED)
            text(slide, "Signal faible", 920, 502, 300, 38, size=21, bold=True)
            text(slide, "Constant à n = 600", 920, 548, 300, 42, size=19, color=MUTED)
            text(
                slide,
                "Point = combinaison, moyenne sur 5 graines. "
                "Couleur : k variables. Zoom sur de faibles écarts.",
                72,
                628,
                1136,
                50,
                size=17,
                color=MUTED,
            )
        elif kind == "financial":
            text(
                slide,
                "497 prévisions communes sur 2024–2025",
                72,
                131,
                1136,
                38,
                size=20,
                color=MUTED,
            )
            financial_chart(slide, financial)
            text(slide, "3 variables sur 5", 880, 217, 328, 58, size=26, bold=True)
            text(slide, "40 % en moins", 880, 287, 328, 59, size=27, color=ACCENT, bold=True)
            text(slide, f"+{delta:.2f} %".replace(".", ","), 880, 397, 328, 70, size=39, bold=True)
            text(slide, "de RMSE face\nà Ridge complet", 880, 479, 328, 83, size=21, color=MUTED)
            text(
                slide,
                "Validation : −2,25 % de RMSE. Test : +0,77 %, dans la tolérance de +2 %.",
                72,
                623,
                1136,
                56,
                size=19,
                color=MUTED,
            )
        elif kind == "conclusion":
            text(
                slide,
                "Un modèle plus court respecte la tolérance\nsur cette période de test",
                72,
                153,
                1136,
                116,
                size=29,
                bold=True,
                color=ACCENT,
            )
            text(
                slide,
                "Le codage choisi influence la sélection\n"
                "Une seule série financière, des cibles qui se chevauchent\n"
                "Pas de supériorité statistique démontrée",
                72,
                333,
                1136,
                210,
                size=23,
            )
            text(
                slide,
                "La compacité et la qualité prédictive se mesurent séparément",
                72,
                587,
                1136,
                64,
                size=22,
                bold=True,
            )
        elif kind == "coding_details":
            table(
                slide,
                [
                    ["Terme", "Convention"],
                    ["Identifier S", "⌈log₂(p + 1)⌉ + ⌈log₂ C(p, k)⌉"],
                    ["Paramètres", "64 × (k + 2) bits"],
                    ["Cibles sachant le modèle", "⌈−Σ log₂ P(rᵢ sachant q)⌉"],
                ],
                164,
                268,
                [400, 736],
                font_size=22,
            )
            text(
                slide,
                "p = 5, k = 3 : identification en 3 + 4 = 7 bits",
                72,
                471,
                1136,
                50,
                size=23,
                bold=True,
                color=ACCENT,
            )
            text(
                slide,
                "k coefficients, une constante, un paramètre q\n"
                "Cibles quantifiées à δ = 0,0001, X connu",
                72,
                545,
                1136,
                92,
                size=21,
                color=MUTED,
            )
        elif kind == "sensitivity":
            values = [["Entraînement", "δ = 0,0001", "δ = 0,01", "δ = 0,1"]]
            for size in (150, 600, 2400, 9600):
                row = [f"{size:,}".replace(",", " ")]
                for precision in (0.0001, 0.01, 0.1):
                    record = next(
                        r
                        for r in sensitivity
                        if r["scenario"] == "weak_signal"
                        and r["method"] == "mdl"
                        and int(r["n_train"]) == size
                        and float(r["target_precision"]) == precision
                    )
                    row.append(f"{round(float(record['mean_features']) * 5)}/5")
                values.append(row)
            text(
                slide,
                "Graines conservant le vrai signal faible",
                72,
                141,
                1136,
                48,
                size=23,
                color=MUTED,
            )
            table(slide, values, 223, 278, [310, 275, 275, 276], font_size=22, highlight=4)
            text(
                slide,
                "67 bits supplémentaires à rembourser par le gain de codage",
                72,
                548,
                1136,
                56,
                size=23,
                bold=True,
                color=ACCENT,
            )
            text(
                slide,
                "Le codage du cas SPY reste inchangé après cette étude synthétique.",
                72,
                621,
                1136,
                42,
                size=18,
                color=MUTED,
            )
        elif kind == "full_scatter":
            scatter(slide, signal, zoom=False, position=(54, 153, 1154, 473))
            text(
                slide,
                "Les 64 sous-ensembles sont inclus. "
                "Les copies exactes créent des points superposés.",
                72,
                639,
                1136,
                40,
                size=17,
                color=MUTED,
            )
        elif kind == "scope":
            table(
                slide,
                [
                    ["Mesure", "Portée"],
                    ["Longueur de description", "Code explicite, conditionnel à X"],
                    ["Paramètres", "Binary64 réellement sérialisés"],
                    ["Résidus", "Longueurs calculées, flux binaire non produit"],
                    ["Tolérance de RMSE", "Règle exploratoire, pas un test statistique"],
                ],
                175,
                330,
                [370, 766],
                font_size=21,
            )
            text(
                slide,
                "Le test final est désormais consulté. Toute évolution informée\n"
                "par ses scores demande un nouveau plan d’évaluation.",
                72,
                560,
                1136,
                93,
                size=21,
                color=MUTED,
            )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_chart:
                synchronize_chart_caches(shape.chart)
    deck.save(output)
    sources = [
        "presentations/financial-mdl/content.json",
        "docs/results/mdl-frontier-v1-summary.csv",
        "docs/results/spy-final-test-v1-summary.csv",
        "docs/results/mdl-sensitivity-v1-summary.csv",
    ]
    (output.parent / "sources.json").write_text(
        json.dumps(
            {
                "main_slides": 6,
                "hidden_appendices": 4,
                "duration_seconds": 420,
                "source_sha256": {
                    path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources
                },
                "all_charts_and_tables_native": True,
                "python_pptx_version": "1.0.2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
