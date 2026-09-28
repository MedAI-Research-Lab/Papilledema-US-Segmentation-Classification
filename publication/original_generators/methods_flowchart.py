"""One editable vector scene, exported consistently as draw.io, SVG, PDF and PNG.

Only this audit directory is written. No clinical pixels or study code are read.
Run with the bundled Codex Python. PNGs are rendered from the final vector PDF.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import os
import shutil

from PIL import ImageFont
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

OUT = Path(__file__).resolve().parent
BASE = "Flowchart_2_visual"
W, H = 1800, 2760
PRINT_WIDTH_MM = 180
SCALE = PRINT_WIDTH_MM * mm / W
FONT = Path(os.environ.get("PUBLICATION_ARIAL_REGULAR", "Arial.ttf"))
BOLD = Path(os.environ.get("PUBLICATION_ARIAL_BOLD", "Arial-Bold.ttf"))
POPPLER = Path(os.environ.get("PUBLICATION_PDFTOPPM", shutil.which("pdftoppm") or "pdftoppm"))
NODE = Path(os.environ.get("PUBLICATION_NODE", shutil.which("node") or "node"))
SHARP = Path(os.environ.get("PUBLICATION_SHARP", "sharp"))
INK, GRAY, LIGHT = "#23333F", "#68737B", "#DEE3E6"
TEAL, BLUE, PURPLE = "#337C70", "#416B91", "#806291"
WHITE = "#FFFFFF"
items: list[dict] = []
edges: list[dict] = []
nodes: dict[str, tuple[float, float, float]] = {}


def circle(key, x, y, radius=18, color=GRAY, fill=WHITE):
    nodes[key] = (x, y, radius)
    items.append(dict(kind="circle", id=key, x=x-radius, y=y-radius,
                      w=radius*2, h=radius*2, color=color, fill=fill, stroke=4))


def text(key, value, x, y, width, size=32, bold=False, color=INK, line_height=None):
    f = ImageFont.truetype(str(BOLD if bold else FONT), size)
    leading = line_height or round(size*1.26)
    lines = []
    for explicit in value.split("\n"):
        current = ""
        for word in explicit.split():
            trial = current + (" " if current else "") + word
            if f.getlength(trial) > width and current:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    h = len(lines)*leading
    items.append(dict(kind="text", id=key, x=x, y=y, w=width, h=h, lines=lines,
                      size=size, bold=bold, color=color, leading=leading))
    return h


def line(key, points, color=GRAY, width=4, dashed=False, arrow=False,
         source=None, target=None, role="data"):
    edges.append(dict(id=key, points=points, color=color, width=width,
                      dashed=dashed, arrow=arrow, source=source, target=target, role=role))


def station(key, x, y, title, body, width, color, title_size=37, top=None, body_y=None):
    circle(key, x, y, color=color)
    title_y = y-49 if top is None else top
    th = text(key+"_title", title, x+52, title_y, width, title_size, True, color)
    text(key+"_body", body, x+52, title_y+th+8 if body_y is None else body_y,
         width, 32)


def lock(key, x, y, color, title, body, width):
    # An editable geometric symbol, not a clinical or acquired image.
    items.append(dict(kind="rect", id=key, x=x, y=y, w=36, h=29,
                      fill=WHITE, color=color, stroke=3))
    line(key+"_shackle", [(x+7,y),(x+7,y-17),(x+29,y-17),(x+29,y)], color, 3, role="icon")
    text(key+"_title", title, x+53, y-15, width-53, 32, True, color)
    text(key+"_body", body, x, y+51, width, 31, color=INK)


def build():
    text("title", "One anatomical route, two diagnostic tasks", 62, 39, 1676, 46, True)
    text("subtitle", "Strict predicted-ROI workflow  |  Schematic, not patient data", 65, 108, 1670, 32, color=GRAY)
    for key, x, col, name in [("legend_a",65,TEAL,"Anatomy"),("legend_b",550,BLUE,"Binary"),("legend_c",970,PURPLE,"Direct three-class")]:
        line(key+"_line", [(x,193),(x+70,193)], col, 6, role="legend")
        circle(key+"_dot", x+35,193,10,col)
        text(key+"_text", name,x+95,169,480,31,True,col)

    line("shared_route", [(102,307),(102,843)], GRAY, 5, source="cohort", target="segmenters")
    station("cohort",102,307,"Shared cohort", "91 patients · 182 eyes · 1,274 frames\nTwo eyes per patient; seven frames per eye",1580,GRAY)
    station("preprocess",102,485,"Fixed paired preprocessing", "Fixed crop and aspect-preserving resize/pad to 768 × 768\nPaired images and masks undergo preprocessing quality control",1580,GRAY)
    station("holdouts",102,663,"Five repeated patient holdouts", "55 train / 18 tuning / 18 test patients per split\nBoth eyes and all frames stay together; not outer 5-fold CV",1580,GRAY)
    station("segmenters",102,843,"Four anatomical segmenters", "YOLO26s · ViT Method2 · EMCAD (PVTv2-B0) · SAM2-U-Net (Hiera-Tiny)\nCommon binary anatomical target",1580,TEAL)

    line("segmenter_roi_route",[(102,861),(102,1019)],TEAL,5,source="segmenters",target="freeze_roi")
    line("frozen_anatomy_route",[(102,1055),(102,1882)],TEAL,5,source="freeze_roi",target="test_masks")
    line("test_mask_assessment_route",[(102,1918),(102,2357)],TEAL,5,arrow=True,source="test_masks",target="seg_assessment")
    station("freeze_roi",102,1037,"Freeze localization and build strict predicted ROIs", "Train: 5 inner OOF folds  |  Tuning: full-training-fit segmenter\nHard mask, tight crop and letterbox to 224 × 224; invalid ROI → abstain",1580,TEAL)

    text("same_roi", "Both strategies receive identical ROI artifacts within each model and split", 350,1150,1390,30,color=GRAY)
    line("roi_branch",[(102,1215),(1230,1215),(1230,1300)],TEAL,4,arrow=True,source="freeze_roi",target="three_train")
    line("binary_branch",[(670,1215),(670,1300)],TEAL,4,arrow=True,source="freeze_roi",target="binary_train")
    circle("anatomy_junction",102,1215,11,TEAL,TEAL)

    station("binary_train",670,1318,"Binary", "Normal vs P + PP\nNew two-logit classifiers\nPrimary: matched family\nSecondary: ResNet-18",480,BLUE,38,top=1263,body_y=1330)
    station("three_train",1230,1318,"Direct three-class", "Normal / P / PP\nNew three-logit classifiers\nPrimary: matched family\nSecondary: ResNet-18",470,PURPLE,38,top=1263,body_y=1330)
    text("noncascade", "Separate task training\nNo binary-to-three-class\nclassifier cascade",150,1292,430,32,True,TEAL)
    text("anatomy_independence", "Anatomical outputs\nfeed segmentation\nassessment directly",150,1485,430,32,color=TEAL)

    # Training/prediction routes are solid; separate dashed connectors express
    # the time conditions for original binary test access and extension access.
    line("binary_model_route",[(670,1336),(670,1882)],BLUE,5,arrow=True,source="binary_train",target="binary_aggregate")
    line("three_model_route",[(1230,1336),(1230,1882)],PURPLE,5,arrow=True,source="three_train",target="three_aggregate")
    lock("binary_gate",725,1600,BLUE,"Original test gate", "20 composite locks\nBoth classifier strategies\nCheckpoints, T, thresholds",435)
    lock("three_gate",1285,1600,PURPLE,"Extension test gate", "40 strategy locks\nCheckpoints and T\nTest ROIs only after gate",435)
    text("gate_legend", "Dashed = timing condition,\nnot a data input",150,1694,430,30,color=GRAY)
    line("binary_gate_condition",[(750,1775),(750,1810),(670,1810)],BLUE,3,True,True,"binary_gate","binary_aggregate","timing")
    line("three_gate_condition",[(1310,1775),(1310,1810),(1230,1810)],PURPLE,3,True,True,"three_gate","three_aggregate","timing")
    line("original_gate_anatomy_condition",[(710,1749),(613,1749),(613,1788),(102,1788)],GRAY,3,True,True,"binary_gate","test_masks","timing")

    station("test_masks",102,1900,"Locked test masks", "From the frozen anatomical\nfull-training-fit segmenter\nNo classifier output used",430,TEAL,35,top=1844,body_y=1901)
    station("binary_aggregate",670,1900,"Raw probabilities", "Valid frames → mean raw eye\nAt least 4 of 7 frames\nBoth raw eyes → mean patient\nBoth eyes must be valid",480,BLUE,35,top=1844,body_y=1901)
    station("three_aggregate",1230,1900,"Raw probability vectors", "Valid frames → mean raw eye\nAt least 4 of 7 frames\nBoth raw eyes → mean patient\nBoth eyes must be valid",470,PURPLE,35,top=1844,body_y=1901)

    circle("test_references",565,2132,14,GRAY)
    text("test_references_label","Test reference masks",150,2098,410,31,True,GRAY)
    text("test_references_detail","Retrospective assessment",150,2140,410,30,color=GRAY)
    line("references_to_assessment",[(565,2146),(565,2200),(102,2200)],GRAY,3,arrow=True,source="test_references",target="seg_assessment")

    line("binary_readout_route",[(670,1918),(670,2182)],BLUE,5,arrow=True,source="binary_aggregate",target="binary_calibration")
    line("three_readout_route",[(1230,1918),(1230,2182)],PURPLE,5,arrow=True,source="three_aggregate",target="three_calibration")
    station("binary_calibration",670,2200,"Calibrate final units", "Separate eye T and patient T\nApply tuning-locked values\nThen tuning-locked thresholds",480,BLUE,35,top=2147,body_y=2204)
    station("three_calibration",1230,2200,"Calibrate final units", "Separate eye T and patient T\nApply tuning-locked values\nArgmax; no confidence gate",470,PURPLE,35,top=2147,body_y=2204)
    station("seg_assessment",102,2375,"Segmentation assessment", "Dice · IoU · boundary metrics\nROI coverage",430,TEAL,33,top=2301,body_y=2390)
    line("binary_outcome_route",[(670,2218),(670,2452)],BLUE,5,arrow=True,source="binary_calibration",target="binary_outcome")
    line("three_outcome_route",[(1230,2218),(1230,2452)],PURPLE,5,arrow=True,source="three_calibration",target="three_outcome")
    station("binary_outcome",670,2470,"Primary: eye-level BA", "Failure-aware\nAbstentions count as errors",480,BLUE,35,top=2417,body_y=2473)
    station("three_outcome",1230,2470,"Primary: patient BA", "Failure-aware\nAbstentions count as errors",470,PURPLE,35,top=2417,body_y=2473)

    line("footer_rule",[(65,2585),(1735,2585)],LIGHT,2,role="divider")
    text("inference_safeguard", "Test reference masks never determine inference ROIs or abstention.",65,2610,1670,31,True)
    text("scope", "Exploratory, post-hoc internal studies using previously examined patient holdouts.",65,2656,1670,30,color=GRAY)
    text("abbreviations", "P, papilledema; PP, pseudopapilledema; OOF, out-of-fold; T, temperature; BA, balanced accuracy.",65,2701,1670,30,color=GRAY)


def arrow_vertices(edge):
    (ax,ay),(bx,by)=edge["points"][-2:]
    ang=math.atan2(by-ay,bx-ax)
    length,half=16,7
    return [(bx,by),(bx-length*math.cos(ang)+half*math.sin(ang),by-length*math.sin(ang)-half*math.cos(ang)),
            (bx-length*math.cos(ang)-half*math.sin(ang),by-length*math.sin(ang)+half*math.cos(ang))]


def svg_export():
    root=ET.Element("svg",xmlns="http://www.w3.org/2000/svg",width="180mm",height=f"{H/10}mm",viewBox=f"0 0 {W} {H}")
    ET.SubElement(root,"title").text="One anatomical route, two diagnostic tasks"
    ET.SubElement(root,"desc").text="Editable route map. Solid paths denote data/model flow; dashed connectors denote test-access timing conditions."
    ET.SubElement(root,"rect",width=str(W),height=str(H),fill=WHITE)
    for edge in edges:
        attrs={"id":edge["id"],"points":" ".join(f"{x},{y}" for x,y in edge["points"]),"fill":"none","stroke":edge["color"],"stroke-width":str(edge["width"]),"stroke-linejoin":"round","data-role":edge["role"]}
        if edge["source"]: attrs["data-source"]=edge["source"]
        if edge["target"]: attrs["data-target"]=edge["target"]
        if edge["dashed"]: attrs["stroke-dasharray"]="10 8"
        ET.SubElement(root,"polyline",**attrs)
        if edge["arrow"]: ET.SubElement(root,"polygon",points=" ".join(f"{x},{y}" for x,y in arrow_vertices(edge)),fill=edge["color"])
    for it in items:
        if it["kind"] in {"circle","rect"}:
            attrs=dict(id=it["id"],fill=it["fill"],stroke=it["color"],**{"stroke-width":str(it["stroke"])})
            if it["kind"]=="circle": ET.SubElement(root,"ellipse",cx=str(it["x"]+it["w"]/2),cy=str(it["y"]+it["h"]/2),rx=str(it["w"]/2),ry=str(it["h"]/2),**attrs)
            else: ET.SubElement(root,"rect",x=str(it["x"]),y=str(it["y"]),width=str(it["w"]),height=str(it["h"]),rx="2",**attrs)
        else:
            for n,s in enumerate(it["lines"]):
                node=ET.SubElement(root,"text",id=f"{it['id']}_{n}",x=str(it["x"]),y=str(it["y"]+it["size"]+n*it["leading"]),fill=it["color"],**{"font-family":"Arial, Helvetica, sans-serif","font-size":str(it["size"]),"font-weight":"700" if it["bold"] else "400"})
                node.text=s
    ET.indent(root)
    ET.ElementTree(root).write(OUT/f"{BASE}.svg",encoding="utf-8",xml_declaration=True)


def drawio_export():
    root=ET.Element("mxfile",host="app.diagrams.net",agent="Codex",version="24.7.17",type="device",compressed="false")
    diagram=ET.SubElement(root,"diagram",id="visual-route-map",name="Visual methods route map")
    graph=ET.SubElement(diagram,"mxGraphModel",dx=str(W),dy=str(H),grid="1",gridSize="10",page="1",pageScale="1",pageWidth=str(W),pageHeight=str(H),background=WHITE,math="0",shadow="0")
    cells=ET.SubElement(graph,"root")
    ET.SubElement(cells,"mxCell",id="0")
    ET.SubElement(cells,"mxCell",id="1",parent="0")
    for edge in edges:
        style=f"edgeStyle=none;rounded=0;strokeColor={edge['color']};strokeWidth={edge['width']};endArrow={'block' if edge['arrow'] else 'none'};endFill=1;endSize=12;"
        if edge["dashed"]: style+="dashed=1;dashPattern=10 8;"
        attrs=dict(id=edge["id"],parent="1",edge="1",style=style,**{"data-role":edge["role"]})
        if edge["source"]: attrs["data-source"]=edge["source"]
        if edge["target"]: attrs["data-target"]=edge["target"]
        cell=ET.SubElement(cells,"mxCell",**attrs)
        geom=ET.SubElement(cell,"mxGeometry",relative="1",**{"as":"geometry"})
        ET.SubElement(geom,"mxPoint",x=str(edge["points"][0][0]),y=str(edge["points"][0][1]),**{"as":"sourcePoint"})
        ET.SubElement(geom,"mxPoint",x=str(edge["points"][-1][0]),y=str(edge["points"][-1][1]),**{"as":"targetPoint"})
        bends=ET.SubElement(geom,"Array",**{"as":"points"})
        for x,y in edge["points"][1:-1]: ET.SubElement(bends,"mxPoint",x=str(x),y=str(y))
    for it in items:
        if it["kind"] in {"circle","rect"}:
            style=f"shape={'ellipse' if it['kind']=='circle' else 'rectangle'};fillColor={it['fill']};strokeColor={it['color']};strokeWidth={it['stroke']};shadow=0;"
            cell=ET.SubElement(cells,"mxCell",id=it["id"],parent="1",vertex="1",value="",style=style)
            ET.SubElement(cell,"mxGeometry",x=str(it["x"]),y=str(it["y"]),width=str(it["w"]),height=str(it["h"]),**{"as":"geometry"})
        else:
            for n,s in enumerate(it["lines"]):
                style=f"text;html=0;strokeColor=none;fillColor=none;align=left;verticalAlign=top;whiteSpace=wrap;rounded=0;spacing=0;fontFamily=Arial;fontSize={it['size']};fontColor={it['color']};fontStyle={1 if it['bold'] else 0};"
                cell=ET.SubElement(cells,"mxCell",id=f"{it['id']}_{n}",parent="1",vertex="1",value=s,style=style)
                ET.SubElement(cell,"mxGeometry",x=str(it["x"]),y=str(it["y"]+n*it["leading"]),width=str(it["w"]),height=str(it["leading"]+4),**{"as":"geometry"})
    ET.indent(root)
    ET.ElementTree(root).write(OUT/f"{BASE}.drawio",encoding="utf-8",xml_declaration=True)


def pdf_export():
    pdfmetrics.registerFont(TTFont("RouteArial",str(FONT)))
    pdfmetrics.registerFont(TTFont("RouteArialBold",str(BOLD)))
    height=H*SCALE
    c=canvas.Canvas(str(OUT/f"{BASE}.pdf"),pagesize=(W*SCALE,height),pageCompression=1,invariant=1)
    c.setTitle("One anatomical route, two diagnostic tasks")
    c.setAuthor("Methods audit")
    c.setFillColor(HexColor(WHITE));c.rect(0,0,W*SCALE,height,fill=1,stroke=0)
    for edge in edges:
        c.setStrokeColor(HexColor(edge["color"]));c.setFillColor(HexColor(edge["color"]));c.setLineWidth(edge["width"]*SCALE)
        c.setDash([10*SCALE,8*SCALE] if edge["dashed"] else [])
        p=c.beginPath()
        for n,(x,y) in enumerate(edge["points"]): (p.moveTo if n==0 else p.lineTo)(x*SCALE,height-y*SCALE)
        c.drawPath(p);c.setDash([])
        if edge["arrow"]:
            p=c.beginPath()
            for n,(x,y) in enumerate(arrow_vertices(edge)): (p.moveTo if n==0 else p.lineTo)(x*SCALE,height-y*SCALE)
            p.close();c.drawPath(p,fill=1,stroke=0)
    for it in items:
        if it["kind"] in {"circle","rect"}:
            c.setFillColor(HexColor(it["fill"]));c.setStrokeColor(HexColor(it["color"]));c.setLineWidth(it["stroke"]*SCALE)
            x,y,w,h=it["x"]*SCALE,height-(it["y"]+it["h"])*SCALE,it["w"]*SCALE,it["h"]*SCALE
            (c.ellipse if it["kind"]=="circle" else c.rect)(x,y,x+w if it["kind"]=="circle" else w,y+h if it["kind"]=="circle" else h,fill=1,stroke=1)
        else:
            c.setFillColor(HexColor(it["color"]));c.setFont("RouteArialBold" if it["bold"] else "RouteArial",it["size"]*SCALE)
            for n,s in enumerate(it["lines"]): c.drawString(it["x"]*SCALE,height-(it["y"]+it["size"]+n*it["leading"])*SCALE,s)
    c.showPage();c.save()


def validate():
    texts=[i for i in items if i["kind"]=="text"]
    assert min(i["size"]*SCALE for i in texts)>=8
    for i in items:
        assert 0<=i["x"] and 0<=i["y"] and i["x"]+i["w"]<=W and i["y"]+i["h"]<=H,("bounds",i["id"])
    # Check actual drawn line extents, not the much larger text cell rectangles.
    actual=[]
    for i in texts:
        f=ImageFont.truetype(str(BOLD if i["bold"] else FONT),i["size"])
        for n,s in enumerate(i["lines"]): actual.append((i["id"],i["x"],i["y"]+n*i["leading"]+5,f.getlength(s),i["size"]))
    overlaps=[]
    for n,(aid,ax,ay,aw,ah) in enumerate(actual):
        for bid,bx,by,bw,bh in actual[n+1:]:
            if min(ax+aw,bx+bw)>max(ax,bx)+1 and min(ay+ah,by+bh)>max(ay,by)+1: overlaps.append((aid,bid))
    assert not overlaps,("text overlaps",overlaps)
    for e in edges:
        if e["role"] in {"icon","legend","divider"}:continue
        for (x1,y1),(x2,y2) in zip(e["points"],e["points"][1:]):
            for key,x,y,w,h in actual:
                hit=(x1==x2 and x<x1<x+w and max(min(y1,y2),y)<min(max(y1,y2),y+h)) or (y1==y2 and y<y1<y+h and max(min(x1,x2),x)<min(max(x1,x2),x+w))
                assert not hit,("edge crosses text",e["id"],key)
    incoming=[e for e in edges if e["target"]=="seg_assessment" and e["role"]=="data"]
    assert {e["source"] for e in incoming}=={"test_masks","test_references"}
    assert not any(e["source"]=="binary_train" and e["target"]=="three_train" for e in edges)
    assert all(e["dashed"] for e in edges if e["role"]=="timing")


CAPTION="""# Figure 1, alternative route-map design. Anatomical segmentation and two independently trained diagnostic tasks

The station map uses a shared preparation route, an anatomical track (green), a binary classification track (blue), and a direct three-class track (purple). All symbols are schematic vector marks, not images or measurements from patients. The cohort comprised 91 patients, 182 eyes and 1,274 frames, with seven frames per eye. Paired images and reference masks underwent the fixed preprocessing and quality-control pipeline before partition assignment. Five repeated, fixed patient-level holdouts allocated 55/18/18 patients to training/tuning/test; they are not five disjoint outer cross-validation folds. Tuning denotes the partition named `validation` in the code. Within each outer training partition, five patient-grouped inner folds generated out-of-fold predicted masks for classifier training; the full-training-fit anatomical segmenter generated tuning masks and, only after the original study's gate, test masks. All segmenters learned the same binary anatomical ROI target, irrespective of diagnosis. Segmenter checkpoints and ROI policy were frozen. Accepted ROIs were hard-masked, tightly cropped and letterboxed to 224 × 224; invalid ROIs produced abstention without full-image, unmasked-box or ground-truth-ROI fallback.

The two diagnostic tasks are normal versus combined papilledema/pseudopapilledema and direct normal/papilledema/pseudopapilledema classification. Each task trains new classifiers: its primary strategy uses the architecture matched to the segmentation family; the secondary strategy uses independently fitted ImageNet-initialized ResNet-18 models on identical ROI artifacts within each model and split. There is no binary-to-three-class cascade. The three-class extension reuses frozen anatomical artifacts, not binary classifier weights, logits, temperatures or thresholds. The original binary gate required all 20 model-seed composite validation locks, each covering both classifier strategies and eye/patient temperatures and thresholds. The later three-class extension required all 40 model-seed-strategy locks before test ROI artifact import. Dashed connectors denote these timing conditions, not model inputs.

Valid frame probabilities (binary scalars or three-class vectors) are averaged to raw eye probabilities when at least four of seven frames are valid. Raw patient probabilities average the two raw eye probabilities only when both eyes are evaluable. Temperature calibration is fitted independently on validation data at each final unit level; patient calibration is not an average of calibrated eye probabilities. Binary decisions use validation-locked thresholds after calibration; three-class primary decisions use argmax without an additional confidence-rejection gate. The primary endpoints are failure-aware eye-level balanced accuracy for binary classification and failure-aware patient-level balanced accuracy for the three-class task. Structural abstentions count as errors in the true-class denominators; coverage and conditional discrimination/calibration results accompany these endpoints.

The anatomical branch is classifier-independent: frozen anatomical test predictions and test reference masks feed segmentation assessment directly. Test reference masks never determine inference ROIs or abstention; they are available for retrospective segmentation/localization assessment, and paired masks were also present during earlier preprocessing quality control. The figure deliberately omits loss, optimizer, architecture-readout and statistical-comparison detail, which belongs in the Methods. Both studies are exploratory, post-hoc internal analyses using previously examined holdouts; across-split mean and SD are descriptive, not independent-cohort inference. The binary study also has a documented evaluation-only, post-test-access gate-path compatibility amendment, without retraining or changes to predictions, calibration or thresholds.

Abbreviations: P, papilledema; PP, pseudopapilledema; OOF, out-of-fold; ROI, region of interest; T, temperature; BA, balanced accuracy. This figure is designed at 180 mm print width with a minimum text size of 8.50 pt.
"""


def semantic_qa():
    svg=ET.parse(OUT/f"{BASE}.svg").getroot()
    drawio=ET.parse(OUT/f"{BASE}.drawio").getroot()
    svg_text=[n.text or "" for n in svg.iter() if n.tag.endswith("}text")]
    drawio_text=[n.get("value") for n in drawio.iter("mxCell") if n.get("value")]
    expected=[s for i in items if i["kind"]=="text" for s in i["lines"]]
    assert expected==svg_text==drawio_text,"SVG and draw.io text differ"
    pdf=PdfReader(str(OUT/f"{BASE}.pdf"))
    pdf_text=pdf.pages[0].extract_text()
    norm=lambda s:" ".join(s.split())
    for s in expected: assert norm(s) in norm(pdf_text),("Missing PDF label",s)
    svgedges={n.get("id"):(n.get("data-source"),n.get("data-target"),n.get("data-role")) for n in svg.iter() if n.tag.endswith("}polyline")}
    dxedges={n.get("id"):(n.get("data-source"),n.get("data-target"),n.get("data-role")) for n in drawio.iter("mxCell") if n.get("edge")=="1"}
    assert svgedges==dxedges
    assert abs(float(pdf.pages[0].mediabox.width)/mm-180)<0.001
    report=dict(status="passed",canvas=[W,H],print_width_mm=180,print_height_mm=H/10,
                minimum_font_pt=round(min(i["size"]*SCALE for i in items if i["kind"]=="text"),3),
                text_lines=len(expected),edge_records=len(edges),identical_svg_drawio_text=True,
                identical_svg_drawio_edge_semantics=True,pdf_all_text_present=True,
                no_text_overlap=True,no_data_edge_through_text=True,
                segmentation_inputs=[e["source"] for e in edges if e["target"]=="seg_assessment" and e["role"]=="data"],
                test_access_edges_are_timing_only=True,embedded_clinical_images=0,
                formats={ext:hashlib.sha256((OUT/f"{BASE}.{ext}").read_bytes()).hexdigest() for ext in ["drawio","svg","pdf","png"]})
    (OUT/f"{BASE}_qa.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


def render_exported_vectors_for_qa():
    """Rasterize the exported SVG and a primitive-level draw.io XML reconstruction.

    The latter is explicitly an XML/geometry renderer, not a claim of opening
    the file in diagrams.net. It catches missing nodes, style/coordinate drift,
    and wrong editable labels independently of the original scene/PDF renderer.
    """
    qa=OUT/f"{BASE}_QA"
    qa.mkdir(exist_ok=True)
    root=ET.Element("svg",xmlns="http://www.w3.org/2000/svg",width="180mm",height=f"{H/10}mm",viewBox=f"0 0 {W} {H}")
    ET.SubElement(root,"rect",width=str(W),height=str(H),fill=WHITE)
    dx=ET.parse(OUT/f"{BASE}.drawio").getroot()
    for cell in dx.iter("mxCell"):
        raw=cell.get("style","")
        style=dict(part.split("=",1) for part in raw.split(";") if "=" in part)
        geom=cell.find("mxGeometry")
        if geom is None:continue
        if cell.get("edge")=="1":
            p0=geom.find("mxPoint[@as='sourcePoint']")
            p1=geom.find("mxPoint[@as='targetPoint']")
            points=[(float(p0.get("x")),float(p0.get("y")))]
            points.extend((float(p.get("x")),float(p.get("y"))) for p in geom.findall("Array/mxPoint"))
            points.append((float(p1.get("x")),float(p1.get("y"))))
            attrs={"points":" ".join(f"{x},{y}" for x,y in points),"fill":"none","stroke":style["strokeColor"],"stroke-width":style["strokeWidth"]}
            if style.get("dashed")=="1":attrs["stroke-dasharray"]=style["dashPattern"]
            ET.SubElement(root,"polyline",**attrs)
            if style.get("endArrow")=="block":
                ET.SubElement(root,"polygon",points=" ".join(f"{x},{y}" for x,y in arrow_vertices({"points":points})),fill=style["strokeColor"])
        elif cell.get("vertex")=="1":
            x,y,w,h=(float(geom.get(k)) for k in ("x","y","width","height"))
            if cell.get("value"):
                s=float(style["fontSize"])
                ET.SubElement(root,"text",x=str(x),y=str(y+s),fill=style["fontColor"],**{"font-family":style["fontFamily"],"font-size":str(s),"font-weight":"700" if style["fontStyle"]=="1" else "400"}).text=cell.get("value")
            else:
                attrs={"fill":style["fillColor"],"stroke":style["strokeColor"],"stroke-width":style["strokeWidth"]}
                if style.get("shape")=="ellipse":ET.SubElement(root,"ellipse",cx=str(x+w/2),cy=str(y+h/2),rx=str(w/2),ry=str(h/2),**attrs)
                else:ET.SubElement(root,"rect",x=str(x),y=str(y),width=str(w),height=str(h),**attrs)
    check_svg=qa/"drawio_xml_render.svg"
    ET.ElementTree(root).write(check_svg,encoding="utf-8",xml_declaration=True)
    js="const sharp=require(process.argv[1]); sharp(process.argv[2], {density:150}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});"
    for src,name in [(OUT/f"{BASE}.svg","svg_render.png"),(check_svg,"drawio_xml_render.png")]:
        subprocess.run([str(NODE),"-e",js,str(SHARP),str(src),str(qa/name)],check=True)


def main():
    build();validate();svg_export();drawio_export();pdf_export()
    (OUT/f"{BASE}_caption.md").write_text(CAPTION,encoding="utf-8")
    for dpi,suffix in [(600,""),(150,"_preview")]:
        subprocess.run([str(POPPLER),"-singlefile","-r",str(dpi),"-png",str(OUT/f"{BASE}.pdf"),str(OUT/f"{BASE}{suffix}")],check=True)
    semantic_qa()
    render_exported_vectors_for_qa()
    print(json.dumps({"status":"complete","base":str(OUT/BASE),"minimum_font_pt":30*SCALE}))


if __name__=="__main__":main()
