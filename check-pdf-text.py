# See https://stackoverflow.com/a/33533505/434243
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import ceil
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Line:
    start: float
    end: float

    @property
    def length(self):
        return self.end - self.start

    @property
    def valid(self):
        return self.end >= self.start

    def intersect(self, other: Line) -> Line:
        if not self.valid or not other.valid:
            return self.empty()
        return Line(start=max(self.start, other.start), end=min(self.end, other.end))

    def union(self, other: Line) -> Line:
        if not self.valid or not other.valid:
            return self.empty()
        return Line(start=min(self.start, other.start), end=max(self.end, other.end))

    @classmethod
    def empty(self):
        return Line(start=0, end=0)


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def area(self):
        if not self.valid:
            return 0.0
        return self.width * self.height

    @property
    def valid(self):
        return self.right >= self.left and self.bottom >= self.top

    @property
    def horizontal_span(self):
        return Line(start=self.left, end=self.right)

    @property
    def vertical_span(self):
        return Line(start=self.top, end=self.bottom)

    def intersect(self, other: Rect):
        if not self.valid or not other.valid:
            return self.empty()
        return Rect(
            left=max(self.left, other.left),
            right=min(self.right, other.right),
            top=max(self.top, other.top),
            bottom=min(self.bottom, other.bottom),
        )

    def union(self, other: Rect):
        if not self.valid or not other.valid:
            return self.empty()
        return Rect(
            left=min(self.left, other.left),
            right=max(self.right, other.right),
            top=min(self.top, other.top),
            bottom=max(self.bottom, other.bottom),
        )

    @classmethod
    def empty(cls):
        return Rect(left=0, right=0, top=0, bottom=0)


@dataclass(frozen=True)
class TextWord:
    text: str
    box: Rect


@dataclass
class TextPage:
    box: Rect
    words: list[TextWord]


class Timer:
    """
    Utility for recording and reporting duration of steps in an operation.
    """

    def __init__(self):
        self.checkpoints: list[tuple[str, float]] = []
        self.last_checkpoint = time.monotonic()

    def checkpoint(self, name: str):
        checkpoint = time.monotonic()
        delta = checkpoint - self.last_checkpoint
        self.last_checkpoint = checkpoint
        self.checkpoints.append((name, delta))

    def print(self, description: str):
        print(f"{description} timings:")
        for event, delta in self.checkpoints:
            print(f"  {event}: {delta}")


def run_tool(command: str, args: list[str]):
    """
    Run a command and return the exit status.
    """
    proc = subprocess.run([command, *args], capture_output=True)
    proc.check_returncode()


def iter_lines(text: str):
    for line in text.split("\n"):
        line = line.strip()
        if line:
            yield line


class PDFRenderer:
    """
    Render PDF pages to images and extract the text layer from a page.
    """

    def __init__(self, file_path, dpi=150):
        self.dpi = dpi
        self.file_path = file_path

    def count_pages(self) -> int:
        proc = subprocess.run(
            ["pdfinfo", self.file_path], capture_output=True, text=True
        )
        proc.check_returncode()
        info = {}
        for line in iter_lines(proc.stdout):
            field, value = line.split(":", 1)
            value = value.strip()
            info[field] = value
        return int(info["Pages"])

    def render_to_image(self, page: int) -> str:
        """
        Render a page from a PDF to a JPEG image.
        """
        out_base = "/tmp/pdf-render-image-result"
        run_tool(
            "pdftocairo",
            [
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(self.dpi),
                "-jpeg",
                "-singlefile",
                self.file_path,
                out_base,
            ],
        )
        return out_base + ".jpg"

    def render_to_text(self, page: int) -> TextPage:
        """
        Extract text with bounding boxes from a PDF page.
        """

        # - Generate text file path
        out_path = "/tmp/pdf-render-text-result.html"
        run_tool(
            "pdftotext",
            [
                "-f",
                str(page),
                "-l",
                str(page),
                "-r",
                str(self.dpi),
                "-bbox",
                self.file_path,
                out_path,
            ],
        )

        with open(out_path) as xml_file:
            xml_content = xml_file.read()

            # Strip out the XHTML namespace. This simplifies the `find` calls
            # below.
            xml_content = xml_content.replace(
                'xmlns="http://www.w3.org/1999/xhtml"', ""
            )

            # Strip out illegal characters. pdftotext doesn't escape eg.
            # control characters appearing in words.
            def is_legal_xml_char(char: str):
                code = ord(char)
                return code >= 32 or code == 0x09 or code == 0x0A or code == 0x0D

            xml_content = "".join([c for c in xml_content if is_legal_xml_char(c)])

            tree = ElementTree.fromstring(xml_content)

            page_el = tree.find(".//page")

            if not page_el:
                raise Exception('"page" element not found in output')

            # The `width` and `height` attributes do not change to reflect the
            # `-r` argument to `pdftotext`, but instead always reflect the
            # default 72 DPI, so we have to scale + ceil them.
            dpi_scale = self.dpi / 72.0
            width = ceil(float(page_el.attrib["width"]) * dpi_scale)
            height = ceil(float(page_el.attrib["height"]) * dpi_scale)

            text_box = Rect(left=0.0, top=0.0, right=width, bottom=height)
            text_words = []
            word_els = page_el.findall("word")
            for word_el in word_els:
                text_word = TextWord(
                    text=str(word_el.text),
                    box=Rect(
                        left=float(word_el.attrib["xMin"]),
                        top=float(word_el.attrib["yMin"]),
                        right=float(word_el.attrib["xMax"]),
                        bottom=float(word_el.attrib["yMax"]),
                    ),
                )
                text_words.append(text_word)
            text_page = TextPage(box=text_box, words=text_words)

        return text_page


class OCR:
    """
    OCR uses performs Optical Character Recognition on the pixels of an image.
    """

    def run_ocr(self, image: str) -> TextPage:
        """
        Extract text from an image using OCR.
        """
        img = Image.open(image)

        out_base = "/tmp/ocr-result"
        run_tool("tesseract", [image, out_base, "-l", "eng", "tsv"])
        out_path = out_base + ".tsv"
        with open(out_path) as tsv_file:
            reader = csv.DictReader(
                tsv_file,
                delimiter="\t",
                # The "text" field may contain unescaped quotes, so we should
                # not treat them as enclosing field values.
                quoting=csv.QUOTE_NONE,
            )
            text_page = TextPage(box=Rect(0, 0, img.width, img.height), words=[])
            for row in reader:
                confidence = float(row["conf"])
                if confidence < 0:
                    # TODO - Explain what these entries with negative `conf`
                    # values are.
                    continue

                text = row["text"].strip()
                if not text:
                    # Tesseract can produce large empty text "words", sometimes
                    # for images and figures.
                    continue

                left = float(row["left"])
                top = float(row["top"])
                width = float(row["width"])
                height = float(row["height"])

                word = TextWord(
                    text=row["text"],
                    box=Rect(
                        left=left, top=top, right=left + width, bottom=top + height
                    ),
                )
                text_page.words.append(word)
        return text_page


def create_text_page_mask(page: TextPage) -> np.ndarray:
    """
    Create a binary mask indicating regions of text in a page.
    """

    width = int(page.box.width)
    height = int(page.box.height)
    mask = np.zeros((height, width), dtype=np.bool_)

    for word in page.words:
        left = int(word.box.left)
        right = int(word.box.right)
        top = int(word.box.top)
        bottom = int(word.box.bottom)
        mask[top : bottom + 1, left : right + 1] = True

    return mask


def compare_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compare two binary masks.

    Returns a score indicating the similarity of the masks. A score of 1.0
    indicates corresponding elements in both masks all have identical values.
    A score of 0 indicates corresponding elements all have different values.
    """

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Masks have different sizes ({mask_a.shape}, {mask_b.shape})")

    diff_mask = mask_a ^ mask_b

    # TODO - Erode thin edges in the diff masks to reduce penalty for minor
    # mis-alignments.

    diff_count = len(diff_mask[diff_mask == True])
    width, height = mask_a.shape
    total_count = width * height

    mask_a_count = len(mask_a[mask_a == True])
    mask_b_count = len(mask_b[mask_b == True])
    max_count = max(mask_a_count, mask_b_count)

    # Might want to normalize score to take account of total amount of text on
    # the page. eg. `1 - non_matching_pixels / max(mask_a_text_pixels, mask_b_text_pixels)`

    # print(f"diff_count {diff_count} total_count {total_count}")

    return 1 - diff_count / max_count


def save_mask(mask: np.ndarray, path: str):
    """
    Save a binary mask to an image.
    """
    im = Image.fromarray(mask)
    im.save(path)


def intersection_over_union(a: Rect, b: Rect) -> float:
    return a.intersect(b).area / a.union(b).area


def compute_iou_metrics(
    pdf_text_page: TextPage, ocr_text_page: TextPage
) -> dict[str, float]:
    """
    Compare a PDF text layer against OCR output based on overlap between matching words.

    Returns a dict of metric name to value.
    """
    # Compute score by finding an alignment between OCR and PDF text layer word
    # boxes and computing an average match score over those.
    pdf_text_words = set(pdf_text_page.words)

    # Subset of OCR output which are considered when scoring the PDF text layer.
    #
    # We use a sample to reduce the costs of finding matches between the OCR
    # and text layer words below, at the cost of a small loss of accuracy.
    # We might want to improve this in future by biasing the sample to ensure
    # it represents all areas of the page (ie. over-weight words from under-dense
    # regions) or filter out very large or very small words which may be
    # outliers in the OCR or PDF output.
    sample_size = min(300, len(ocr_text_page.words))
    ocr_words_subset = random.sample(ocr_text_page.words, sample_size)

    # Find an alignment between words in the OCR output and words in the PDF
    # text layer.
    # nb. This loop is currently O(ocr_words * pdf_words)
    iou_threshold = 0.1
    word_matches = []
    for ocr_word in ocr_words_subset:
        best_pdf_word = None
        best_iou = 0.0
        for pdf_word in pdf_text_words:
            if (
                pdf_word.box.top > ocr_word.box.bottom
                or pdf_word.box.bottom < ocr_word.box.top
            ):
                # Quickly skip words that have no vertical overlap
                continue

            iou = intersection_over_union(ocr_word.box, pdf_word.box)
            if iou > iou_threshold and iou > best_iou:
                best_pdf_word = pdf_word
                best_iou = iou

        if not best_pdf_word:
            continue

        word_matches.append((ocr_word, best_pdf_word))
        pdf_text_words.remove(best_pdf_word)

    # Compute a localization score over matched words
    iou_sum = 0
    iou_sum_x = 0
    iou_sum_y = 0

    for ocr_word, pdf_word in word_matches:
        ocr_word_x_span = ocr_word.box.horizontal_span
        pdf_word_x_span = pdf_word.box.horizontal_span

        ocr_word_y_span = ocr_word.box.vertical_span
        pdf_word_y_span = pdf_word.box.vertical_span

        iou_x = (
            ocr_word_x_span.intersect(pdf_word_x_span).length
            / ocr_word_x_span.union(pdf_word_x_span).length
        )
        iou_sum_x += iou_x

        iou_y = (
            ocr_word_y_span.intersect(pdf_word_y_span).length
            / ocr_word_y_span.union(pdf_word_y_span).length
        )
        iou_sum_y += iou_y
        iou_sum += iou_x * iou_y

    # Calculate weighted IoU scores. For the purposes of text selection,
    # horizontal alignment is more important than vertical alignment, at least
    # for languages that are written in horizontal lines. Also vertical
    # alignment is more likely to differ between the OCR and the text layer as
    # it can be affected by differences such as whether each box includes
    # ascenders and descenders.

    mean_iou = iou_sum / len(ocr_words_subset)
    mean_iou_x = iou_sum_x / len(ocr_words_subset)
    mean_iou_y = iou_sum_y / len(ocr_words_subset)
    mean_iou_weighted = 0.7 * mean_iou_x + 0.3 * mean_iou_y

    return {
        "mean_iou": mean_iou,
        "mean_iou_x": mean_iou_x,
        "mean_iou_y": mean_iou_y,
        "mean_iou_weighted": mean_iou_weighted,
    }


def compute_mask_metric(
    pdf_text_page: TextPage, ocr_text_page: TextPage, debug=False
) -> dict[str, float]:
    """
    Compare a PDF text layer against OCR output using binary masks of the text
    and non-text regions of a page.

    Returns a dict of metric name to value.
    """

    text_page_mask = create_text_page_mask(pdf_text_page)
    ocr_text_page_mask = create_text_page_mask(ocr_text_page)
    match_score = compare_masks(text_page_mask, ocr_text_page_mask)

    if debug:
        save_mask(text_page_mask, "debug/pdf_text_mask.jpg")
        save_mask(ocr_text_page_mask, "debug/ocr_text_mask.jpg")
        save_mask(text_page_mask ^ ocr_text_page_mask, "debug/xor_mask.jpg")

    return {"mask_overlap": match_score}


def process_page(
    pdf_renderer: PDFRenderer,
    page: int,
    debug=False,
    mask_metric=True,
    iou_metric=True,
    timing=False,
):
    """
    Extract the PDF text layer from a page and compare it against OCR output.
    """
    t = Timer()
    image = pdf_renderer.render_to_image(page=page)
    t.checkpoint("render_to_image")
    pdf_text_page = pdf_renderer.render_to_text(page=page)
    t.checkpoint("render_to_text")

    ocr = OCR()
    ocr_text_page = ocr.run_ocr(image)
    t.checkpoint("ocr")

    def print_metrics(name: str, metrics: dict[str, float]):
        formatted_metrics = [f"{key}: {value:.2f}" for key, value in metrics.items()]
        print(f"Page {page} {name}:", ", ".join(formatted_metrics))

    if mask_metric:
        mask_metrics = compute_mask_metric(pdf_text_page, ocr_text_page, debug=debug)
        print_metrics("mask", mask_metrics)
        t.checkpoint("mask_metrics")

    if iou_metric:
        iou_metrics = compute_iou_metrics(pdf_text_page, ocr_text_page)
        print_metrics("IoU", iou_metrics)
        t.checkpoint("iou_metrics")

    if timing:
        t.print("process_page")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="PDF file to check")
    parser.add_argument(
        "--first-page",
        type=int,
        dest="first_page",
        help="Number of first page to process",
    )
    parser.add_argument(
        "--mask-metrics",
        action="store_true",
        dest="mask_metrics",
    )
    parser.add_argument(
        "--last-page", type=int, dest="last_page", help="Number of last page to process"
    )
    parser.add_argument(
        "--debug", action="store_true", dest="debug", help="Store debug outputs"
    )
    parser.add_argument(
        "--timing", action="store_true", dest="timing", help="Print timing info"
    )
    args = parser.parse_args()

    if args.debug:
        os.makedirs("debug", exist_ok=True)

    dpi = 150
    pdf_renderer = PDFRenderer(file_path=args.pdf_file, dpi=dpi)

    page_count = pdf_renderer.count_pages()
    first_page = args.first_page or 1
    last_page = max(first_page, args.last_page or page_count)

    print(f"Checking text pages {first_page} to {last_page}")

    for page in range(first_page, last_page + 1):
        try:
            process_page(
                pdf_renderer,
                page=page,
                debug=args.debug,
                mask_metric=args.mask_metrics,
                iou_metric=True,
                timing=args.timing,
            )
        except Exception as e:
            print(f"Error processing page {page}", repr(e), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
