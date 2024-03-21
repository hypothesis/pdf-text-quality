# See https://stackoverflow.com/a/33533505/434243
from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
import asyncio
import csv
from dataclasses import dataclass
import dataclasses
import json
from io import BytesIO, StringIO
from math import ceil
import os
import random
import secrets
import subprocess
import sys
import time
from typing import cast, Iterator, Optional, Protocol, Union
import xml.etree.ElementTree as ElementTree

import numpy as np
from numpy import typing as npt
from PIL import Image
from PIL import ImageDraw


@dataclass(frozen=True)
class Line:
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def valid(self) -> bool:
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
    def empty(cls) -> Line:
        return Line(start=0, end=0)


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        if not self.valid:
            return 0.0
        return self.width * self.height

    @property
    def valid(self) -> bool:
        return self.right >= self.left and self.bottom >= self.top

    @property
    def horizontal_span(self) -> Line:
        return Line(start=self.left, end=self.right)

    @property
    def vertical_span(self) -> Line:
        return Line(start=self.top, end=self.bottom)

    def intersect(self, other: Rect) -> Rect:
        if not self.valid or not other.valid:
            return self.empty()
        return Rect(
            left=max(self.left, other.left),
            right=min(self.right, other.right),
            top=max(self.top, other.top),
            bottom=min(self.bottom, other.bottom),
        )

    def union(self, other: Rect) -> Rect:
        if not self.valid or not other.valid:
            return self.empty()
        return Rect(
            left=min(self.left, other.left),
            right=max(self.right, other.right),
            top=min(self.top, other.top),
            bottom=max(self.bottom, other.bottom),
        )

    @classmethod
    def empty(cls) -> Rect:
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

    def __init__(self) -> None:
        self.checkpoints: list[tuple[str, float]] = []
        self.last_checkpoint = time.monotonic()

    def checkpoint(self, name: str) -> None:
        checkpoint = time.monotonic()
        delta = checkpoint - self.last_checkpoint
        self.last_checkpoint = checkpoint
        self.checkpoints.append((name, delta))

    def print(self, description: str) -> None:
        print(f"{description} timings:")
        for event, delta in self.checkpoints:
            print(f"  {event}: {delta}")


async def run_command(
    command: str, args: list[str], stdin: Optional[bytes] = None
) -> bytes:
    """
    Run a command and return its stdout output.

    Raises if the command exits with a non-zero status.
    """
    stdin_mode = subprocess.PIPE if stdin is not None else None
    proc = await asyncio.create_subprocess_exec(
        command, *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=stdin_mode
    )
    stdout, stderr = await proc.communicate(stdin)
    if proc.returncode != 0:
        raise Exception(f"Command {command} failed (status {proc.returncode})")
    return stdout


def iter_lines(text: str) -> Iterator[str]:
    for line in text.split("\n"):
        line = line.strip()
        if line:
            yield line


class PDFRenderer:
    """
    Render PDF pages to images and extract the text layer from a page.
    """

    def __init__(self, file_path: str, dpi: int = 150) -> None:
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

    async def render_to_image(self, page: int) -> bytes:
        """
        Render a page from a PDF to a JPEG image.
        """
        img_data = await run_command(
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
                # Write to stdout
                "-",
            ],
        )
        return img_data

    async def render_to_text(self, page: int) -> TextPage:
        """
        Extract text with bounding boxes from a PDF page.
        """

        xml_content = (
            await run_command(
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
                    # Write to stdout
                    "-",
                ],
            )
        ).decode()

        # Strip out the XHTML namespace. This simplifies the `find` calls
        # below.
        xml_content = xml_content.replace('xmlns="http://www.w3.org/1999/xhtml"', "")

        # Strip out illegal characters. pdftotext doesn't escape eg.
        # control characters appearing in words.
        def is_legal_xml_char(char: str) -> bool:
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

    async def run_ocr(self, image: bytes) -> TextPage:
        """
        Extract text from an image using OCR.
        """
        # We might want to set `OMP_THREAD_LIMIT=1` here to improve throughput
        # when Tesseract is being run for multiple pages in parallel.
        # See https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#environment-variables.
        tsv_content = (
            await run_command(
                "tesseract", ["stdin", "stdout", "-l", "eng", "tsv"], stdin=image
            )
        ).decode()

        with (
            StringIO(tsv_content) as tsv_file,
            BytesIO(image) as img_io,
            Image.open(img_io) as img,
        ):
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


Mask = npt.NDArray[np.bool_]


def create_text_page_mask(page: TextPage) -> Mask:
    """
    Create a binary mask indicating regions of text in a page.
    """

    width = int(page.box.width)
    height = int(page.box.height)
    mask: Mask = np.zeros((height, width), dtype=np.bool_)

    for word in page.words:
        left = int(word.box.left)
        right = int(word.box.right)
        top = int(word.box.top)
        bottom = int(word.box.bottom)
        mask[top : bottom + 1, left : right + 1] = True

    return mask


def compare_masks(mask_a: Mask, mask_b: Mask) -> float:
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


def save_mask(mask: Mask, path: str) -> None:
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
    pdf_text_words = set(pdf_text_page.words)

    # Subset of OCR output which are considered when scoring the PDF text layer.
    #
    # We use a sample to reduce the costs of finding matches between the OCR
    # and text layer words below, at the cost of a small loss of accuracy.
    # We might want to improve this in future by biasing the sample to ensure
    # it represents all areas of the page (ie. over-weight words from under-dense
    # regions) or filter out very large or very small words which may be
    # outliers in the OCR or PDF output.
    sample_size = min(1000, len(ocr_text_page.words))
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
    iou_sum = 0.0
    iou_sum_x = 0.0
    iou_sum_y = 0.0

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

    epsilon = 0.00001  # Prevent divide-by-zero
    sampled_word_count = len(ocr_words_subset) + epsilon
    mean_iou = iou_sum / sampled_word_count
    mean_iou_x = iou_sum_x / sampled_word_count
    mean_iou_y = iou_sum_y / sampled_word_count
    mean_iou_weighted = 0.7 * mean_iou_x + 0.3 * mean_iou_y

    return {
        "iou": mean_iou,
        "iou_x": mean_iou_x,
        "iou_y": mean_iou_y,
        "iou_weighted": mean_iou_weighted,
    }


def compute_mask_metric(
    pdf_text_page: TextPage, ocr_text_page: TextPage, debug: bool = False
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


def draw_boxes(im: Image.Image, page: TextPage, color: str, width: int) -> None:
    draw = ImageDraw.Draw(im)

    for word in page.words:
        box = word.box
        draw.rectangle(
            (box.left, box.top, box.right, box.bottom),
            fill=None,
            outline=color,
            width=width,
        )


async def process_page(
    pdf_renderer: PDFRenderer,
    page: int,
    debug: bool = False,
    mask_metric: bool = True,
    iou_metric: bool = True,
    timing: bool = False,
    save_prefix: Optional[str] = None,
) -> dict[str, float]:
    """
    Extract the PDF text layer from a page and compare it against OCR output.

    Returns a dict of comparison metrics.
    """
    t = Timer()
    image_data = await pdf_renderer.render_to_image(page=page)
    t.checkpoint("render_to_image")
    pdf_text_page = await pdf_renderer.render_to_text(page=page)
    t.checkpoint("render_to_text")

    ocr = OCR()
    ocr_text_page = await ocr.run_ocr(image_data)
    t.checkpoint("ocr")

    if save_prefix:
        with (
            open(f"{save_prefix}-image.jpg", "wb") as img_file,
            open(f"{save_prefix}-pdf-text.json", "w") as pdf_text_file,
            open(f"{save_prefix}-ocr-text.json", "w") as ocr_text_file,
        ):
            img_file.write(image_data)
            json.dump(dataclasses.asdict(pdf_text_page), pdf_text_file)
            json.dump(dataclasses.asdict(ocr_text_page), ocr_text_file)

    if debug:
        with BytesIO(image_data) as img_io, Image.open(image_data) as im:
            draw_boxes(im, ocr_text_page, color="rgb(0, 180, 0)", width=2)
            draw_boxes(im, pdf_text_page, color="rgb(255,0,0)", width=3)
            im.save("debug/boxes.jpg")

    metrics: dict[str, float] = {}

    if mask_metric:
        mask_metrics = compute_mask_metric(pdf_text_page, ocr_text_page, debug=debug)
        metrics |= mask_metrics
        t.checkpoint("mask_metrics")

    if iou_metric:
        iou_metrics = compute_iou_metrics(pdf_text_page, ocr_text_page)
        metrics |= iou_metrics
        t.checkpoint("iou_metrics")

    if timing:
        t.print("process_page")

    return metrics


class OutputWriter(ABC):
    @abstractmethod
    def write_row(self, fields: dict[str, str]) -> None:
        """Write a set of fields to the output."""


class TextOutputWriter(OutputWriter):
    """
    Writer that outputs metrics as a comma-separated list of "key: value" pairs.
    """

    def write_row(self, fields: dict[str, str]) -> None:
        formatted_fields = [f"{key}: {value}" for key, value in fields.items()]
        print(", ".join(formatted_fields))


class CSVOutputWriter(OutputWriter):
    """
    Writer that outputs metrics in CSV format.
    """

    csv_writer: Optional[csv.DictWriter[str]]

    def __init__(self) -> None:
        self.csv_writer = None

    def write_row(self, fields: dict[str, str]) -> None:
        if not self.csv_writer:
            csv_fields = list(fields.keys())
            self.csv_writer = csv.DictWriter(sys.stdout, csv_fields)
            self.csv_writer.writeheader()
        self.csv_writer.writerow(fields)


def windows(start: int, end: int, window_size: int) -> Iterator[tuple[int, int]]:
    """
    Iterate over offsets for windows in the range [start, end].

    For example `windows(0, 6, 2)` yields `(0, 1)`, `(2, 3)`, `(4, 5)`, `(6, 6)`.

    :param start: Start of first window
    :param end: End (inclusive) of last window
    :param window_size: Maximum size of each window.
    """
    for win_start in range(start, end + 1, window_size):
        yield (win_start, min(win_start + window_size - 1, end))


def process_file(
    pdf_file: str,
    out_writer: OutputWriter,
    debug: bool = False,
    first_page: int = 1,
    iou_metrics: bool = False,
    last_page: Optional[int] = None,
    mask_metrics: bool = False,
    print_timings: bool = False,
    save_dir: Optional[str] = None,
) -> None:
    """
    Check the PDF text layers for pages in `pdf_file`.

    Extract the text layer and run OCR over pages in `pdf_file` and compare
    them. Metrics about the quality of the match are written to `out_writer`.

    If `save_dir` is set, intermediate outputs including the rendered page images
    and PDF/OCR text pages are saved to that directory.
    """

    # Resolution to render PDF at for OCR. Larger values result in slower
    # but more accurate processing. The resolution needs to be high enough that
    # the result is clearly readable. Depending on the PDF, there will be some
    # threshold below which OCR accuracy drops off rapidly.
    dpi = 150

    pdf_renderer = PDFRenderer(file_path=pdf_file, dpi=dpi)

    try:
        page_count = pdf_renderer.count_pages()
    except Exception as e:
        print(f"Error counting pages in {pdf_file}", repr(e))
        return

    if last_page is None:
        last_page = page_count

    first_page = min(max(first_page, 1), page_count)
    last_page = min(max(last_page, first_page), page_count)

    file_basename = os.path.basename(pdf_file)
    if save_dir:
        save_prefix = f"{save_dir}/{file_basename}"
        os.makedirs(save_prefix, exist_ok=True)
    else:
        save_prefix = None

    async def process_pages(first_page: int, last_page: int) -> None:
        print(
            f"Checking pages {first_page} to {last_page} in {file_basename}",
            file=sys.stderr,
        )

        # Run OCR and text layer extraction on multiple pages concurrently.
        page_tasks = []
        page_indexes = [i for i in range(first_page, last_page + 1)]
        for page in page_indexes:
            page_save_prefix = f"{save_prefix}/page-{page}" if save_prefix else None
            task = process_page(
                pdf_renderer,
                page=page,
                debug=debug,
                mask_metric=mask_metrics,
                iou_metric=iou_metrics,
                timing=print_timings,
                save_prefix=page_save_prefix,
            )
            page_tasks.append(task)
        page_metrics = await asyncio.gather(*page_tasks)

        # Process the results for all pages in order.
        for i, page in enumerate(page_indexes):
            try:
                metrics = page_metrics[i]
                row = {"file": pdf_file, "page": str(page)}
                formatted_metrics = {
                    key: f"{value:.2f}" for key, value in metrics.items()
                }
                row.update(formatted_metrics)
                out_writer.write_row(row)

            except Exception as e:
                print(
                    f"Error processing page {page} in {pdf_file}",
                    repr(e),
                    file=sys.stderr,
                )

    # Process up to 10 pages at a time. The window size aims to maximize
    # throughput while not having an unbounded number of OCR / PDF rendering
    # processes running at once, and providing regular progress updates.
    for window_start, window_end in windows(first_page, last_page, window_size=10):
        asyncio.run(process_pages(window_start, window_end))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", nargs="+", help="PDF file to check")
    parser.add_argument(
        "--iou-metrics",
        action=argparse.BooleanOptionalAction,
        dest="iou_metrics",
        default=True,
    )
    parser.add_argument(
        "--mask-metrics",
        action=argparse.BooleanOptionalAction,
        dest="mask_metrics",
        default=False,
    )
    parser.add_argument(
        "--csv",
        help="Output metrics in CSV format (one line per page)",
        action="store_true",
        dest="csv",
    )

    parser.add_argument(
        "--page",
        type=int,
        dest="page",
        help="Number of first and last page to process",
    )
    parser.add_argument(
        "--first-page",
        type=int,
        dest="first_page",
        help="Number of first page to process",
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
    parser.add_argument(
        "--save-dir",
        type=str,
        dest="save_dir",
        help="Save rendered outputs to specified dir",
    )

    args = parser.parse_args()

    # Make this script work with `tee`. See https://stackoverflow.com/a/68341808/434243.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore

    if args.debug:
        os.makedirs("debug", exist_ok=True)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    file_count = len(args.pdf_file)

    if args.csv:
        out_writer: OutputWriter = CSVOutputWriter()
    else:
        out_writer = TextOutputWriter()

    for file_index, pdf_file in enumerate(args.pdf_file):
        if args.page:
            first_page = args.page
            last_page = args.page
        else:
            first_page = args.first_page or 1
            last_page = args.last_page or None

        print(
            f"Checking {pdf_file} (file {file_index+1}/{file_count})",
            file=sys.stderr,
        )

        process_file(
            pdf_file,
            out_writer,
            first_page=first_page,
            last_page=last_page,
            debug=args.debug,
            mask_metrics=args.mask_metrics,
            iou_metrics=args.iou_metrics,
            print_timings=args.timing,
            save_dir=args.save_dir,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
