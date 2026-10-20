"""Тесты сверки разметки .xml/.txt в EDA."""

import pytest
from PIL import Image

from explore_data import check_annotations

XML = """<annotation><size><width>100</width><height>50</height></size>
<object><name>{name}</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>20</xmax><ymax>45</ymax></bndbox></object>
</annotation>"""


@pytest.fixture
def img(tmp_path):
    Image.new("RGB", (100, 50)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("16 0.15 0.5 0.1 0.8\n")
    return tmp_path / "a.png"


def test_consistent_annotation_has_no_issues(img):
    img.with_suffix(".xml").write_text(XML.format(name="J"))

    assert check_annotations(img) == set()


def test_class_mismatch_and_box_outside_image(img):
    img.with_suffix(".xml").write_text(XML.format(name="L"))
    img.with_suffix(".txt").write_text("16 0.15 0.5 0.1 1.2\n")

    assert check_annotations(img) == {
        "символ в .xml не совпадает с классом в .txt",
        "рамки в .xml и .txt не совпадают",
        "рамка выходит за край изображения",
    }
