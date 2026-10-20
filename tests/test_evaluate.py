"""Тесты скрипта оценки (без модели: предсказания подставляются вручную)."""

import torch
from PIL import Image

import evaluate
from src.data.classes import CLASSES
from src.evaluation.postprocess import sort_left_to_right
from src.models.fcos import FCOSDetector


def write_plate(labels_dir, stem, class_ids):
    lines = [f"{class_id} {0.1 * (i + 1):.2f} 0.5 0.05 0.6" for i, class_id in enumerate(class_ids)]
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def fake_predict(predictions):
    """predict(img_path), возвращающий заранее заданные class_id слева направо."""

    def predict(img_path):
        return [([10 * i, 0, 10 * i + 8, 50], class_id, 0.9) for i, class_id in enumerate(predictions[img_path.stem])]

    return predict


def test_evaluate_compares_plates_and_marks_seen(tmp_path):
    write_plate(tmp_path, "a", [5, 7, 3, 2, 18, 25, 12])
    write_plate(tmp_path, "b", [1, 2, 3, 4, 10, 10, 10])
    predict = fake_predict({"a": [5, 7, 3, 2, 18, 25, 12], "b": [1, 2, 3, 4, 10, 10]})

    rows = evaluate.evaluate([tmp_path / "a.jpeg", tmp_path / "b.png"], tmp_path, predict, train_plates={"5732LXD"})

    assert rows == [
        {"name": "a.jpeg", "gt": "5732LXD", "pred": "5732LXD", "seen": True},
        {"name": "b.png", "gt": "1234AAA", "pred": "1234AA", "seen": False},
    ]


def test_summarize_prints_exact_match_interval_and_cer(capsys):
    rows = [{"gt": "1234AAA", "pred": "1234AAA"}, {"gt": "1234AAA", "pred": "1234AA"}]

    evaluate.summarize(rows, "test")

    assert capsys.readouterr().out == (
        "test: Exact Match 50.0% (1/2), 95% ДИ [9.5%; 90.5%], CER 7.1%, по номерам 50.0% (1 номеров)\n"
    )


def test_summarize_weights_plates_equally(capsys):
    # Частый номер распознан на всех 3 кадрах, редкий - нет
    rows = [{"gt": "1111AAA", "pred": "1111AAA"}] * 3 + [{"gt": "2222BBB", "pred": "2222BB"}]

    evaluate.summarize(rows, "val")

    out = capsys.readouterr().out
    assert "Exact Match 75.0% (3/4)" in out
    assert "по номерам 50.0% (2 номеров)" in out


def test_fcos_predictor_returns_boxes_classes_and_scores(tmp_path):
    torch.manual_seed(0)
    model = FCOSDetector(num_classes=len(CLASSES), backbone_pretrained=False).eval()
    Image.new('RGB', (134, 89), 'white').save(tmp_path / 'a.png')

    preds = evaluate.make_fcos_predictor(model, conf=0.001)(tmp_path / 'a.png')

    assert preds
    for box, class_id, conf in preds:
        assert len(box) == 4
        assert 0 <= class_id < len(CLASSES)
        assert conf > 0.001


def test_evaluate_without_filter_keeps_all_predictions(tmp_path):
    write_plate(tmp_path, "a", [1, 2, 3, 4, 10, 10, 10])
    predict = fake_predict({"a": [1, 2, 3, 4, 10, 10, 10, 10]})

    rows = evaluate.evaluate([tmp_path / "a.jpeg"], tmp_path, predict, set(), postprocess=sort_left_to_right)

    assert rows[0]["pred"] == "1234AAAA"
