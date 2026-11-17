import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties

from library.config import FILE_PATHS
from library.data_io import load_ini_data_from_csv
from scripts.create_manuscript_spatial_comparisons import (
    COUNTRY_NAMES_PATH,
    _keep_labels_left_of_colorbar,
    _load_country_display_names,
)
from scripts.create_section_6_5_visual_prototype import _draw_error_map, _draw_overlay


def test_country_names_cover_formal_model_nodes():
    _, vertices, _, _, _ = load_ini_data_from_csv(FILE_PATHS)
    names = _load_country_display_names(COUNTRY_NAMES_PATH, vertices)
    assert len(names) == 35
    assert names["\u90fd\u8b77\u6cbb/\u70cf\u58d8"] == "Wulei"
    assert names["\u8eca\u5e2b\u524d"] == "Nearer Jushi"
    assert names["\u9ad8\u660c\u58c1"] == "fortress of Gaochang"


@pytest.mark.parametrize(
    ("content", "message"),
    [("A,Alpha\n", "missing"), ("A,Alpha\nB,Beta\nC,Gamma\n", "extra"),
     ("A,Alpha\nB,Beta\nA,Another\n", "duplicate"), ("A,Alpha\nB,\n", "Empty")],
)
def test_invalid_display_names_fail_instead_of_silently_falling_back(tmp_path, content, message):
    path = tmp_path / "names.csv"
    path.write_text("model_name,english_name\n" + content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_country_display_names(path, ["A", "B"])


def test_translation_changes_annotations_but_not_overlay_or_error_map_geometry():
    keys = ["\u8eca\u5e2b\u524d", "\u9ad8\u660c\u58c1"]
    points = np.array([[1., 2.], [5., 7.]])
    dni = dict(zip(keys, range(2)))
    targets = {keys[0]: np.array([2., 3.])}
    names = dict(zip(keys, ["Nearer Jushi", "fortress of Gaochang"]))
    norm, cmap, font = Normalize(0, 3), plt.get_cmap("viridis"), FontProperties()
    fig, axes = plt.subplots(2, 2)
    try:
        for col, labels in enumerate((None, names)):
            _draw_overlay(axes[0, col], points, targets, {}, keys, dni, [keys[1]], [keys[0]],
                          norm, cmap, font, draw_title=False, display_labels=labels)
            _draw_error_map(axes[1, col], points, [(0, 1, 0.5)], {0}, norm, cmap, keys, font,
                            display_labels=labels)
        for original, translated in axes:
            assert [ann.get_text() for ann in original.texts] == keys
            assert [ann.get_text() for ann in translated.texts] == list(names.values())
            assert [ann.xy for ann in original.texts] == [ann.xy for ann in translated.texts]
            for old, new in zip(original.lines, translated.lines, strict=True):
                np.testing.assert_array_equal(old.get_xydata(), new.get_xydata())
                assert old.get_color() == new.get_color()
            for old, new in zip(original.collections, translated.collections, strict=True):
                np.testing.assert_array_equal(old.get_offsets(), new.get_offsets())
                np.testing.assert_array_equal(old.get_facecolors(), new.get_facecolors())
        assert all(ann.get_clip_on() for ann in axes[0, 0].texts)
        assert not any(ann.get_clip_on() for ann in axes[0, 1].texts)
    finally:
        plt.close(fig)


def test_long_display_name_stays_left_of_colorbar_without_moving_node():
    fig, ax = plt.subplots()
    try:
        bar = fig.add_axes([.91, .1, .02, .8])
        annotation = ax.annotate("the state of the Chief of the Further Town of Jushi",
                                 xy=(.9, .5), xytext=(5, 5), textcoords="offset points")
        _keep_labels_left_of_colorbar(fig, [annotation], bar)
        fig.canvas.draw()
        assert annotation.xy == (.9, .5)
        assert annotation.get_window_extent(fig.canvas.get_renderer()).x1 < bar.bbox.x0
    finally:
        plt.close(fig)
