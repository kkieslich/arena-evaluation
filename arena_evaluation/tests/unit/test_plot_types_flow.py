"""Flow plots on hand-made maps of one room: an edit against its base world, drawn as maps, as scores, and as the bundled `flow` report."""

from __future__ import annotations

import base64
import json
import pathlib
import re

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")
pytest.importorskip("shapely")
pytest.importorskip("plotly")

from arena_evaluation.presentation.plot_types.flow import JSD_FLOOR, FlowJsdRenderer, FlowMapRenderer
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.processing.flow import FLOW_CELLS_FILE, FlowGrid
from arena_evaluation.processing.flow.benchmark import FLOW_CELL_SCHEMA
from arena_evaluation.storage.schemas import PlotSpec

ROOM = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
GRID = FlowGrid.from_polygons([ROOM])
BASE = "flowplot_room"
MOVE = f"{BASE}__move_fountain__ab"
INERT = f"{BASE}__insert_fountain_inert__cd"
PNG = b"\x89PNG\r\n\x1a\n"
STAND, EAST = GRID.headings, 4
WORLD_YAML = f"""format: arena-world/2
levels:
- id: '0'
  zones:
  - name: room
    polygon: {[list(p) for p in ROOM]}
"""


def _counts(*bins: tuple[int, int, int, float]) -> np.ndarray:
    """A map of GRID holding `seconds` in each (row, column, bin)."""
    counts = GRID.empty()
    for i, j, b, seconds in bins:
        counts[i, j, b] = seconds
    return counts


def _run(world: str, seed: int, counts: np.ndarray, planner: str = "peds_only") -> pl.DataFrame:
    return GRID.cells(counts).with_columns(planner=pl.lit(planner), stage=pl.lit(f"{world}__s{seed}"), flow_world=pl.lit(world), flow_base=pl.lit(BASE), flow_seed=pl.lit(seed, pl.Int64)).select(list(FLOW_CELL_SCHEMA))


# A corridor walked east in row 1 every run. The fountain queue stands in cell (0, 0); the move takes it to (3, 7).
# The inert copy changes nothing. The move's seed 6 has no base run and stays out of every map.
WALK = [(1, j, EAST, 1.0) for j in range(8)]
CELLS = pl.concat(
    [
        _run(BASE, 1, _counts((0, 0, STAND, 8.0), *WALK)),
        _run(BASE, 2, _counts((0, 0, STAND, 4.0), *WALK)),
        _run(MOVE, 1, _counts((3, 7, STAND, 8.0), *WALK)),
        _run(MOVE, 2, _counts((3, 7, STAND, 4.0), *WALK)),
        _run(MOVE, 6, _counts((2, 2, STAND, 50.0))),
        _run(INERT, 1, _counts((0, 0, STAND, 8.0), *WALK)),
    ]
)


def _snapshot(run_dir: pathlib.Path) -> pathlib.Path:
    """`run_dir` holding the base world's snapshot, as a benchmark run does."""
    world = run_dir / "worlds" / BASE
    world.mkdir(parents=True)
    (world / "world.yaml").write_text(WORLD_YAML)
    return run_dir


def _figure(html: str) -> tuple[list[dict], dict]:
    """(traces, layout) of a plotly `to_html` fragment."""
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html)
    assert match
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(html[match.end() :])
    layout, _ = decoder.raw_decode(html[match.end() + end :].lstrip()[1:].lstrip())
    return data, layout


def _array(value: list | dict) -> np.ndarray:
    """A trace array as plotly wrote it: a list, or typed binary data."""
    if isinstance(value, dict):
        flat = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        return flat.reshape([int(n) for n in str(value["shape"]).split(",")]) if "shape" in value else flat
    return np.array(value, dtype=float)


def _map_renderer(run_dir: pathlib.Path | None) -> FlowMapRenderer:
    renderer = FlowMapRenderer(PlotSpec(id="flow_map", type="flow_map", title="Flow map", data_key="ped_seconds"))
    renderer.run_dir = run_dir
    return renderer


def test_each_edit_maps_its_base_world_itself_and_the_difference_largest_change_first(tmp_path: pathlib.Path) -> None:
    move, inert = _map_renderer(_snapshot(tmp_path)).render_plotly(CELLS)
    traces, layout = _figure(move)
    assert "move_fountain__ab against flowplot_room, mean of seeds 1, 2" in layout["title"]["text"]
    base, edit, diff = (_array(t["z"]) for t in traces if t["type"] == "heatmap")
    # a 0.5 m cell is 0.25 m²; seeds 1 and 2 averaged: (8 + 4) / 2 = 6 s in the queue cell
    assert base[0, 0] == pytest.approx(24.0) and np.isnan(base[3, 7])
    assert edit[3, 7] == pytest.approx(24.0) and np.isnan(edit[0, 0]) and np.isnan(edit[2, 2])
    assert diff[3, 7] == pytest.approx(24.0) and diff[0, 0] == pytest.approx(-24.0) and np.isnan(diff[1, 3])
    assert "against flowplot_room, mean of seeds 1" in _figure(inert)[1]["title"]["text"]
    assert np.isnan(_array([t for t in _figure(inert)[0] if t["type"] == "heatmap"][2]["z"])).all()


def test_the_walked_cells_carry_arrows_along_their_walking_direction(tmp_path: pathlib.Path) -> None:
    traces, _ = _figure(_map_renderer(_snapshot(tmp_path)).render_plotly(CELLS)[0])
    arrows = [t for t in traces if t["type"] == "scatter" and t["mode"] == "markers"]
    assert len(arrows) == 2
    # east is the 45° bin from 0°, its centre 22.5°; plotly turns markers clockwise from north
    assert set(_array(arrows[0]["y"]).tolist()) == {0.75} and _array(arrows[0]["marker"]["angle"]) == pytest.approx(90 - 22.5)


def test_every_edit_shares_one_scale_so_a_small_change_looks_small(tmp_path: pathlib.Path) -> None:
    layouts = [_figure(html)[1] for html in _map_renderer(_snapshot(tmp_path)).render_plotly(CELLS)]
    assert layouts[0]["coloraxis"]["cmax"] == layouts[1]["coloraxis"]["cmax"]
    assert layouts[0]["coloraxis2"]["cmax"] == layouts[1]["coloraxis2"]["cmax"] == -layouts[1]["coloraxis2"]["cmin"] > 0


def test_walls_come_from_the_base_world_and_are_left_out_without_it(tmp_path: pathlib.Path) -> None:
    traces, _ = _figure(_map_renderer(_snapshot(tmp_path)).render_plotly(CELLS)[0])
    walls = [t for t in traces if t["type"] == "scatter" and t["mode"] == "lines"]
    assert len(walls) == 3 and {0.0, 4.0} <= {x for x in walls[0]["x"] if x is not None}


def test_each_edit_is_one_png(tmp_path: pathlib.Path) -> None:
    out = _snapshot(tmp_path) / "plots" / "flow_map.png"
    out.parent.mkdir()
    _map_renderer(tmp_path).render_seaborn(CELLS, out)
    pngs = sorted(p.name for p in out.parent.iterdir())
    assert pngs == [f"flow_map_{INERT}.png", f"flow_map_{MOVE}.png"]
    assert all((out.parent / name).read_bytes().startswith(PNG) for name in pngs)


def test_each_contestant_gets_its_own_maps(tmp_path: pathlib.Path) -> None:
    other = CELLS.with_columns(planner=pl.lit("orca"))
    htmls = _map_renderer(_snapshot(tmp_path)).render_plotly(pl.concat([CELLS, other]))
    titles = [_figure(html)[1]["title"]["text"] for html in htmls]
    assert len(titles) == 4 and sum("(orca)" in t for t in titles) == 2


def test_base_worlds_alone_draw_nothing(tmp_path: pathlib.Path) -> None:
    renderer = _map_renderer(_snapshot(tmp_path))
    out = tmp_path / "flow_map.png"
    assert renderer.render_plotly(CELLS.filter(pl.col("flow_world") == BASE)) is None
    renderer.render_seaborn(CELLS.filter(pl.col("flow_world") == BASE), out)
    assert not list(tmp_path.glob("*.png"))


SCORES = pl.DataFrame(
    {
        "episode": [0, 1, 2, 3, 4, 5],
        "planner": ["peds_only"] * 6,
        "stage": [f"{BASE}__s1", f"{BASE}__s2", f"{MOVE}__s1", f"{MOVE}__s2", f"{INERT}__s1", f"{INERT}__s2"],
        "status": ["evaluated"] * 6,
        "is_reference": [False] * 6,
        "flow_world": [BASE, BASE, MOVE, MOVE, INERT, INERT],
        "flow_base": [BASE] * 6,
        "flow_seed": [1, 2, 1, 2, 1, 2],
        "flow_reference_stage": [None, None, f"{BASE}__s1", f"{BASE}__s2", f"{BASE}__s1", f"{BASE}__s2"],
        "flow_jsd": [None, None, 0.4, 0.3, 0.0, 0.002],
        "flow_jsd_occupancy": [None, None, 0.3, 0.2, 0.0, 0.001],
        "flow_jsd_pooled": [None, None, 0.33, 0.33, 0.0, 0.0],
        "flow_jsd_pooled_occupancy": [None, None, 0.25, 0.25, 0.0, 0.0],
        "flow_pooled_seeds": [None, None, 2, 2, 2, 2],
        "flow_paired_peds": [None, None, 14, 12, 14, 12],
        "flow_unpaired_peds": [None, None, 0, 0, 0, 0],
        "flow_room_ped_seconds": [16.0, 12.0, 16.0, 12.0, 16.0, 12.0],
        "flow_peds": [14, 12, 14, 12, 14, 12],
    }
)


def _jsd_renderer() -> FlowJsdRenderer:
    return FlowJsdRenderer(PlotSpec(id="flow_jsd_overview", type="flow_jsd", title="Flow change per edit", data_key="flow_jsd_pooled"))


def test_scores_show_the_largest_change_on_top_with_a_dot_per_seed() -> None:
    traces, layout = _figure(_jsd_renderer().render_plotly(SCORES))
    bars, dots = traces
    assert bars["y"] == ["move_fountain__ab", "insert_fountain_inert__cd"] and _array(bars["x"]) == pytest.approx([0.33, JSD_FLOOR])
    assert dots["y"] == ["move_fountain__ab"] * 2 + ["insert_fountain_inert__cd"] * 2 and _array(dots["x"]) == pytest.approx([0.4, 0.3, JSD_FLOOR, 0.002])
    assert layout["xaxis"]["type"] == "log" and layout["yaxis"]["categoryarray"] == ["insert_fountain_inert__cd", "move_fountain__ab"]


def test_scores_without_flow_columns_draw_nothing() -> None:
    assert _jsd_renderer().render_plotly(pl.DataFrame({"planner": ["dwb"], "success": [True]})) is None


def test_scores_are_one_png(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "flow_jsd.png"
    _jsd_renderer().render_seaborn(SCORES, out)
    assert out.read_bytes().startswith(PNG)


def test_the_flow_report_draws_the_scores_and_a_map_per_edit(tmp_path: pathlib.Path) -> None:
    run = _snapshot(tmp_path)
    SCORES.write_parquet(run / "combined_metrics.parquet")
    CELLS.write_parquet(run / FLOW_CELLS_FILE)
    ReportBuilder.from_dirs([run], run, manifest="flow").build()
    report = (run / "report.html").read_text()
    assert "Flow Change Report" in report and "move_fountain__ab against flowplot_room" in report
    assert {f"flow_map_{MOVE}.png", f"flow_map_{INERT}.png", "flow_jsd_overview.png"} <= {p.name for p in (run / "plots").iterdir()}
