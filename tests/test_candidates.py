"""candidates 的規則（用手寫節點，不需要 h5py）。"""

from __future__ import annotations

from types import SimpleNamespace

from assembly_monitor.candidates import _bool_like, build_candidates


def _g(path, n):
    return {"path": path, "name": path.rsplit("/", 1)[-1] or "/", "kind": "group", "n_members": n, "shape": None,
            "media": None, "time_stats": None, "preview": None, "dtype": None}


def _s(path, value):
    return {"path": path, "name": path.rsplit("/", 1)[-1], "kind": "dataset", "shape": [], "media": None,
            "time_stats": None, "preview": value, "dtype": "float64"}


def _nodes(seg_values):
    nodes = [_g("/", 1), _g("/ann", len(seg_values))]
    for i, (a, b, ok, txt) in enumerate(seg_values):
        nodes += [_g(f"/ann/{i}", 4), _s(f"/ann/{i}/t_start", a), _s(f"/ann/{i}/t_end", b),
                  _s(f"/ann/{i}/success", ok), _s(f"/ann/{i}/name", txt)]
    return nodes


def test_bool_like():
    assert _bool_like(True) is True and _bool_like(0) is False
    assert _bool_like("Failure") is False and _bool_like("yes") is True
    assert _bool_like(2) is None and _bool_like("maybe") is None


def test_container_detection_by_structure_not_name():
    nodes = _nodes([(1.0, 2.0, True, "Insert gear"), (2.0, 3.0, False, "Insert gear"), (3.0, 2.5, True, "Pick bolt")])
    c = build_candidates(nodes, SimpleNamespace(expected_cameras=[]))
    ct = c["segments"]["containers"][0]
    assert ct["path"] == "/ann"
    assert ct["keys"]["start"] == "t_start" and ct["keys"]["end"] == "t_end"
    assert ct["keys"]["success"] == "success" and ct["keys"]["text"] == "name"
    codes = {a["code"] for a in c["anomalies"]}
    assert "SEG_START_GE_END" in codes
    ins = c["segments"]["actions"]["level0"]["insert"]
    assert ins["count"] == 2 and ins["objects"]["gear"]["success_false"] == 1
    assert c["time_base"]["relation"] == "unknown"  # 沒有時間陣列可比對


def test_no_container_without_start_end():
    nodes = [_g("/", 1), _g("/x", 2), _g("/x/0", 1), _s("/x/0/value", 1.0), _g("/x/1", 1), _s("/x/1/value", 2.0)]
    c = build_candidates(nodes, SimpleNamespace(expected_cameras=[]))
    assert c["segments"]["containers"] == []
    assert any(q["topic"] == "segment 欄位" for q in c["questions"])
