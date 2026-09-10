from scripts.export_site_results import build_payload, load_summary


def test_site_export_scrubs_paths_and_selects_highlight(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "case,variant,path,resolution,t_end,peak_speed,resolved\n"
        "n64,reference,/private/run,64,0.9,1.25,True\n",
        encoding="utf-8",
    )

    rows = load_summary(summary)
    payload = build_payload(rows, "n64", "2026-09-10T00:00:00+00:00")

    assert payload["headline"]["case"] == "n64"
    assert payload["headline"]["resolution"] == 64
    assert payload["headline"]["resolved"] is True
    assert payload["history_count"] == 1
    assert "path" not in payload["headline"]
