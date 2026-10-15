import json

from clearance.static_demo import record_trace, write


def test_trace_is_the_real_decisions():
    steps = record_trace()["steps"]
    asks = [s for s in steps if s["decision"] == "ASK"]
    assert [s["surface"] for s in asks] == ["permission", "email"]  # one permission, one exception
    assert asks[0]["proposal"]["recipients"] and asks[1]["to"] == ["omar@contoso.test"]
    sends = [s for s in steps if s["op"] == "send"]
    assert len(sends) == 5 and sum(s["decision"] == "ALLOW" for s in sends) == 4
    assert all(s["by"].startswith("permission:") for s in sends if s["decision"] == "ALLOW")


def test_page_embeds_the_trace(tmp_path):
    out = write(tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "/*TRACE*/" not in html and out["steps"] == len(record_trace()["steps"])
    blob = html.split("const TRACE = ", 1)[1].split(";\nconst $", 1)[0]
    assert json.loads(blob.replace("<\\/", "</"))["steps"][0]["surface"] == "permission"
