from fastapi.testclient import TestClient

from api.main import app


def test_state_endpoint_exposes_alert_only_machine_status():
    client = TestClient(app)
    res = client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "DailyEdge"
    assert data["execution_mode"] == "alert_only"
    assert data["auto_buy_enabled"] is False
    assert data["primary_setup"] == "breakout_20_rr3"
    assert isinstance(data["armed_watchers"], list)


def test_agent_context_endpoint_is_machine_readable():
    client = TestClient(app)
    res = client.get("/api/agent/context")
    assert res.status_code == 200
    data = res.json()
    assert data["contract"]["no_auto_buy"] is True
    assert "scanner" in data["capabilities"]
    assert "evidence" in data["capabilities"]
    assert "intraday" in data["capabilities"]


def test_glossary_endpoint_contains_core_terms():
    client = TestClient(app)
    res = client.get("/api/glossary")
    assert res.status_code == 200
    terms = {item["term"]: item for item in res.json()["terms"]}
    for required in ["breakout", "trigger", "R", "R:R", "ledger", "baseline", "re-stamp"]:
        assert required in terms
        assert terms[required]["plain"]
