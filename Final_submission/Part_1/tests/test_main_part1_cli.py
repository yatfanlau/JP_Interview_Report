# tests/test_main_part1_cli.py
def test_main_dispatch_basic(monkeypatch):
    import main_part1

    calls = {"basic": 0, "risky": 0}

    monkeypatch.setattr(main_part1, "run_basic_model", lambda: calls.__setitem__("basic", calls["basic"] + 1))
    monkeypatch.setattr(main_part1, "run_risky_debt_model", lambda: calls.__setitem__("risky", calls["risky"] + 1))

    main_part1.main(["--model", "basic"])
    assert calls["basic"] == 1
    assert calls["risky"] == 0


def test_main_dispatch_risky(monkeypatch):
    import main_part1

    calls = {"basic": 0, "risky": 0}

    monkeypatch.setattr(main_part1, "run_basic_model", lambda: calls.__setitem__("basic", calls["basic"] + 1))
    monkeypatch.setattr(main_part1, "run_risky_debt_model", lambda: calls.__setitem__("risky", calls["risky"] + 1))

    main_part1.main(["--model", "risky"])
    assert calls["basic"] == 0
    assert calls["risky"] == 1


def test_main_dispatch_both(monkeypatch):
    import main_part1

    calls = {"basic": 0, "risky": 0}

    monkeypatch.setattr(main_part1, "run_basic_model", lambda: calls.__setitem__("basic", calls["basic"] + 1))
    monkeypatch.setattr(main_part1, "run_risky_debt_model", lambda: calls.__setitem__("risky", calls["risky"] + 1))

    main_part1.main(["--model", "both"])
    assert calls["basic"] == 1
    assert calls["risky"] == 1