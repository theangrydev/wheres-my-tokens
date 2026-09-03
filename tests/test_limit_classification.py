"""Tests for rate-limit event classification.

These exist because the classifier failed SILENTLY for three months. Anthropic
reworded the limit message in June 2026, from

    You've hit your limit - resets 4pm (UTC)
to
    You've hit your weekly limit - resets 9pm (Europe/London)

and the classifier tested `"hit your limit" in text`, which does not match the
new string. Every event after 2026-05 fell through to "unknown", downstream
stages filter on limit_type, and the charts simply went empty. Nothing raised.

Run with:  python3 -m unittest discover tests
"""

import unittest
from datetime import datetime, timezone

from wheres_my_tokens.reports.analyze import (
    _find_rate_limit_events, _limit_window_key, _reset_delta_hours,
)


def at(iso):
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


class ResetDeltaTest(unittest.TestCase):
    def test_bare_clock_time_later_today(self):
        self.assertAlmostEqual(
            _reset_delta_hours("You've hit your limit · resets 4pm (UTC)",
                               at("2026-04-07T11:02:00")),
            4 + 58 / 60, places=3)

    def test_bare_clock_time_rolls_to_tomorrow(self):
        # 3am quoted at 23:30 is 3.5h away, not 20.5h in the past.
        self.assertAlmostEqual(
            _reset_delta_hours("resets 3am (Europe/London)",
                               at("2026-04-06T23:30:00")),
            3.5, places=3)

    def test_dated_reset_can_still_be_five_hourly(self):
        # The date appears whenever the reset crosses midnight, so its presence
        # says NOTHING about which limit was hit. This is the case the old
        # date-presence heuristic got wrong 2191 times.
        self.assertAlmostEqual(
            _reset_delta_hours("resets Apr 7, 4pm (UTC)", at("2026-04-07T11:02:00")),
            4 + 58 / 60, places=3)

    def test_dated_reset_days_away(self):
        self.assertAlmostEqual(
            _reset_delta_hours("resets Aug 3 at 9pm (Europe/London)",
                               at("2026-07-29T14:00:00")),
            127.0, places=3)

    def test_year_boundary_does_not_read_as_eleven_months(self):
        self.assertAlmostEqual(
            _reset_delta_hours("resets Jan 2, 9am (UTC)", at("2025-12-31T22:00:00")),
            35.0, places=3)

    def test_no_reset_time(self):
        self.assertIsNone(_reset_delta_hours("no reset time here", at("2026-04-07T11:02:00")))

    def test_empty_and_none(self):
        self.assertIsNone(_reset_delta_hours("", at("2026-04-07T11:02:00")))
        self.assertIsNone(_reset_delta_hours(None, at("2026-04-07T11:02:00")))


class WindowKeyTest(unittest.TestCase):
    """One limit = one key, however many retries and directories logged it."""

    def ev(self, text, ts, group="acct", profile="p"):
        return {"text": text, "timestamp": at(ts), "group": group, "profile": profile}

    def test_retries_against_one_limit_collapse(self):
        # Retries fire every few minutes and all quote the same reset.
        a = self.ev("You've hit your session limit · resets 4pm", "2026-08-15T11:02:00")
        b = self.ev("You've hit your session limit · resets 4pm", "2026-08-15T11:47:00")
        self.assertEqual(_limit_window_key(a), _limit_window_key(b))

    def test_same_limit_from_different_directories_collapses(self):
        # Three agents on one account each log the same account-level limit.
        a = self.ev("You've hit your weekly limit · resets Aug 20 at 9pm",
                    "2026-08-15T11:02:00", profile="agent-1")
        b = self.ev("You've hit your weekly limit · resets Aug 20 at 9pm",
                    "2026-08-15T11:05:00", profile="agent-3")
        self.assertEqual(_limit_window_key(a), _limit_window_key(b))

    def test_weekly_retries_days_apart_still_collapse(self):
        # The whole reason the key is not a time bucket: a weekly limit is
        # retried for DAYS, and every retry quotes the same dated reset.
        a = self.ev("You've hit your weekly limit · resets Aug 20 at 9pm",
                    "2026-08-15T11:02:00")
        b = self.ev("You've hit your weekly limit · resets Aug 20 at 9pm",
                    "2026-08-18T22:30:00")
        self.assertEqual(_limit_window_key(a), _limit_window_key(b))

    def test_bare_time_on_different_days_does_not_collapse(self):
        # "resets 4pm" recurs daily, so these are two distinct limits.
        a = self.ev("You've hit your session limit · resets 4pm", "2026-08-15T11:02:00")
        b = self.ev("You've hit your session limit · resets 4pm", "2026-08-16T11:02:00")
        self.assertNotEqual(_limit_window_key(a), _limit_window_key(b))

    def test_different_accounts_do_not_collapse(self):
        a = self.ev("resets 4pm", "2026-08-15T11:02:00", group="acct-a")
        b = self.ev("resets 4pm", "2026-08-15T11:02:00", group="acct-b")
        self.assertNotEqual(_limit_window_key(a), _limit_window_key(b))

    def test_timezone_label_is_not_part_of_the_key(self):
        # Same limit, one message rendered in each timezone label.
        a = self.ev("resets 9pm (UTC)", "2026-08-15T11:02:00")
        b = self.ev("resets 9pm (Europe/London)", "2026-08-15T11:20:00")
        self.assertEqual(_limit_window_key(a), _limit_window_key(b))

    def test_no_reset_time_falls_back_to_a_bucket(self):
        a = self.ev("something went wrong", "2026-08-15T11:02:00")
        b = self.ev("something went wrong", "2026-08-15T11:47:00")
        c = self.ev("something went wrong", "2026-08-15T18:02:00")
        self.assertEqual(_limit_window_key(a), _limit_window_key(b))
        self.assertNotEqual(_limit_window_key(a), _limit_window_key(c))

    def test_group_resolved_from_map_when_event_lacks_one(self):
        e = {"text": "resets 4pm", "timestamp": at("2026-08-15T11:02:00"),
             "profile": "agent-1"}
        self.assertEqual(_limit_window_key(e, {"agent-1": "acct"})[0], "acct")


class FakeProfile:
    """Minimal stand-in: _find_rate_limit_events only needs .path and .name."""

    def __init__(self, path, name="fake"):
        self.path = path
        self.name = name
        self.email = "fake@example.com"


class ClassifyTest(unittest.TestCase):
    """End-to-end through _find_rate_limit_events, against real message text."""

    def classify(self, text, ts="2026-08-15T14:00:00Z", error="rate_limit"):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "projects" / "p"
            proj.mkdir(parents=True)
            rec = {
                "timestamp": ts,
                "error": error,
                "sessionId": "s1",
                "message": {"content": [{"type": "text", "text": text}]},
            }
            (proj / "a.jsonl").write_text(json.dumps(rec) + "\n")
            events = _find_rate_limit_events([FakeProfile(Path(d))])
        self.assertEqual(len(events), 1, f"expected 1 event for {text!r}")
        return events[0]["limit_type"]

    def test_new_wording_is_weekly(self):
        # The regression that motivated this file.
        self.assertEqual(
            self.classify("You've hit your weekly limit · resets 9pm (Europe/London)"),
            "weekly")

    def test_new_wording_with_date_is_weekly(self):
        self.assertEqual(
            self.classify("You've hit your weekly limit · resets Aug 3 at 9pm (Europe/London)"),
            "weekly")

    def test_old_wording_short_reset_is_five_hour(self):
        self.assertEqual(
            self.classify("You've hit your limit · resets 6pm (UTC)",
                          ts="2026-04-07T14:00:00Z"),
            "5-hour")

    def test_old_wording_dated_but_near_is_five_hour(self):
        self.assertEqual(
            self.classify("You've hit your limit · resets Apr 7, 4pm (UTC)",
                          ts="2026-04-07T11:02:00Z"),
            "5-hour")

    def test_old_wording_dated_and_far_is_weekly(self):
        self.assertEqual(
            self.classify("You've hit your limit · resets Apr 7, 4pm (UTC)",
                          ts="2026-04-03T09:00:00Z"),
            "weekly")

    def test_extra_usage(self):
        self.assertEqual(
            self.classify("You're out of extra usage · resets May 17, 2pm (UTC)"),
            "extra-usage")

    def test_usage_credits_is_its_own_class(self):
        # A model-scoped credit balance, exhausted while the account still has
        # ordinary headroom. An agent pinned to that model stops dead.
        self.assertEqual(
            self.classify("You're out of usage credits · resets 8pm (UTC)"),
            "credits")

    def test_compaction_cascade_is_detected(self):
        # Carries error "invalid_request", not "rate_limit", because the limit
        # landed on a full context window and auto-compaction could not run.
        self.assertEqual(
            self.classify("Prompt is too long · automatic compaction failed: "
                          "You've hit your weekly limit · resets 9pm",
                          error="invalid_request"),
            "weekly")

    def test_ordinary_invalid_request_is_not_a_limit_event(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "projects" / "p"
            proj.mkdir(parents=True)
            rec = {
                "timestamp": "2026-08-15T14:00:00Z",
                "error": "invalid_request",
                "message": {"content": [{"type": "text", "text": "malformed tool input"}]},
            }
            (proj / "a.jsonl").write_text(json.dumps(rec) + "\n")
            self.assertEqual(_find_rate_limit_events([FakeProfile(Path(d))]), [])


if __name__ == "__main__":
    unittest.main()
