import numpy as np

from app.config import get_settings
from app.recsys.bandit import CommunityThompsonBandit
from app.recsys.pipeline import ab_summary, assign_variant


class _T:  # Toolbox without a recommender, for pure-finance tools
    from app.agents.tools import Toolbox
    tb = Toolbox.__new__(Toolbox)
    tb.s = get_settings()


def test_affordability_expat_first_home():
    a = _T.tb.check_affordability(price_aed=2_000_000, monthly_income_aed=40_000)
    assert a["max_ltv"] == 0.80 and a["loan_aed"] == 1_600_000
    assert a["upfront_costs"]["dld_transfer_fee_4pct"] == 80_000
    assert 8_000 < a["monthly_installment_aed"] < 10_000 and a["within_dbr_cap"]


def test_affordability_ltv_tiers():
    assert _T.tb.check_affordability(6_000_000, 100_000)["max_ltv"] == 0.70
    assert _T.tb.check_affordability(1_000_000, 30_000, off_plan=True)["max_ltv"] == 0.50
    assert _T.tb.check_affordability(1_000_000, 30_000, uae_national=True)["max_ltv"] == 0.85


def test_golden_visa_threshold():
    assert _T.tb.check_golden_visa(2_000_000)["eligible"]
    assert not _T.tb.check_golden_visa(1_999_999)["eligible"]


def test_variant_assignment_is_sticky_and_balanced():
    assert assign_variant("abc", 0.5) == assign_variant("abc", 0.5)
    share = np.mean([assign_variant(str(i), 0.5) == "ranker" for i in range(4000)])
    assert 0.45 < share < 0.55


def test_ab_summary_detects_clear_winner():
    ev = [{"variant": "ranker", "event": "impression"}] * 1000 + [{"variant": "ranker", "event": "click"}] * 150
    ev += [{"variant": "baseline", "event": "impression"}] * 1000 + [{"variant": "baseline", "event": "click"}] * 80
    s = ab_summary(ev)
    assert s["p_ranker_better"] > 0.99 and s["decision"] == "ship ranker"


def test_bandit_learns_from_feedback():
    b = CommunityThompsonBandit(seed=0)
    for _ in range(30):
        b.update("Dubai Marina", "save")
        b.update("Arjan", "dismiss")
    post = b.posterior()
    assert post["Dubai Marina"]["mean"] > 0.9 and post["Arjan"]["mean"] < 0.1
