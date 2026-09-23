from app.agents.parser import parse_query


def test_family_buy_with_golden_visa():
    q = parse_query("Family of 4, budget AED 2.2M, I work in DIFC, want good schools and Golden Visa eligibility")
    assert q.purpose == "sale" and q.budget_aed == 2_200_000
    assert q.min_bedrooms == 3 and q.family_with_kids and q.wants_golden_visa
    assert q.work_location == "DIFC"


def test_monthly_rent_is_annualised():
    q = parse_query("1 bed to rent near the metro, 8k per month, I work at Internet City")
    assert q.purpose == "rent" and q.budget_aed == 96_000 and q.min_bedrooms == 1
    assert q.work_location == "Dubai Internet City" and "metro" in q.lifestyle_tags


def test_work_hub_not_treated_as_home_area():
    q = parse_query("2 bed apartment in JVC, 1.2 million, office in Business Bay")
    assert q.work_location == "Business Bay"
    assert q.preferred_communities == ["Jumeirah Village Circle"]


def test_arabic_villa():
    q = parse_query("أبحث عن فيلا للعائلة قريبة من المدارس بميزانية 3 مليون")
    assert q.property_types == ["villa"] and q.budget_aed == 3_000_000 and q.family_with_kids


def test_please_is_not_lease():
    assert parse_query("Buy a 2 bed in Marina for 1.8M please").purpose == "sale"


def test_rental_yield_means_buying():
    q = parse_query("Investment apartment under 1M with high rental yield")
    assert q.purpose == "sale" and q.budget_aed == 1_000_000 and "investment" in q.lifestyle_tags
