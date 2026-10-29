"""The shop counter (plan item b) [user review of run 20261029-184142, note 2: the counter's offers were
read as prices only]: each offer's tooltip names it, prices in hearts and on SALE are read, and the
hero's HP past 10 (Healthy) is seen. Frames: recording 2 (collect/20261029-160032), the user's cursor on
the counter; recording 1 (collect/20261028-190026); run 20261029-184142."""

from test_bot_flow import frame

from shogunbot.perception import shop_screen as SS, text
from shogunbot.perception.perceive import normalize, perceive

REC1, REC2 = "collect/20261028-190026/", "collect/20261029-160032/"


def test_tooltip_samples_read_back_exactly():
    for rel, title in text.TIP_SAMPLES.items():
        assert text.read_title(normalize(frame(rel)))[1] == title, rel


def test_the_tooltip_names_the_hovered_offer():
    want = {REC2 + "f0177_idle.png": ("skill", "combo_recharge"), REC2 + "f0784_idle.png": ("skill", "healthy"),
            REC2 + "f0446_idle.png": ("skill", "quick_recovery"), REC2 + "f0459_idle.png": ("skill", "two_way_move"),
            REC2 + "f0448_idle.png": ("skill", "dynamic_boost"), REC2 + "f1134_idle.png": ("skill", "combo_coin"),
            REC2 + "f1519_idle.png": ("skill", "reactive_shield"), REC2 + "f1744_idle.png": ("skill", "mindfulness"),
            REC1 + "f0332_idle.png": ("skill", "damaging_move"),
            REC2 + "f1144_idle.png": ("consumable", "edamame_brew"),
            REC2 + "f0182_CLICK.png": ("other", None), REC2 + "f0185_idle.png": ("other", None),
            REC2 + "f1752_idle.png": ("other", None)}
    for rel, (kind, oid) in want.items():
        tip = SS.tooltip(normalize(frame(rel)))
        assert tip is not None and (tip["kind"], tip["id"]) == (kind, oid), (rel, tip)
    assert SS.tooltip(normalize(frame(REC2 + "f0176_idle.png"))) is None          # no slot under the cursor
    # the box stands over the hovered slot's price (x 1160), not its neighbour's (1328)
    tip = SS.tooltip(normalize(frame(REC2 + "f0446_idle.png")))
    assert SS.tip_over(tip, 1160) and not SS.tip_over(tip, 1328)


def test_titles_with_unread_letters_still_name_the_offer():
    assert SS.identify("COM?O REC?ARGE") == ("skill", "combo_recharge")      # 0.31's glyphs lacked B and H
    assert SS.identify("OUICK RECO?ERY") == ("skill", "quick_recovery")
    assert SS.identify("SHOP UPGRADE") == ("other", None)
    assert SS.identify("") == ("other", None)


def test_prices_in_hearts_and_on_sale():
    def prices(rel):
        return [(o["price"], o["currency"]) for o in SS.counter(normalize(frame(rel)))]
    # the reroll for 1 heart read as 1 coin in 0.24-0.31
    assert prices("runs/20261029-184142/frames/shop_1.png") == [(10, "coin"), (4, "skull"), (1, "heart")]
    assert prices(REC2 + "f0790_idle.png") == [(7, "coin")]                  # Dynamic Boost level 2 on SALE
    assert prices(REC2 + "f1144_idle.png") == [(2, "coin"), (2, "heart")]    # Edamame Brew on SALE, a reroll
    assert prices(REC2 + "f0446_idle.png") == [(15, "coin"), (15, "coin"), (4, "skull")]
    assert prices(REC1 + "f0172_CLICK.png") == [(15, "coin"), (1, "skull")]


def test_the_hero_hp_past_ten():
    def hp(rel):
        p = perceive(normalize(frame(rel)), ["swirl", "arrow"])
        return p.hero["hp"], p.hero["hp_max"]
    assert hp(REC2 + "f0802_idle.png") == (10, 12)        # after Healthy (f0787); 0.31 read 10 of 10
    assert hp(REC2 + "f2006_SPACE.png") == (7, 12)
    assert hp(REC1 + "f0368_idle.png") == (5, 5)          # the board sliding in: as before (the drop check skips it)


# ---- the bot at the counter -------------------------------------------------------------------------
import json

from test_bot_flow import _stub_windows_modules, make_bot
from test_series import _GW, _Rec


class _HoverGW(_GW):
    """A shop whose counter shows a tooltip frame while the cursor rests on a slot (tips: slot x -> frame)."""
    def __init__(self, steps, tips):
        super().__init__(steps)
        self.tips, self.tip, self.hovers = tips, None, []

    def hover(self, fx, fy):
        x, y = round(fx * 1920), round(fy * 1080)
        self.hovers.append((x, y))
        self.tip = next((f for sx, f in self.tips.items() if abs(sx - x) < 60), None)

    def grab(self):
        return self.tip if self.tip is not None else super().grab()

    def move_mouse_away(self):
        self.tip = None

    def click(self, fx, fy):
        self.tip = None
        super().click(fx, fy)


def _events(b):
    return [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]


def test_the_bot_names_the_counters_offers_by_their_tooltips(tmp_path, monkeypatch):
    # the Moonlit Port shop of recording 2: Healthy for 15 on the first slot (tooltip at f0784), a padlock for
    # 1 skull, which is not hovered
    _stub_windows_modules()
    from shogunbot import bot as B
    monkeypatch.setattr(B, "HOVER_S", 0.01)
    gw = _HoverGW([(frame(REC2 + "f0783_idle.png"), [])], {1076: frame(REC2 + "f0784_idle.png")})
    b = make_bot(tmp_path, gw)
    b.rec, b.shop_n = _Rec(), 1
    offer = SS.read(normalize(gw.grab()))
    goods = b._shop_goods(gw.grab(), offer)
    assert [(g["x"], g["price"], g["kind"], g["id"]) for g in goods] == [(1076, 15, "skill", "healthy")]
    assert gw.hovers == [(1076, 375)]


def test_the_bot_buys_healthy_with_one_click_on_its_slot(tmp_path, monkeypatch):
    # your purchase at f0786: one click on the slot, coins 38 -> 23 (f0788)
    _stub_windows_modules()
    from shogunbot import bot as B
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    monkeypatch.setattr(B, "HOVER_S", 0.01)
    gw = _HoverGW([(frame(REC2 + "f0784_idle.png"), [(1000, 300, 1150, 450)]),
                   (frame(REC2 + "f0788_idle.png"), [])], {})
    b = make_bot(tmp_path, gw)
    b.rec, b.shop_n = _Rec(), 1
    tr = Tracker(Kit.of("swirl", "arrow"))
    offer = {"upgrade": None, "coins": 38,
             "goods": [{"x": 1076, "price": 15, "currency": "coin", "title": "HEALTHY", "kind": "skill", "id": "healthy"}]}
    b._shop_buy(gw.grab(), offer, [], tr, 10)
    assert gw.clicks == [(1076, 375)]
    assert tr.kit.level("healthy") == 1 and tr.kit.max_hp == 12
    bought = next(e for e in _events(b) if e["event"] == "shop_bought")
    assert (bought["choice"], bought["coins_before"], bought["coins_after"]) == ("skill:healthy", 38, 23)


def test_the_shops_order_follows_your_priorities():
    from shogunbot import rewards
    g = lambda sid, price=15: {"kind": "skill", "id": sid, "price": price, "currency": "coin"}
    pri = rewards.shop_priority
    # -2 cooldown and +1 damage first, then Healthy, then +1 upgrade slot, then other skills, then trade-offs
    assert pri("cooldown_minus_2") == 1 and pri("damage_1") == 1
    assert pri(good=g("healthy")) == 2
    assert pri("slot_1") == 3
    assert pri(good=g("quick_recovery")) == pri(good=g("two_way_move")) == 4
    assert pri("damage_2_cooldown_3") == pri(good=g("iron_skin")) == pri(good=g("fortress")) == 5
    # never: unsimulated skills, items, services, skull and heart prices, Sacrifice
    assert pri(good=g("combo_recharge")) is None
    assert pri(good={"kind": "consumable", "id": "edamame_brew", "price": 2, "currency": "coin"}) is None
    assert pri(good={"kind": "other", "id": None, "price": 1, "currency": "heart"}) is None
    assert pri(good=dict(g("healthy"), currency="heart")) is None
    assert pri("sacrifice") is None


def test_a_skill_is_bought_when_affordable_and_simulated():
    from shogunbot import rewards
    from shogunbot.model import Kit
    kit = Kit.of("swirl", "arrow")
    assert rewards.decide_skill(kit, "healthy", 15, 38, 10).best.name == "skill:healthy"
    assert rewards.decide_skill(kit, "healthy", 15, 14, 10).best.name == "skip"            # not affordable
    assert rewards.decide_skill(kit, "combo_recharge", 20, 31, 10).best.name == "skip"     # not simulated
    assert rewards.decide_skill(kit, None, 15, 38, 10).best.name == "skip"                 # unread
    d = rewards.decide_skill(kit.with_skill("dynamic_boost"), "dynamic_boost", 7, 23, 10)  # rec2 f0790: level 2
    assert d.best.name == "skill:dynamic_boost" and d.best.kit.level("dynamic_boost") == 2


def test_a_trade_off_skill_is_bought_only_above_keeping_the_coins():
    from shogunbot import rewards
    from shogunbot.model import Kit

    def ev(diff, se):
        def f(opts, hp, seeds, depth, workers, scenario, **kw):
            for o in opts:
                o.scores = [5.0] * 4
            for o in opts[1:]:
                o.diff, o.se = diff, se
        return f
    kit = Kit.of("swirl", "arrow")
    assert rewards.decide_skill(kit, "iron_skin", 20, 30, 10, evaluator=ev(0.5, 0.1)).best.name == "skill:iron_skin"
    assert rewards.decide_skill(kit, "iron_skin", 20, 30, 10, evaluator=ev(0.1, 0.1)).best.name == "skip"


def test_the_shop_buys_in_your_order(tmp_path, monkeypatch):
    # the Moonlit Port shop of recording 2 (f0783): the statue's +1 UPGRADE SLOT for 10, Healthy for 15 on the
    # counter, 38 coins; you bought Healthy, Dynamic Boost and then the slot. The bot: Healthy (2) before the
    # slot-only upgrade (3), other simulated skills (4) after, unsimulated ones never; each step sees the coins
    # the step before left
    _stub_windows_modules()
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    b = make_bot(tmp_path, _GW([(frame(REC2 + "f0783_idle.png"), [])]))
    b.rec, b.shop_n = _Rec(), 1
    calls = []

    def statue(frame_, offer, hand, tr, hp, coins, notes):
        calls.append(("statue", offer["upgrade"], coins))
        return frame_, coins - offer["upgrade_price"]

    def skill(frame_, g, tr, hp, coins):
        calls.append(("skill", g["id"], coins))
        return frame_, coins - g["price"]
    monkeypatch.setattr(b, "_shop_statue", statue)
    monkeypatch.setattr(b, "_shop_skill", skill)
    g = lambda sid, x, price=15: {"x": x, "price": price, "currency": "coin", "title": sid.upper(), "kind": "skill", "id": sid}
    offer = {"upgrade": "slot_1", "upgrade_price": 10, "coins": 38,
             "goods": [g("quick_recovery", 1160), g("combo_recharge", 1328, 20), g("healthy", 1496)]}
    b._shop_buy(b.gw.grab(), offer, [], Tracker(Kit.of("swirl", "arrow")), 10)
    assert calls == [("skill", "healthy", 38), ("statue", "slot_1", 23), ("skill", "quick_recovery", 13)]
    calls.clear()
    offer["upgrade"] = "cooldown_minus_2"                   # -2 cooldown comes first
    b._shop_buy(b.gw.grab(), offer, [], Tracker(Kit.of("swirl", "arrow")), 10)
    assert [c[1] for c in calls] == ["cooldown_minus_2", "healthy", "quick_recovery"]


def test_the_report_names_the_counters_offers(tmp_path):
    from shogunbot.analyze import analyze
    ev = [{"event": "shop", "n": 1, "upgrade": "slot_1", "upgrade_text": ["+1 UPGRADE SLOT"], "upgrade_price": 10,
           "coins": 38, "counter": [{"x": 1076, "price": 15, "currency": "coin"}, {"x": 1220, "price": 1, "currency": "skull"}],
           "goods": [{"x": 1076, "price": 15, "currency": "coin", "title": "HEALTHY", "kind": "skill", "id": "healthy"}]},
          {"event": "shop_decision", "skill": "healthy", "price": 15, "coins": 38, "hp": 10, "seconds": 0.0,
           "choice": "skill:healthy", "rule": "shop: Healthy for 15 of 38 coins, a pure gain: always bought",
           "options": [{"name": "skip", "scored": False, "value": 0.0, "wins": 0}]},
          {"event": "shop_bought", "choice": "skill:healthy", "coins_before": 38, "coins_after": 23}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
    text = analyze(tmp_path)
    assert "counter HEALTHY 15 coins, 1 skulls" in text
    assert "shop decision (skill healthy, 0.0s, HP 10)" in text and "shop_bought: skill:healthy  coins 38 -> 23" in text
