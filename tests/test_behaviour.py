"""M3: enemy behaviour learned from the run logs (behaviour.py, data/behaviour.json).

Each test is a pattern the logs show [runs 20261028-044407 .. 20261028-104830,
shogunbot.mine]; the shares quoted are from those runs.
"""

from shogunbot import behaviour as B
from shogunbot.mine import Run, enemy_samples
from shogunbot.model import Enemy, Hero, State


def board(cells, hero_x, enemies, **kw):
    es = []
    for i, e in enumerate(enemies):
        kind, x, facing, hp, queue = e[:5]
        es.append(Enemy(kind, x, facing, hp, list(queue), uid=i + 1, **(e[5] if len(e) > 5 else {})))
    return State(cells, Hero(hero_x, 1), es, **kw)


def probs(s, i=0):
    return dict((it, p) for p, it in B.intents(s, s.enemies[i]))


def test_the_model_file_is_there_and_beats_the_old_presets():
    m = B.model()
    assert m is not None
    cv = m.meta["cross_validated"]
    assert cv["learned"]["top1"] >= 0.85 > cv["presets"]["top1"]


def test_a_new_enemy_holds_its_first_phase():
    # enemies seen for the first time telegraph 'hold' (93-100% across distances)
    s = board(7, 3, [("ashigaru", 5, -1, 3, [])])
    p = probs(s)
    assert max(p, key=p.get) == "hold"


def test_a_just_loaded_enemy_holds_and_never_fires():
    # R1: the phase after queuing, every loaded enemy held (ashigaru 41/41 at range, 18/18 adjacent)
    s = board(5, 2, [("ashigaru", 3, -1, 3, ["sword"], {"last": "queue", "wind": 0})])
    p = probs(s)
    assert "fire" not in p and p.get("hold", 0) > 0.9


def test_loaded_earlier_with_the_hero_in_reach_fires():
    # R2 [run 20261028-053637]; in the logs 17/17, 31/31 and 19/20 fired
    s = board(5, 2, [("ashigaru", 3, -1, 3, ["sword"], {"last": "hold"})])
    assert probs(s) == {"fire": 1.0}


def test_after_firing_an_ashigaru_steps_back_then_reloads():
    # the phase after firing it does not reload: with the hero out of reach it steps
    # away (70%), next to the hero it mostly holds; one phase later it queues (90%)
    s = board(7, 3, [("ashigaru", 5, -1, 3, [], {"last": "fire", "fage": 0})])
    p = probs(s)
    assert "queue" not in p and max(p, key=p.get) == "move-right"
    s = board(7, 3, [("ashigaru", 4, -1, 3, [], {"last": "fire", "fage": 0})])
    assert "queue" not in probs(s)
    s = board(7, 3, [("ashigaru", 5, -1, 3, [], {"last": "move", "fage": 1})])
    p = probs(s)
    assert max(p, key=p.get) == "queue"


def test_an_advancing_enemy_keeps_advancing():
    # loaded, two cells away, moved last phase: toward 95%
    s = board(7, 1, [("ashigaru", 3, -1, 3, ["sword"], {"last": "move"})])
    p = probs(s)
    assert max(p, key=p.get) == "move-left"


def test_an_enemy_facing_away_turns():
    # facing away after a hold: turn 94-100%
    s = board(7, 3, [("tachi", 5, 1, 2, [], {"last": "hold"})])
    p = probs(s)
    assert max(p, key=p.get) == "turn"


def test_a_charger_with_a_blocked_lane_never_fires():
    # [user] a charge needs a clear lane; 0 of 35 blocked Chargers fired in the logs
    s = board(5, 0, [("ashigaru", 2, -1, 3, []), ("charger", 4, -1, 1, ["charge"], {"last": "hold"})])
    assert "fire" not in probs(s, 1)


def test_the_miner_labels_telegraphs_and_links_turns():
    # two turns of a synthetic run: the Ashigaru telegraphs a step, then is found one cell on
    s1 = board(5, 1, [("ashigaru", 4, -1, 3, ["sword"])], level="bamboo-1")
    s1.enemies[0].intent = "move-left"
    s2 = board(5, 1, [("ashigaru", 3, -1, 3, ["sword"])], level="bamboo-1")
    s2.enemies[0].intent = "hold"
    run = Run("synthetic", "0.17", [{"turn": 0, "level": "bamboo-1", "state": s1.to_dict()},
                                    {"turn": 1, "level": "bamboo-1", "state": s2.to_dict()}])
    rows = enemy_samples(run)
    assert [r["y"] for r in rows] == ["toward", "hold"]
    assert rows[0]["last"] is None and rows[1]["last"] == "move"


def test_the_tracker_carries_each_enemys_history_into_the_next_turn():
    # live boards get the same history features the model was learned with
    from types import SimpleNamespace
    from shogunbot.bridge import Tracker

    def read(intent, queue):
        hero = {"x": 1, "facing": 1, "hp": 10, "hp_max": 10, "queue": [], "swap_red": 0}
        en = [{"kind": "ashigaru", "x": 2, "facing": -1, "hp": 3, "queue": queue, "intent": intent}]
        return SimpleNamespace(hero=hero, hand=[], cells=list(range(5)), screen="combat", enemies=en,
                               confidence={"hero_facing": 1.0}, boss_frac=None, items=[])

    tr = Tracker()
    s1 = tr.state(read("fire", ["sword"]))
    tr.acted(s1, "wait")
    s2 = tr.state(read("move-right", []))
    assert (s2.enemies[0].last, s2.enemies[0].fage) == ("fire", 0)
    tr.acted(s2, "wait")
    s3 = tr.state(read("queue", []))
    assert (s3.enemies[0].last, s3.enemies[0].fage) == ("move", 1)


def test_memoised_predictions_are_the_same_and_safe_to_change():
    # 0.29: predictions are memoised by feature values (about a tenth of a valuation fight's time);
    # intents() edits the dict it gets, which must not change the next prediction
    m = B.model()
    s = board(5, 0, [("ashigaru", 2, -1, 3, ["sword"])])
    f = B.features(s, s.enemies[0], "hold", 2)
    fresh = B.Model(m.counts, m.meta, m.boss_tiles).predict(f)
    first = m.predict(f)
    first["fire"] = 0.0
    assert m.predict(f) == fresh and m.predict(f) is not m.predict(f)
