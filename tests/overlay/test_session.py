import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from vizaudit.overlay.config import (
    MarkerConfig,
    ObjectConfig,
    OrientationConfig,
    OverlayConfig,
    PatternConfig,
)
from vizaudit.overlay.dataset_watcher import DatasetInfo
from vizaudit.overlay.session import run_session

CAMERA_KEY = "observation.images.top"


def _points_pattern(points: list) -> PatternConfig:
    return PatternConfig(shape="points", points=points)


def _run(config: OverlayConfig, finished_episode_indices: list):
    """Runs run_session with dataset/watcher/rerun internals mocked, driving show_targets for
    episode 0 plus one call per entry in `finished_episode_indices` (each representing a
    finished-episode-index the watcher reports, so show_targets(finished + 1) fires).
    Returns every log_target/clear_target call, in order, as (kind, args, kwargs)."""
    calls = []
    mock_watcher = MagicMock()
    mock_watcher.next_episode_boundary.return_value = iter(finished_episode_indices)

    with (
        patch("vizaudit.overlay.session.wait_for_dataset_root"),
        patch(
            "vizaudit.overlay.session.read_dataset_info",
            return_value=DatasetInfo(fps=30.0, image_keys=[CAMERA_KEY]),
        ),
        patch("vizaudit.overlay.session.connect"),
        patch("vizaudit.overlay.session.EpisodeBoundaryWatcher", return_value=mock_watcher),
        patch(
            "vizaudit.overlay.session.log_target",
            side_effect=lambda *a, **kw: calls.append(("log_target", a, kw)),
        ),
        patch(
            "vizaudit.overlay.session.clear_target",
            side_effect=lambda *a, **kw: calls.append(("clear_target", a, kw)),
        ),
    ):
        run_session(config, Path("/fake/dataset"), "localhost", 9876)
    return calls


def test_no_variable_objects_does_nothing():
    config = OverlayConfig(
        camera_key=CAMERA_KEY,
        objects=[ObjectConfig(name="fixed", count=1, variable=False, pattern=None, marker=MarkerConfig())],
        marker=MarkerConfig(),
    )
    calls = _run(config, [])
    assert calls == []


def test_default_knobs_reproduce_lockstep_position_cycling():
    points = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    obj = ObjectConfig(name="cube", count=3, variable=True, pattern=_points_pattern(points), marker=MarkerConfig())
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=[obj], marker=MarkerConfig())

    calls = _run(config, [0, 1, 2, 3])  # episodes 0,1,2,3,4
    log_calls = [c for c in calls if c[0] == "log_target"]
    assert len(log_calls) == 5
    shown_points = [c[1][1] for c in log_calls]
    assert shown_points == [points[i % 3] for i in range(5)]
    # Default behavior: no stacking, no fan-out, no level badge.
    for _, _, kwargs in log_calls:
        assert kwargs["level"] == 0
        assert kwargs["stack_size"] == 1


def test_stack_forms_when_objects_share_a_point_and_assigns_levels():
    # Stacking is no longer an authored, separate config block -- it's derived directly from
    # 2+ objects' own `pattern: {shape: points}` lists sharing a literal coordinate, with
    # config.level_strategy/level_seed (scene-level) deciding z-order. Default co_location is
    # "stack" (today's behavior).
    point_a = ObjectConfig(
        name="A", count=1, variable=True, pattern=_points_pattern([(50.0, 50.0)]), marker=MarkerConfig(),
    )
    point_b = ObjectConfig(
        name="B", count=1, variable=True, pattern=_points_pattern([(50.0, 50.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=[point_a, point_b], marker=MarkerConfig())

    calls = _run(config, [])  # just episode 0
    log_calls = {c[1][2]: c for c in calls if c[0] == "log_target"}  # keyed by object_name
    assert log_calls["A"][1][1] == (50.0, 50.0)
    assert log_calls["B"][1][1] == (50.0, 50.0)
    assert log_calls["A"][2]["stack_size"] == 2
    assert log_calls["B"][2]["stack_size"] == 2
    # "fixed" (default) level strategy + declaration order: A (declared first) is level 0.
    assert log_calls["A"][2]["level"] == 0
    assert log_calls["B"][2]["level"] == 1


def test_scene_level_balanced_level_strategy_changes_across_episodes():
    point_a = ObjectConfig(
        name="A", count=1, variable=True, pattern=_points_pattern([(50.0, 50.0)]), marker=MarkerConfig(),
    )
    point_b = ObjectConfig(
        name="B", count=1, variable=True, pattern=_points_pattern([(50.0, 50.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[point_a, point_b], marker=MarkerConfig(),
        level_strategy="balanced", level_seed=3,
    )

    calls = _run(config, [0])  # episodes 0, 1
    levels_by_episode = []
    for _kind, args, kwargs in [c for c in calls if c[0] == "log_target"]:
        levels_by_episode.append((args[2], kwargs["level"]))
    # 4 log_target calls total (2 objects x 2 episodes); A and B must always be at distinct
    # levels within the same episode (Latin-square guarantee from pattern.level_order).
    by_episode = [levels_by_episode[0:2], levels_by_episode[2:4]]
    for pair in by_episode:
        assert {level for _name, level in pair} == {0, 1}


def test_emergent_stack_forms_when_independent_patterns_coincide():
    # A always sits at (10, 10) (length-1 pattern). B alternates (99, 99)/(10, 10) lockstep --
    # so episode 0 has no coincidence (stack_size 1 each), episode 1 does (stack_size 2 each).
    obj_a = ObjectConfig(
        name="A", count=1, variable=True, pattern=_points_pattern([(10.0, 10.0)]), marker=MarkerConfig(),
    )
    obj_b = ObjectConfig(
        name="B", count=2, variable=True,
        pattern=_points_pattern([(99.0, 99.0), (10.0, 10.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig())

    calls = _run(config, [0])  # episodes 0, 1
    log_calls = [c for c in calls if c[0] == "log_target"]
    by_episode_and_name = {(i // 2, c[1][2]): c for i, c in enumerate(log_calls)}

    assert by_episode_and_name[(0, "A")][2]["stack_size"] == 1
    assert by_episode_and_name[(0, "B")][2]["stack_size"] == 1
    assert by_episode_and_name[(1, "A")][2]["stack_size"] == 2
    assert by_episode_and_name[(1, "B")][2]["stack_size"] == 2
    # Declaration order (A before B) decides level under the default "fixed" level_strategy.
    assert by_episode_and_name[(1, "A")][2]["level"] == 0
    assert by_episode_and_name[(1, "B")][2]["level"] == 1


def test_stacked_object_orientation_cycles_every_episode():
    # A length-1 pattern naturally gives visit_number = episode_index // 1 == episode_index --
    # i.e. its orientation cycles every episode regardless of whether it's part of a stack.
    obj_a = ObjectConfig(
        name="A", count=1, variable=True, pattern=_points_pattern([(50.0, 50.0)]), marker=MarkerConfig(),
        orientation=OrientationConfig(count=3, method="uniform"),
    )
    obj_b = ObjectConfig(
        name="B", count=1, variable=True, pattern=_points_pattern([(50.0, 50.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig())

    calls = _run(config, [0, 1, 2])  # episodes 0..3
    a_calls = [c for c in calls if c[0] == "log_target" and c[1][2] == "A"]
    orientation_tips = [c[2]["orientation_tip"] for c in a_calls]
    # 4 episodes, orientation.count=3 -> tips cycle with period 3, not "stuck" at one angle.
    assert len(set(orientation_tips)) == 3
    assert orientation_tips[0] == orientation_tips[3]


def test_co_location_keep_apart_separates_objects_with_alternative_points():
    # Both objects naturally want index 0 this episode (lockstep, same shape), but B has a
    # free alternative (index 1) in its own assigned list -- keep_apart should use it instead
    # of letting them coincide.
    obj_a = ObjectConfig(
        name="A", count=2, variable=True,
        pattern=_points_pattern([(0.0, 0.0), (1.0, 1.0)]), marker=MarkerConfig(),
    )
    obj_b = ObjectConfig(
        name="B", count=2, variable=True,
        pattern=_points_pattern([(0.0, 0.0), (2.0, 2.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), co_location="keep_apart",
    )

    calls = _run(config, [])  # episode 0
    log_calls = {c[1][2]: c for c in calls if c[0] == "log_target"}
    assert log_calls["A"][1][1] == (0.0, 0.0)
    assert log_calls["B"][1][1] == (2.0, 2.0)  # nudged off the shared point
    assert log_calls["A"][2]["stack_size"] == 1
    assert log_calls["B"][2]["stack_size"] == 1


def test_co_location_keep_apart_residual_collision_falls_back_to_stack_and_warns(caplog):
    # Both objects have ONLY one shared point -- there's nowhere for either to go, so
    # keep_apart degrades to rendering them as a (visible, fanned) stack and logs a warning
    # rather than silently overlapping with no indication.
    obj_a = ObjectConfig(
        name="A", count=1, variable=True, pattern=_points_pattern([(5.0, 5.0)]), marker=MarkerConfig(),
    )
    obj_b = ObjectConfig(
        name="B", count=1, variable=True, pattern=_points_pattern([(5.0, 5.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), co_location="keep_apart",
    )

    with caplog.at_level(logging.WARNING):
        calls = _run(config, [])
    log_calls = {c[1][2]: c for c in calls if c[0] == "log_target"}
    assert log_calls["A"][2]["stack_size"] == 2
    assert log_calls["B"][2]["stack_size"] == 2
    assert any("keep_apart" in r.message for r in caplog.records)
    assert any("B" in r.message for r in caplog.records)


def test_co_location_keep_apart_disjoint_objects_unaffected():
    obj_a = ObjectConfig(
        name="A", count=1, variable=True, pattern=_points_pattern([(1.0, 1.0)]), marker=MarkerConfig(),
    )
    obj_b = ObjectConfig(
        name="B", count=1, variable=True, pattern=_points_pattern([(2.0, 2.0)]), marker=MarkerConfig(),
    )
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), co_location="keep_apart",
    )

    calls = _run(config, [])
    log_calls = {c[1][2]: c for c in calls if c[0] == "log_target"}
    assert log_calls["A"][1][1] == (1.0, 1.0)
    assert log_calls["B"][1][1] == (2.0, 2.0)
    assert log_calls["A"][2]["stack_size"] == 1
    assert log_calls["B"][2]["stack_size"] == 1


def test_episode_targets_one_rotates_independent_objects_round_robin():
    # A and B never coincide -- every episode has 2 singleton sites, declaration order [A, B].
    # Under episode_targets="one" exactly one site is shown per episode, alternating.
    obj_a = ObjectConfig(name="A", count=1, variable=True, pattern=_points_pattern([(1.0, 1.0)]), marker=MarkerConfig())
    obj_b = ObjectConfig(name="B", count=1, variable=True, pattern=_points_pattern([(2.0, 2.0)]), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), episode_targets="one",
    )

    calls = _run(config, [0, 1, 2])  # episodes 0,1,2,3
    by_episode = [calls[i * 2:i * 2 + 2] for i in range(4)]
    expected_active = ["A", "B", "A", "B"]
    for episode_calls, active_name in zip(by_episode, expected_active):
        # clear_target's args are (camera_key, object_name) -- object_name is args[1], NOT
        # args[2] (that's only true for log_target's (camera_key, point, object_name, marker)).
        kinds_by_name = {
            (c[1][2] if c[0] == "log_target" else c[1][1]): c[0] for c in episode_calls
        }
        other_name = "B" if active_name == "A" else "A"
        assert kinds_by_name[active_name] == "log_target"
        assert kinds_by_name[other_name] == "clear_target"


def test_episode_targets_one_keeps_a_permanent_stack_together_as_one_site():
    # A/B/C all pinned to the same point, every episode -- always ONE site (declared first,
    # since A is ordinal 0); D is independent (its own site). Only one site is active per
    # episode, so the stack and the independent object alternate, never both at once.
    shared = [(50.0, 50.0)]
    obj_a = ObjectConfig(name="A", count=1, variable=True, pattern=_points_pattern(shared), marker=MarkerConfig())
    obj_b = ObjectConfig(name="B", count=1, variable=True, pattern=_points_pattern(shared), marker=MarkerConfig())
    obj_c = ObjectConfig(name="C", count=1, variable=True, pattern=_points_pattern(shared), marker=MarkerConfig())
    obj_d = ObjectConfig(name="D", count=1, variable=True, pattern=_points_pattern([(99.0, 99.0)]), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b, obj_c, obj_d], marker=MarkerConfig(),
        episode_targets="one",
    )

    calls = _run(config, [0])  # episodes 0, 1
    # clear_target's args are (camera_key, object_name) -- object_name is args[1], NOT args[2]
    # (that's only true for log_target's longer (camera_key, point, object_name, marker) args).
    by_kind_episode0 = {c[1][2]: c for c in calls[:4] if c[0] == "log_target"}
    cleared_episode0 = {c[1][1] for c in calls[:4] if c[0] == "clear_target"}
    assert set(by_kind_episode0) == {"A", "B", "C"}
    assert cleared_episode0 == {"D"}
    for name in ("A", "B", "C"):
        assert by_kind_episode0[name][2]["stack_size"] == 3

    by_kind_episode1 = {c[1][2]: c for c in calls[4:8] if c[0] == "log_target"}
    cleared_episode1 = {c[1][1] for c in calls[4:8] if c[0] == "clear_target"}
    assert set(by_kind_episode1) == {"D"}
    assert cleared_episode1 == {"A", "B", "C"}
    assert by_kind_episode1["D"][2]["stack_size"] == 1


def test_combination_count_unset_falls_back_to_lockstep_byte_for_byte():
    # Default (no combination_count): identical to the pre-existing lockstep behavior.
    points = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    obj = ObjectConfig(name="cube", count=3, variable=True, pattern=_points_pattern(points), marker=MarkerConfig())
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=[obj], marker=MarkerConfig())

    calls = _run(config, [0, 1, 2, 3])
    log_calls = [c for c in calls if c[0] == "log_target"]
    shown_points = [c[1][1] for c in log_calls]
    assert shown_points == [points[i % 3] for i in range(5)]


def test_combination_count_decouples_combinations_from_pattern_length():
    # 3 objects, each with a 4-point pattern -- under plain lockstep this would cap the
    # distinct combination count at 4 (reported directly: "if i have 3 objects and 20 steps
    # the combinations are limited to the nr of steps"). combination_count=10 decouples that:
    # the resolved index per object must match pattern.combination_index directly, and the
    # whole combination cycles with period 10 (episode 10 reproduces episode 0 exactly).
    from vizaudit.overlay.pattern import combination_index

    points_per_object = [
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        [(10.0, 10.0), (11.0, 11.0), (12.0, 12.0), (13.0, 13.0)],
        [(20.0, 20.0), (21.0, 21.0), (22.0, 22.0), (23.0, 23.0)],
    ]
    objs = [
        ObjectConfig(name=n, count=4, variable=True, pattern=_points_pattern(pts), marker=MarkerConfig())
        for n, pts in zip(["A", "B", "C"], points_per_object)
    ]
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=objs, marker=MarkerConfig(), combination_count=10)

    calls = _run(config, list(range(10)))  # episodes 0..10
    log_calls = [c for c in calls if c[0] == "log_target"]
    by_episode = [log_calls[i * 3:i * 3 + 3] for i in range(11)]
    for episode_index, episode_calls in enumerate(by_episode):
        by_name = {c[1][2]: c[1][1] for c in episode_calls}
        for ordinal, name in enumerate(["A", "B", "C"]):
            expected_idx = combination_index(episode_index % 10, 4, ordinal, 10)
            assert by_name[name] == points_per_object[ordinal][expected_idx]
    # Episode 10 reproduces episode 0 exactly -- period is combination_count, not 4.
    assert by_episode[10] == by_episode[0]


def test_combination_count_oversampling_no_longer_stalls_on_consecutive_episodes():
    # The reported bug: setting combination_count higher than an object's own pattern length
    # repeated the SAME point for several consecutive episodes instead of cycling.
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    obj = ObjectConfig(name="A", count=4, variable=True, pattern=_points_pattern(points), marker=MarkerConfig())
    config = OverlayConfig(camera_key=CAMERA_KEY, objects=[obj], marker=MarkerConfig(), combination_count=20)

    calls = _run(config, list(range(19)))  # episodes 0..19
    shown = [c[1][1] for c in calls if c[0] == "log_target"]
    for a, b in zip(shown, shown[1:]):
        assert a != b, "consecutive episodes repeated the same point -- the stalling bug is back"


def test_combination_mode_random_uses_full_joint_space_not_capped_by_one_length():
    # 3 objects, all length 5 -- under "systematic"/"coprime" the JOINT combination space is
    # mathematically capped at 5 (every object is a deterministic function of i mod 5). Random
    # draws independently per object, so it can exceed that cap.
    points_per_object = [[(float(i), float(i)) for i in range(5)] for _ in range(3)]
    objs = [
        ObjectConfig(name=n, count=5, variable=True, pattern=_points_pattern(pts), marker=MarkerConfig())
        for n, pts in zip(["A", "B", "C"], points_per_object)
    ]
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=objs, marker=MarkerConfig(),
        combination_count=60, combination_mode="random", combination_seed=7,
    )
    calls = _run(config, list(range(59)))
    log_calls = [c for c in calls if c[0] == "log_target"]
    by_episode = [log_calls[i * 3:i * 3 + 3] for i in range(60)]
    joint_tuples = {tuple(c[1][1] for c in episode_calls) for episode_calls in by_episode}
    assert len(joint_tuples) > 5  # escapes the per-object cap that systematic/coprime can't


def test_combination_mode_coprime_is_deterministic_and_capped_like_systematic():
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    obj = ObjectConfig(name="A", count=3, variable=True, pattern=_points_pattern(points), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj], marker=MarkerConfig(),
        combination_count=9, combination_mode="coprime",
    )
    calls_a = _run(config, list(range(8)))
    calls_b = _run(config, list(range(8)))
    assert calls_a == calls_b  # no randomness at all
    shown = {c[1][1] for c in calls_a if c[0] == "log_target"}
    assert shown == set(points)  # still capped at this object's own 3 points


def test_combination_mode_cartesian_escapes_the_per_object_cap_deterministically():
    # 2 objects, length 3 each -- cartesian's natural period is the full product (9), every
    # combination appearing exactly once, fully deterministically (no seed needed).
    points_a = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    points_b = [(10.0, 10.0), (11.0, 11.0), (12.0, 12.0)]
    obj_a = ObjectConfig(name="A", count=3, variable=True, pattern=_points_pattern(points_a), marker=MarkerConfig())
    obj_b = ObjectConfig(name="B", count=3, variable=True, pattern=_points_pattern(points_b), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), combination_mode="cartesian",
    )

    calls = _run(config, list(range(8)))  # episodes 0..8, natural period is 3*3=9
    log_calls = [c for c in calls if c[0] == "log_target"]
    by_episode = [log_calls[i * 2:i * 2 + 2] for i in range(9)]
    joint_tuples = {tuple(c[1][1] for c in episode_calls) for episode_calls in by_episode}
    assert len(joint_tuples) == 9  # every (A, B) combination appears exactly once
    # Episode 9 (one full period later) reproduces episode 0 exactly.
    calls_extended = _run(config, list(range(9)))
    log_calls_extended = [c for c in calls_extended if c[0] == "log_target"]
    by_episode_extended = [log_calls_extended[i * 2:i * 2 + 2] for i in range(10)]
    assert by_episode_extended[9] == by_episode_extended[0]


def test_combination_mode_lcm_runs_exactly_lcm_episodes_with_no_phase():
    # Lengths 4 and 6 -- lcm(4, 6) = 12, NOT max(4, 6) = 6 (what plain lockstep alone would
    # naturally repeat at) and NOT 4*6 = 24 (the full cartesian product).
    points_a = [(float(i), 0.0) for i in range(4)]
    points_b = [(0.0, float(i)) for i in range(6)]
    obj_a = ObjectConfig(name="A", count=4, variable=True, pattern=_points_pattern(points_a), marker=MarkerConfig())
    obj_b = ObjectConfig(name="B", count=6, variable=True, pattern=_points_pattern(points_b), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), combination_mode="lcm",
    )

    calls = _run(config, list(range(12)))  # episodes 0..12
    log_calls = [c for c in calls if c[0] == "log_target"]
    by_episode = [log_calls[i * 2:i * 2 + 2] for i in range(13)]
    for episode_index, episode_calls in enumerate(by_episode):
        by_name = {c[1][2]: c[1][1] for c in episode_calls}
        assert by_name["A"] == points_a[episode_index % 4]
        assert by_name["B"] == points_b[episode_index % 6]
    joint_tuples = {tuple(c[1][1] for c in episode_calls) for episode_calls in by_episode[:12]}
    assert len(joint_tuples) == 12  # every lockstep pairing appears exactly once
    assert by_episode[12] == by_episode[0]  # repeats only after the full lcm period


def test_episode_targets_one_iterates_every_one_of_an_objects_own_points_as_separate_sites():
    # Under episode_targets="one", sites are the STATIC union of every object's own assigned
    # points (pattern.occupied_sites), not a per-episode sweep -- an object with several
    # assigned points gets one dedicated turn per point, not a single "active turn" that
    # sweeps across them. A has 3 own points (none shared with B), B has 1 -- so there are 4
    # static sites total, visited round-robin in declaration order: A's 3 points (in their own
    # list order), then B's 1 point.
    points_a = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
    obj_a = ObjectConfig(name="A", count=3, variable=True, pattern=_points_pattern(points_a), marker=MarkerConfig())
    obj_b = ObjectConfig(name="B", count=1, variable=True, pattern=_points_pattern([(1.0, 1.0)]), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b], marker=MarkerConfig(), episode_targets="one",
    )

    calls = _run(config, [0, 1, 2, 3, 4, 5, 6])  # episodes 0..7 -- 2 full rotations of period 4
    log_calls = [c for c in calls if c[0] == "log_target"]
    shown = [(c[1][2], c[1][1]) for c in log_calls]
    expected_cycle = [("A", points_a[0]), ("A", points_a[1]), ("A", points_a[2]), ("B", (1.0, 1.0))]
    assert shown == expected_cycle + expected_cycle


def test_episode_targets_one_objects_sharing_any_point_in_their_own_list_always_stack_there():
    # Assignment to the same point is now a GUARANTEED, permanent stack -- not a per-episode
    # coincidence that only sometimes happens to line up (the previous, dynamic-coincidence
    # design this replaced -- see CLAUDE.md). A and B share point (50,50) in their own pattern
    # lists; A also has a private point (5,5), B also has a private point (99,99); C is fully
    # independent. occupied_sites groups by exact coordinate across ALL of each object's own
    # points (not just "this episode's swept position"), giving 4 static sites, ordered by the
    # earliest (object_ordinal, point_index) that touches each: {A,B}@(50,50), {A}@(5,5),
    # {B}@(99,99), {C}@(7,7).
    obj_a = ObjectConfig(
        name="A", count=2, variable=True,
        pattern=_points_pattern([(50.0, 50.0), (5.0, 5.0)]), marker=MarkerConfig(),
    )
    obj_b = ObjectConfig(
        name="B", count=2, variable=True,
        pattern=_points_pattern([(50.0, 50.0), (99.0, 99.0)]), marker=MarkerConfig(),
    )
    obj_c = ObjectConfig(name="C", count=1, variable=True, pattern=_points_pattern([(7.0, 7.0)]), marker=MarkerConfig())
    config = OverlayConfig(
        camera_key=CAMERA_KEY, objects=[obj_a, obj_b, obj_c], marker=MarkerConfig(), episode_targets="one",
    )

    calls = _run(config, [0, 1, 2, 3])  # episodes 0..4 -- one full rotation of period 4, plus one
    by_episode = [calls[i * 3:i * 3 + 3] for i in range(5)]

    # Episode 0: the {A,B} stack at (50,50) -- declared earliest (A is ordinal 0).
    by_name_ep0 = {c[1][2]: c for c in by_episode[0] if c[0] == "log_target"}
    cleared_ep0 = {c[1][1] for c in by_episode[0] if c[0] == "clear_target"}
    assert set(by_name_ep0) == {"A", "B"}
    assert cleared_ep0 == {"C"}
    assert by_name_ep0["A"][1][1] == (50.0, 50.0)
    assert by_name_ep0["B"][1][1] == (50.0, 50.0)
    assert by_name_ep0["A"][2]["stack_size"] == 2
    assert by_name_ep0["B"][2]["stack_size"] == 2

    # Episode 1: A's own private point (5,5) -- a singleton site.
    by_name_ep1 = {c[1][2]: c for c in by_episode[1] if c[0] == "log_target"}
    cleared_ep1 = {c[1][1] for c in by_episode[1] if c[0] == "clear_target"}
    assert set(by_name_ep1) == {"A"}
    assert cleared_ep1 == {"B", "C"}
    assert by_name_ep1["A"][1][1] == (5.0, 5.0)
    assert by_name_ep1["A"][2]["stack_size"] == 1

    # Episode 2: B's own private point (99,99).
    by_name_ep2 = {c[1][2]: c for c in by_episode[2] if c[0] == "log_target"}
    assert set(by_name_ep2) == {"B"}
    assert by_name_ep2["B"][1][1] == (99.0, 99.0)

    # Episode 3: C's independent point -- and episode 4 (start of the next rotation) repeats
    # episode 0's {A,B} stack exactly, confirming the rotation is static/period-4, not subject
    # to any per-episode coincidence check.
    by_name_ep3 = {c[1][2]: c for c in by_episode[3] if c[0] == "log_target"}
    assert set(by_name_ep3) == {"C"}
    by_name_ep4 = {c[1][2]: c for c in calls[12:15] if c[0] == "log_target"}
    assert set(by_name_ep4) == {"A", "B"}
    assert by_name_ep4["A"][2]["stack_size"] == 2
