#!/usr/bin/env python3
import argparse
import itertools
import json
import random
from pathlib import Path
from typing import List, Union


class TimeInterval:
    def __init__(self, lower: int, upper: Union[int, float], l_incl: bool, r_incl: bool):
        self.lower = lower
        self.upper = upper
        self.l_incl = l_incl
        self.r_incl = r_incl

    def is_invalid(self) -> bool:
        return self.lower >= self.upper

    def to_constraint_string(self) -> str:
        left = "[" if self.l_incl else "("
        right = "]" if self.r_incl else ")"
        upper = "+" if self.upper == float("inf") else str(self.upper)
        return f"{left}{self.lower},{upper}{right}"


class Location:
    def __init__(self, loc_id: int, name: str, accepting: bool):
        self.id = loc_id
        self.name = name
        self.accepting = accepting


class Transition:
    def __init__(self, source: Location, target: Location, action: str, guard: TimeInterval, reset: bool):
        self.source = source
        self.target = target
        self.action = action
        self.guard = guard
        self.reset = reset


class TA:
    def __init__(self, name: str, clock_name: str, locations: List[Location], transitions: List[Transition], alphabet: List[str]):
        self.name = name
        self.clock_name = clock_name
        self.locations = locations
        self.transitions = transitions
        self.alphabet = alphabet
        self.init_location = locations[0].name

    def to_dict(self):
        transitions = sorted(
            self.transitions,
            key=lambda t: (t.source.id, t.action, t.guard.lower, t.guard.upper),
        )
        return {
            "name": self.name,
            "clocks": [self.clock_name],
            "actions": self.alphabet,
            "locations": [
                {"name": loc.name, "accepting": loc.accepting, "invariant": {}}
                for loc in self.locations
            ],
            "transitions": [
                {
                    "source": t.source.name,
                    "action": t.action,
                    "guard": {self.clock_name: t.guard.to_constraint_string()},
                    "reset": [self.clock_name] if t.reset else [],
                    "target": t.target.name,
                }
                for t in transitions
            ],
            "init_location": self.init_location,
        }


def validate_ta_parameters(location_cnt: int, transition_cnt: int, max_clock_value: int, alphabet_size: int) -> None:
    if location_cnt < 1 or alphabet_size < 1:
        raise ValueError("locations and alphabet size must be positive")
    if max_clock_value < 0:
        raise ValueError("--max-k must be nonnegative")
    minimum = location_cnt * alphabet_size
    # Sink guards are never split; each other guard can be split at 1, ..., K.
    maximum = minimum + (location_cnt - 1) * alphabet_size * max_clock_value
    if not minimum <= transition_cnt <= maximum:
        raise ValueError(
            f"transitions ({transition_cnt}) must be between {minimum} and {maximum} "
            "for the chosen locations, alphabet size and max-k"
        )


def generate_random_ta(name: str, clock: str, alphabet: List[str], location_cnt: int, transition_cnt: int, max_clock_value: int) -> TA:
    validate_ta_parameters(location_cnt, transition_cnt, max_clock_value, len(alphabet))
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must contain distinct actions")

    sink_idx = location_cnt - 1
    locations = [
        Location(i, "Sink" if i == sink_idx else f"L{i}", i != sink_idx)
        for i in range(location_cnt)
    ]
    sink = locations[sink_idx]

    def random_target(source_idx: int) -> Location:
        return sink if source_idx == sink_idx else random.choice(locations)

    transitions: List[Transition] = []
    covered = set()
    action_cycle = itertools.cycle(alphabet)

    accepting = list(range(sink_idx))
    if accepting:
        visited = {accepting[0]}
        unvisited = set(accepting[1:])
        while unvisited:
            src = random.choice(list(visited))
            action = next((a for a in (next(action_cycle) for _ in range(len(alphabet))) if (src, a) not in covered), None)
            if action is None:
                continue
            target = random.choice(list(unvisited))
            transitions.append(Transition(locations[src], locations[target], action, TimeInterval(0, float("inf"), True, False), random.choice([True, False])))
            covered.add((src, action))
            visited.add(target)
            unvisited.remove(target)

    for src in random.sample(list(range(sink_idx)), sink_idx):
        action = next((a for a in alphabet if (src, a) not in covered), None)
        if action is not None:
            transitions.append(Transition(locations[src], sink, action, TimeInterval(0, float("inf"), True, False), True))
            covered.add((src, action))
            break

    for src in locations:
        for action in alphabet:
            if (src.id, action) in covered:
                continue
            target = random_target(src.id)
            transitions.append(Transition(src, target, action, TimeInterval(0, float("inf"), True, False), target.id == sink_idx or random.choice([True, False])))
            covered.add((src.id, action))

    while len(transitions) < transition_cnt:
        candidates = []
        for idx, transition in enumerate(transitions):
            if transition.source.id == sink_idx:
                continue
            upper = min(transition.guard.upper, max_clock_value)
            for split in range(int(transition.guard.lower), int(upper) + 1):
                left_closed = (TimeInterval(transition.guard.lower, split, transition.guard.l_incl, True),
                               TimeInterval(split, transition.guard.upper, False, transition.guard.r_incl))
                right_closed = (TimeInterval(transition.guard.lower, split, transition.guard.l_incl, False),
                                TimeInterval(split, transition.guard.upper, True, transition.guard.r_incl))
                if not left_closed[0].is_invalid() and not left_closed[1].is_invalid():
                    candidates.append((idx, left_closed))
                if not right_closed[0].is_invalid() and not right_closed[1].is_invalid():
                    candidates.append((idx, right_closed))
        if not candidates:
            raise RuntimeError("no guard split available for the requested transition count")

        idx, (first, second) = random.choice(candidates)
        original = transitions[idx]
        original.guard = first
        target = random_target(original.source.id)
        transitions.append(Transition(original.source, target, original.action, second, target.id == sink_idx or random.choice([True, False])))

    return TA(name, clock, locations, transitions, alphabet)


def generate_small_rta(
    name: str,
    clock: str,
    alphabet: List[str],
    non_sink_locations: int,
    max_clock_value: int,
    rng: random.Random,
) -> TA:
    """Reachable partial RTA with 1–2 business edges/state and one simple cycle.

    A path visits every location. Its last edge closes one cycle. Optional
    forward shortcuts leave only the acyclic prefix, so they add no cycles.
    Missing transitions are rejection, completed to Sink by the Java loader.
    """
    if not 2 <= non_sink_locations <= 5:
        raise ValueError('small RTA requires 2..5 non-Sink locations')
    if len(alphabet) != 2 or len(set(alphabet)) != 2:
        raise ValueError('small RTA review suite uses two distinct local actions')
    if max_clock_value < 1:
        raise ValueError('small RTA guard constants must be positive')
    locations = [Location(i, f'L{i}', True) for i in range(non_sink_locations)]
    rest = locations[1:]
    rng.shuffle(rest)
    path = [locations[0], *rest]
    cycle_entry = rng.randrange(non_sink_locations)
    transitions = []
    for index, source in enumerate(path):
        actions = rng.sample(alphabet, len(alphabet))
        targets = [path[index + 1] if index + 1 < non_sink_locations else path[cycle_entry]]
        if index < cycle_entry and rng.random() < 0.5:
            targets.append(rng.choice(path[index + 1:]))
        for action, target in zip(actions, targets):
            k = rng.randint(1, max_clock_value)
            guard = (TimeInterval(0, k, True, True) if rng.randrange(2) == 0
                     else TimeInterval(k, float('inf'), True, False))
            transitions.append(Transition(source, target, action, guard, True))
    return TA(name, clock, locations, transitions, list(alphabet))


def generate_benchmark_rta(name, clock, private_action, shared_actions,
                           location_count, max_k, width, shared_segments,
                           private_branches, rng):
    """Reset-only benchmark component with a private reachable cycle.

    Shared guards/targets vary with location; private cycles let an independent
    assessor realize each global location and synchronized transition tuple.
    """
    locations = [Location(i, f"L{i}", True) for i in range(location_count)]
    path = [locations[0], *rng.sample(locations[1:], location_count - 1)]
    transitions = [
        Transition(q, path[(j + 1) % location_count], private_action,
                   TimeInterval(0, float("inf"), True, False), True)
        for j, q in enumerate(path)
    ]
    alphabet = [private_action, *shared_actions]
    if private_branches:
        alphabet.append(private_action + "b")
    for q in locations:
        for action in alphabet[1:]:
            segments = shared_segments if action in shared_actions else 1
            for segment in range(segments):
                # Disjoint bands leave a genuine rejected gap between segments.
                start = 0 if segment == 0 else max_k // 2 + 1
                end = max_k // 2 - 1 if segments == 2 and segment == 0 else max_k
                span = min(rng.randint(*width), end - start)
                low = rng.randint(start, end - span)
                high = low + span
                guard = TimeInterval(low, high, bool(rng.randrange(2)), bool(rng.randrange(2)))
                transitions.append(Transition(q, rng.choice(locations), action, guard, True))
    return TA(name, clock, locations, transitions, alphabet)


def generate_local_timing_rta(name, clock, private_action, shared_actions,
                             location_count, max_k, rng,
                             timed_shared=False, shared_effect="random",
                             guard_profile="deadline"):
    """Small NRTA family with observable private deadlines and shared resets.

    The universal private cycle reaches every location. A second private action
    has a location-dependent integer deadline. Shared actions reset the clock
    and may change location; their timing constraint is an explicit complexity
    dimension rather than being imposed on every component from the start.
    """
    if location_count < 2 or max_k < 2:
        raise ValueError("Local timing family needs at least two locations and K>=2")
    if guard_profile not in ("deadline", "window", "split", "narrow"):
        raise ValueError("Unknown private guard profile")
    if guard_profile == "window" and max_k < 3:
        raise ValueError("Window profile needs K>=3")
    if guard_profile == "split" and max_k < 5:
        raise ValueError("Split profile needs K>=5")
    locations = [Location(i, f"L{i}", True) for i in range(location_count)]
    order = [locations[0], *rng.sample(locations[1:], location_count - 1)]
    deadlines = [1 + i % max_k for i in range(location_count)]
    rng.shuffle(deadlines)
    transitions = []
    for j, q in enumerate(order):
        transitions.append(Transition(
            q, order[(j + 1) % location_count], private_action,
            TimeInterval(0, float("inf"), True, False), True))
    for i, q in enumerate(locations):
        if guard_profile == "deadline":
            guards = [TimeInterval(0, deadlines[i], True, True)]
        elif guard_profile == "window":
            width = max_k // 2
            low = i % (max_k - width + 1)
            guards = [TimeInterval(low, low + width,
                                   bool(rng.randrange(2)), bool(rng.randrange(2)))]
        elif guard_profile == "narrow":
            low = i * (max_k - 1) // (location_count - 1)
            guards = [TimeInterval(low, low + 1,
                                   bool(rng.randrange(2)), bool(rng.randrange(2)))]
        else:
            offset = i % 2
            guards = [TimeInterval(offset, offset + 1, True, True),
                      TimeInterval(max_k - 2 + offset, max_k - 1 + offset,
                                   bool(rng.randrange(2)), bool(rng.randrange(2)))]
        for guard in guards:
            transitions.append(Transition(
                q, rng.choice(locations), private_action + "b", guard, True))
        for action in shared_actions:
            target = q if shared_effect == "stay" else rng.choice(locations)
            guard = (TimeInterval(0, max_k, True, True) if timed_shared
                     else TimeInterval(0, float("inf"), True, False))
            transitions.append(Transition(q, target, action, guard, True))
    return TA(name, clock, locations, transitions,
              [private_action, private_action + "b", *shared_actions])


def ring_alphabets(actions: List[str], component_count: int, block_size: int) -> List[List[str]]:
    if component_count < 1:
        raise ValueError("--components must be positive")
    if not 1 <= block_size <= len(actions):
        raise ValueError("--block-size must be between 1 and --actions")
    return [
        sorted({actions[(i + offset) % len(actions)] for offset in range(block_size)})
        for i in range(component_count)
    ]


def action_names(count: int) -> List[str]:
    if not 1 <= count <= 26:
        raise ValueError("--actions must be between 1 and 26")
    return [chr(ord("a") + i) for i in range(count)]


def write_case(
    case_dir: Path,
    alphabets: List[List[str]],
    locations: int,
    transitions_per_location: int,
    max_k: int,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for i, alphabet in enumerate(alphabets, start=1):
        ta = generate_random_ta(
            name=f"P{i}",
            clock=f"x{i}",
            alphabet=alphabet,
            location_cnt=locations,
            transition_cnt=locations * transitions_per_location,
            max_clock_value=max_k,
        )
        path = case_dir / f"P{i}.json"
        path.write_text(json.dumps(ta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random single-clock TA cases with uniform component parameters."
    )
    parser.add_argument("--out", type=Path, required=True, help="output root for the parameter group")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--actions", type=int, default=4, help="size of the global action pool (1-26)")
    parser.add_argument("--block-size", type=int, default=3, help="alphabet size of every component")
    parser.add_argument("--locations", type=int, default=4, help="locations per component, including Sink")
    parser.add_argument(
        "--transitions-per-location", type=int, default=5,
        help="average transitions per location; total per component = locations * this value",
    )
    parser.add_argument("--max-k", type=int, default=12, help="upper bound for integer guard constants")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    try:
        ring_alphabets(action_names(args.actions), args.components, args.block_size)
        validate_ta_parameters(
            args.locations, args.locations * args.transitions_per_location, args.max_k, args.block_size
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    alphabets = ring_alphabets(action_names(args.actions), args.components, args.block_size)
    group_name = (
        f"TA{args.components}_ACT{args.actions}_BLOCK{args.block_size}"
        f"_LOC{args.locations}_TPL{args.transitions_per_location}_K{args.max_k}"
    )
    group_dir = args.out / group_name
    for case_id in range(args.cases):
        write_case(
            group_dir / f"case{case_id}", alphabets,
            args.locations, args.transitions_per_location, args.max_k,
        )
    print(group_dir)


if __name__ == "__main__":
    main()
