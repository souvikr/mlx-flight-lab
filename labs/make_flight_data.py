#!/usr/bin/env python3
"""
make_flight_data.py - build a tiny synthetic dataset for LoRA fine-tuning.

The task: turn a messy flight request into clean JSON.

  "Need 2 business class seats from Paris to Tokyo on the 14th of March"
  -> {"from": "Paris", "to": "Tokyo", "date": "March 14", "passengers": 2, "cabin": "business"}

Rules the model has to pick up purely from examples:
  * dates are normalised to "Month D"      ("14 Mar", "the 14th of March" -> "March 14")
  * passengers is an integer               ("two adults" -> 2, "just me" -> 1, nothing said -> 1)
  * cabin is one of economy / premium_economy / business / first   (nothing said -> economy)

Output is mlx-lm's *chat* format, one JSON object per line:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

valid/test only use cities that never appear in train, so a good test score
means the model learned to *copy* names, not memorise them.

Usage:
  python labs/make_flight_data.py                          # -> data/flight/{train,valid,test}.jsonl
  python labs/make_flight_data.py --train 100 --out data/flight-100   # small-data experiment
  python labs/make_flight_data.py --seed 1 --out data/flight-seed1    # a different random draw
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM = (
    "Convert the flight request into JSON with keys "
    "from, to, date, passengers, cabin."
)

TRAIN_CITIES = [
    "Paris", "Tokyo", "London", "New York", "Delhi", "Mumbai", "Dubai", "Singapore",
    "Sydney", "Toronto", "Berlin", "Madrid", "Rome", "Amsterdam", "Bangkok", "Seoul",
    "Hong Kong", "San Francisco", "Los Angeles", "Chicago", "Boston", "Miami",
    "Mexico City", "Sao Paulo", "Buenos Aires", "Cairo", "Istanbul", "Athens",
    "Lisbon", "Dublin", "Zurich", "Vienna", "Prague", "Warsaw", "Stockholm", "Oslo",
    "Helsinki", "Copenhagen", "Nairobi", "Cape Town", "Doha", "Kuala Lumpur",
    "Jakarta", "Manila", "Hanoi", "Osaka", "Melbourne", "Auckland", "Vancouver",
    "Seattle",
]
# Never used for training: only in valid/test.
HELDOUT_CITIES = [
    "Kolkata", "Lima", "Bogota", "Reykjavik", "Marrakech", "Edinburgh", "Denver",
    "Montreal", "Perth", "Taipei", "Kathmandu", "Lagos", "Brussels", "Budapest",
    "Santiago",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
ABBR = {m: m[:3] for m in MONTHS}
ABBR["September"] = "Sept"

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def ordinal(d):
    if 11 <= d % 100 <= 13:
        return f"{d}th"
    return f"{d}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th') }"


def date_text(rng, month, day):
    return rng.choice([
        f"{month} {day}",
        f"{day} {month}",
        f"{month} {ordinal(day)}",
        f"the {ordinal(day)} of {month}",
        f"{ABBR[month]} {day}",
        f"{day} {ABBR[month]}",
        f"{ordinal(day)} {month}",
    ])


def num(rng, n):
    """2 -> '2' or 'two'"""
    return str(n) if rng.random() < 0.5 else NUM_WORDS[n]


# ---------------------------------------------------------------- passengers
# Each style is a different grammatical slot the templates can use.
#   np   : noun phrase      "Book {np} ..."            -> "2 tickets"
#   for  : prepositional    "... {for}"                -> "for two adults"
#   sent : whole sentence   "{sent} We want to fly..." -> "There are 2 of us."
def pax_np(rng, n):
    if n is None:
        return "a flight"
    if n == 1:
        return rng.choice(["one ticket", "a single ticket", "1 seat", "one seat", "a ticket"])
    return f"{num(rng, n)} {rng.choice(['tickets', 'seats'])}"


def pax_for(rng, n):
    if n is None:
        return ""
    special = {
        1: ["for myself", "for one person", "just for me", "for 1 passenger"],
        2: ["for me and my wife", "for me and my husband", "for the two of us",
            "for me and my partner", "for a couple"],
        3: ["for me and my two kids", "for the three of us"],
        4: ["for my family of four", "for a family of 4"],
        5: ["for a family of five", "for the five of us"],
        6: ["for a group of six", "for 6 friends"],
    }
    if rng.random() < 0.4:
        return rng.choice(special[n])
    noun = rng.choice(["people", "passengers", "adults"]) if n > 1 else rng.choice(["person", "passenger", "adult"])
    return f"for {num(rng, n)} {noun}"


def pax_sent(rng, n):
    if n is None:
        return ""
    special = {
        1: ["It's just me.", "Only me travelling.", "Solo trip.", "Just one passenger."],
        2: ["Me and my wife.", "There are two of us.", "Two passengers.", "Travelling as a couple."],
        3: ["Me and my two kids.", "There are 3 of us.", "Three passengers."],
        4: ["We are a family of four.", "There are four of us.", "4 passengers."],
        5: ["We're a family of five.", "Five passengers.", "There are 5 of us."],
        6: ["We are a group of 6.", "Six passengers.", "There are six of us."],
    }
    return rng.choice(special[n])


# -------------------------------------------------------------------- cabin
CABIN_WORDS = {
    "economy": ["economy", "economy class", "coach", "the cheapest seats"],
    "premium_economy": ["premium economy", "premium economy class"],
    "business": ["business", "business class", "biz class"],
    "first": ["first class", "first"],
}


def cabin_adj(rng, c):            # "business class tickets", "first class seats"
    if c is None:
        return ""
    w = rng.choice([w for w in CABIN_WORDS[c] if w not in ("the cheapest seats", "first")])
    return w


def cabin_in(rng, c):             # "... in business class"
    if c is None:
        return ""
    w = rng.choice([w for w in CABIN_WORDS[c] if w != "the cheapest seats"])
    if w == "first":
        w = "first class"
    return f"in {w}"


def cabin_sent(rng, c):           # "Business class please."
    if c is None:
        return ""
    return rng.choice({
        "economy": ["Economy is fine.", "Cheapest seats please.", "Coach is fine.", "Economy please."],
        "premium_economy": ["Premium economy please.", "Premium economy if possible."],
        "business": ["Business class please.", "We'd like business.", "Biz class if available."],
        "first": ["First class please.", "We want first class.", "Only first class."],
    }[c])


# -------------------------------------------------------------------- route
def route(rng, a, b, after_verb=False):
    options = [
        f"from {a} to {b}",
        f"from {a} to {b}",
        f"to {b} from {a}",
        f"{a} to {b}",
        f"out of {a} to {b}",
    ]
    if not after_verb:                      # "Flying leaving Paris for..." reads badly
        options.append(f"leaving {a} for {b}")
    return rng.choice(options)


def on_date(rng, d):
    return rng.choice([f"on {d}", f"on {d}", f"for {d}", f"departing {d}"])


def join(*parts, sep=", "):
    return sep.join(p for p in parts if p)


# ---------------------------------------------------------------- templates
def render(rng, a, b, d, n, c):
    """Render one request. n=None / c=None mean 'not mentioned'."""
    t = rng.randrange(12)
    R, D = route(rng, a, b, after_verb=t in (4, 6, 9)), on_date(rng, d)
    if t == 0:
        return f"Book {pax_np(rng, n)} {R} {D}" + (f", {cabin_in(rng, c)}" if c else "") + "."
    if t == 1:
        return "I need a flight " + join(f"{R} {D}", pax_for(rng, n), cabin_in(rng, c)) + "."
    if t == 2:
        return join(f"{a} to {b}", d, pax_np(rng, n) if n else "", cabin_adj(rng, c))
    if t == 3:
        adj = cabin_adj(rng, c)
        return f"Can you find {adj + ' ' if adj else ''}tickets {R} {D}" + (f" {pax_for(rng, n)}" if n else "") + "?"
    if t == 4:
        return join("Hi!", pax_sent(rng, n), f"We want to fly {R} {D}.", cabin_sent(rng, c), sep=" ")
    if t == 5:
        return f"{d}: {a} -> {b}" + "".join(f", {x}" for x in (pax_np(rng, n) if n else "", cabin_adj(rng, c)) if x)
    if t == 6:
        return join(f"Looking to travel {R} {D}.", pax_sent(rng, n), cabin_sent(rng, c), sep=" ")
    if t == 7:
        adj = cabin_adj(rng, c)
        return f"Need {pax_np(rng, n)} {R} {D}" + (f", {adj}" if adj else "") + "."
    if t == 8:
        return join(f"Is there anything {R} {D}?", pax_sent(rng, n), cabin_sent(rng, c), sep=" ")
    if t == 9:
        return join(pax_sent(rng, n), f"Flying {R} on {d}.", cabin_sent(rng, c), sep=" ")
    if t == 10:
        adj = cabin_adj(rng, c)
        return f"Please book {adj + ' ' if adj else 'a flight '}{R} for {d}" + (f", {pax_for(rng, n)}" if n else "") + "."
    return join(f"Get me to {b} from {a} {D}", pax_for(rng, n), cabin_in(rng, c), sep=" ") + "."


def make_example(rng, cities):
    a, b = rng.sample(cities, 2)
    month, day = rng.choice(MONTHS), rng.randint(1, 28)
    n = rng.choices([1, 2, 3, 4, 5, 6], weights=[35, 30, 12, 13, 5, 5])[0]
    c = rng.choices(["economy", "premium_economy", "business", "first"], weights=[45, 15, 25, 15])[0]
    n_said = n if rng.random() > 0.2 else None          # 20%: passengers not mentioned
    if n_said is None:
        n = 1
    c_said = c if rng.random() > 0.3 else None          # 30%: cabin not mentioned
    if c_said is None:
        c = "economy"
    text = render(rng, a, b, date_text(rng, month, day), n_said, c_said)
    if rng.random() < 0.25:
        text = rng.choice(["Hello! ", "Hey! ", "Hi there. ", "Quick one: "]) + text
    if rng.random() < 0.2:
        text += rng.choice([" Thanks!", " Thank you.", " Cheers", " ASAP please."])
    answer = {"from": a, "to": b, "date": f"{month} {day}", "passengers": n, "cabin": c}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    }


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):4d} examples -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/flight")
    p.add_argument("--train", type=int, default=800)
    p.add_argument("--valid", type=int, default=100)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.jsonl", [make_example(rng, TRAIN_CITIES) for _ in range(args.train)])
    write(out / "valid.jsonl", [make_example(rng, HELDOUT_CITIES) for _ in range(args.valid)])
    write(out / "test.jsonl", [make_example(rng, HELDOUT_CITIES) for _ in range(args.test)])


if __name__ == "__main__":
    main()
