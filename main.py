import os
import json
import hashlib
import sqlite3
from pathlib import Path
from collections import Counter
from datetime import datetime

import openai

client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-5.4"
PDF_PATH = Path("california-drivers-handbook.pdf")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
FILE_ID_CACHE = CACHE_DIR / "file_id.txt"
DB_PATH = Path("rulebook_pipeline.db")


def ensure_file_uploaded(pdf_path: Path) -> str:
    """Return an OpenAI file_id for the handbook PDF.

    Uploads the PDF on first run and caches the resulting id to
    cache/file_id.txt. On subsequent runs the cached id is reused, unless
    the file has been deleted server-side — in which case we re-upload.
    """
    if FILE_ID_CACHE.exists():
        cached = FILE_ID_CACHE.read_text().strip()
        if cached:
            try:
                client.files.retrieve(cached)
                print(f"  [file_id cache hit: {cached}]")
                return cached
            except Exception as e:
                print(f"  [cached file_id {cached} unusable ({e}); re-uploading]")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    print(f"  [uploading {pdf_path} to OpenAI…]")
    with pdf_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    FILE_ID_CACHE.write_text(uploaded.id)
    print(f"  [uploaded, file_id={uploaded.id}]")
    return uploaded.id


# ---------------------------------------------------------------------------
# Shared simulation context — reused across prompts so the model understands
# what the ScenicRules benchmark can actually observe and score.
# ---------------------------------------------------------------------------
SIMULATION_CONTEXT = """
ScenicRules benchmark — simulation and evaluation model:

A sample is one Scenic driving scenario executed in a 2D road-network simulator
(MetaDrive or the Scenic Newtonian driving simulator). The scenario produces a
finite discrete rollout of traffic participants over time, which is
post-processed into a Realization object and evaluated offline.

Map / world:
  • Known 2D road map with drivable area, lane polygons, lane centerlines,
    lane orientation, lane connectivity, lane maneuvers, intersections, and
    sometimes lane speed limits.
  • Scenarios cover lane following, straight driving, left/right turns, lane
    changes, pedestrian crossings, and some bicycle-passing / near-accident
    situations.

Objects in scope:
  • Ego vehicle (the evaluated subject)
  • Other vehicles (cars, trucks)
  • Vulnerable road users: pedestrians and bicycles

Observable state per object per timestep:
  • 2D position, 2D velocity, orientation/yaw
  • Polygonal footprint from object dimensions
  • Object type, stable identity across time
Derived: acceleration (from Δv), lane occupancy, correct/incorrect-direction
lane sets, polygon and center distances, polygon intersection (= collision),
proximity-filtered nearby vehicles/VRUs, ego trajectory linestring, front/rear
ego path segments, collision timeline.

NOT observable / out of scope — do not rely on:
  • Traffic light or stop sign state
  • Turn signals, brake lights, horn, eye contact, gestures, human intent
  • Raw sensor data, occlusion, perception uncertainty, detection confidence
  • Weather, lighting, road surface, visibility
  • Passenger-comfort signals not inferable from trajectory
  • Legal doctrines or social norms not reducible to map + trajectory
  • Internal policy/network state of the driving agent
  • Fault attribution or what other agents "should have" intended

Rule contract: a rule is a function over (realization, timestep) returning 0
when no violation, a positive scalar otherwise, with larger values for more
severe violations. Values are aggregated (max/sum) over the rollout. Rules
must therefore be grounded in geometry, kinematics, lane topology, and
observable interactions — measurable continuous violation magnitudes, not
vague preferences.
""".strip()


def init_database():
    """Initialize SQLite database for caching all pipeline steps."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create cache table for all steps
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_cache (
            step_id TEXT PRIMARY KEY,
            step_name TEXT NOT NULL,
            output_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create table for STL conversions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stl_conversions (
            rule_id TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            category TEXT,
            stl_formula TEXT NOT NULL,
            value_sum TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def _fingerprint(payload):
    """Stable short hash of any JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cached_api_call(cache_key, fn, fingerprint_payload):
    """Cache API results keyed by (step, fingerprint of prompt+model).

    Any change to the prompt text, schema, or model invalidates the cache
    automatically — no need to manually delete files when prompts evolve.
    """
    fp = _fingerprint({"model": MODEL, "payload": fingerprint_payload})
    cache_file = CACHE_DIR / f"{cache_key}.{fp}.json"
    if cache_file.exists():
        print(f"  [cache hit: {cache_file.name}]")
        return json.loads(cache_file.read_text())
    result = fn()
    cache_file.write_text(json.dumps(result, indent=2))
    return result


def cache_step_to_db(conn, step_id, step_name, output_data):
    """Cache pipeline step results to SQLite database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO pipeline_cache (step_id, step_name, output_json)
        VALUES (?, ?, ?)
    """, (step_id, step_name, json.dumps(output_data)))
    conn.commit()


def retrieve_step_from_db(conn, step_id):
    """Retrieve cached pipeline step from SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT output_json FROM pipeline_cache WHERE step_id = ?", (step_id,))
    row = cursor.fetchone()
    return json.loads(row[0]) if row else None


# ---------------------------------------------------------------------------
# Step 1 — Extract simulation-relevant driving rules from the handbook PDF
# ---------------------------------------------------------------------------
def step1_extract_rules(file_id: str):
    """
    Ask the model to read the California Driver's Handbook and return only
    rules that are relevant to autonomous-vehicle driving simulation
    (vehicle movement, right-of-way, speed, lane usage, signaling, collision
    avoidance). Administrative rules (licensing, registration, DUI paperwork)
    are excluded at this stage.
    """
    prompt_text = (
        "You are helping build the ScenicRules autonomous-driving "
        "safety benchmark.\n\n"
        f"{SIMULATION_CONTEXT}\n\n"
        "Extract a list of driving rules from this California Driver's "
        "Handbook that can realistically be violated — and, crucially, "
        "evaluated — against a realized ego trajectory in the simulator "
        "described above. A rule is only useful if it can be computed "
        "from the observable state listed in the simulation context.\n\n"
        "Focus on rules about:\n"
        "  • Collision avoidance with vehicles, pedestrians, bicycles\n"
        "  • Near-collision risk / time-to-collision style clearance\n"
        "  • Staying inside the drivable region\n"
        "  • Correct side / correct direction of travel\n"
        "  • Lane keeping, lane centering, lane-change execution\n"
        "  • Right-of-way expressible as geometric yielding behavior\n"
        "  • Speed limits when lane speed limit is available\n"
        "  • Following distance and lateral clearance (incl. passing\n"
        "    a bicyclist)\n"
        "  • Comfort properties inferable from motion (acceleration,\n"
        "    jerk)\n\n"
        "Exclude rules whose evaluation would require information the\n"
        "simulator does not expose — in particular:\n"
        "  • Traffic light or stop sign state\n"
        "  • Turn signals, brake lights, horn, gestures, eye contact,\n"
        "    driver intent\n"
        "  • Weather, lighting, visibility, road surface\n"
        "  • Licensing, registration, paperwork, DUI, equipment,\n"
        "    inspections, parking, passenger conduct\n"
        "  • Emergency-vehicle rules that depend on lights/sirens\n"
        "    (keep only the geometric yielding component, if any)\n\n"
        "Prefer rules that are scenario-agnostic and expressible as a\n"
        "continuous violation magnitude over a trajectory. Return each\n"
        "rule as a short, precise statement."
    )
    schema = {
        "type": "object",
        "properties": {
            "rules": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["rules"],
        "additionalProperties": False,
    }

    def call():
        response = client.responses.create(
            model=MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_file", "file_id": file_id},
                ],
            }],
            text={"format": {"type": "json_schema", "name": "extracted_rules",
                              "strict": True, "schema": schema}},
        )
        return json.loads(response.output_text)

    return cached_api_call(
        "step1_extract_rules", call,
        fingerprint_payload={"prompt": prompt_text, "schema": schema, "file_id": file_id},
    )


# ---------------------------------------------------------------------------
# Step 2 — Categorize rules and assign severity
# ---------------------------------------------------------------------------
CATEGORIES = [
    "collision_avoidance",
    "right_of_way",
    "speed_control",
    "lane_discipline",
    "signaling",
    "traffic_signals",
    "pedestrian_cyclist_safety",
    "emergency_vehicle",
    "following_distance",
    "other",
]

SEVERITY_SCALE = """
Severity is rated 1-5 based on likely harm if the rule is violated at typical urban speeds:
  5 = catastrophic — high probability of fatal or life-threatening injury
  4 = severe — serious injury likely (e.g. hospitalization)
  3 = moderate — minor injury or major property damage
  2 = low — minor property damage, near-miss
  1 = minimal — traffic-flow disruption, inconvenience only
"""


def step2_categorize_and_severity(rules):
    """
    For each extracted rule, assign a category, a severity level (1-5),
    and a flag indicating whether the rule involves a *collision* outcome
    (as opposed to e.g. a traffic-flow or signaling rule).
    """
    prompt_text = (
        "You are building a rulebook for the ScenicRules autonomous-driving "
        "safety benchmark.\n\n"
        f"{SIMULATION_CONTEXT}\n\n"
        "For each driving rule below, provide:\n"
        "1. **category** — one of: " + ", ".join(CATEGORIES) + "\n"
        "2. **severity** (integer 1-5):\n" + SEVERITY_SCALE + "\n"
        "   Ground severity in the actors the benchmark actually scores\n"
        "   (ego, other vehicles, pedestrians, bicycles). Treat polygon\n"
        "   intersection between ego and another object as the definition\n"
        "   of a collision.\n"
        "3. **involves_collision** — true if violating this rule can directly "
        "cause or worsen a polygon-intersection event between ego and another "
        "in-scope object; false if the violation is procedural or purely "
        "about trajectory quality (e.g. lane centering with no nearby agent).\n"
        "4. **rationale** — one sentence explaining the severity rating.\n\n"
        f"Rules:\n{json.dumps(rules, indent=2)}"
    )
    schema = {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule": {"type": "string"},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "severity": {"type": "integer"},
                        "involves_collision": {"type": "boolean"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["rule", "category", "severity",
                                 "involves_collision", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rules"],
        "additionalProperties": False,
    }

    def call():
        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": prompt_text}],
            text={"format": {"type": "json_schema", "name": "categorized_rules",
                              "strict": True, "schema": schema}},
        )
        return json.loads(response.output_text)

    return cached_api_call(
        "step2_categorize_severity", call,
        fingerprint_payload={"prompt": prompt_text, "schema": schema},
    )


# ---------------------------------------------------------------------------
# Step 3 — Build a priority hierarchy with collision-severity granularity
# ---------------------------------------------------------------------------
def step3_build_hierarchy(categorized_rules):
    """
    Build a tiered priority hierarchy and a collision-type severity ordering.
    This is the core contribution described in the project proposal: moving
    beyond binary collision avoidance toward severity-aware prioritization.
    """
    prompt_text = (
        "You are designing a *rulebook hierarchy* for the ScenicRules\n"
        "autonomous-driving benchmark.\n\n"
        f"{SIMULATION_CONTEXT}\n\n"
        "Background (from the project proposal):\n"
        "Current AV benchmarks treat collision avoidance as binary. We want to\n"
        "introduce severity-aware rules so the vehicle can reason about harm\n"
        "tradeoffs when not all negative outcomes are avoidable. The hierarchy\n"
        "must be usable on offline realizations — conflicts between rules need\n"
        "to be expressible as geometric/kinematic situations the simulator can\n"
        "actually produce (e.g. leaving drivable area vs. striking a VRU,\n"
        "braking hard vs. rear-end exposure).\n\n"
        "Given the categorized rules below, produce:\n\n"
        "### 1. Priority tiers\n"
        "Group rules into priority tiers (tier 1 = highest). The ordering\n"
        "should reflect the ethical principle:\n"
        "  human life > serious injury > minor injury > property damage > traffic flow\n"
        "Within the safety tiers, pedestrian/cyclist safety ranks above\n"
        "vehicle-occupant safety.\n\n"
        "### 2. Collision severity ordering\n"
        "Define a severity ordering over collision *types* using crash-severity\n"
        "metrics from the literature (delta-v, occupant impact velocity, etc.):\n"
        "  • Head-on\n"
        "  • Side-impact / T-bone\n"
        "  • Rear-end\n"
        "  • Sideswipe\n"
        "  • Fixed-object\n"
        "  • Low-speed contact / scrape\n"
        "For each type give a typical delta-v range and expected harm level.\n\n"
        "### 3. Pairwise priority edges\n"
        "List concrete pairwise orderings (A > B) between specific rules,\n"
        "especially where they might conflict in a simulation scenario.\n\n"
        f"Rules:\n{json.dumps(categorized_rules, indent=2)}"
    )

    def call():
        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": prompt_text}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rule_hierarchy",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tiers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tier": {"type": "integer"},
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "rules": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "ethical_reasoning": {"type": "string"},
                                    },
                                    "required": [
                                        "tier",
                                        "name",
                                        "description",
                                        "rules",
                                        "ethical_reasoning",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "collision_severity_ordering": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "collision_type": {"type": "string"},
                                        "severity_rank": {"type": "integer"},
                                        "typical_delta_v_kmh": {"type": "string"},
                                        "expected_harm": {"type": "string"},
                                    },
                                    "required": [
                                        "collision_type",
                                        "severity_rank",
                                        "typical_delta_v_kmh",
                                        "expected_harm",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "pairwise_priority_edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "higher_priority_rule": {"type": "string"},
                                        "lower_priority_rule": {"type": "string"},
                                        "conflict_scenario": {"type": "string"},
                                        "justification": {"type": "string"},
                                    },
                                    "required": [
                                        "higher_priority_rule",
                                        "lower_priority_rule",
                                        "conflict_scenario",
                                        "justification",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "tiers",
                            "collision_severity_ordering",
                            "pairwise_priority_edges",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return json.loads(response.output_text)

    return cached_api_call(
        "step3_hierarchy", call,
        fingerprint_payload={"prompt": prompt_text},
    )


# ---------------------------------------------------------------------------
# Step 4 — Convert rules to Signal Temporal Logic (STL) format
# ---------------------------------------------------------------------------
STL_EXAMPLES = {
    "no_collisions_vru": {
        "title": "No collisions with Vulnerable Road Users (VRUs)",
        "stl_formula": "G_{[T_1, T_2]}(∀j ∈ V · (p_0(t) ∩ p_j(t) = ∅))",
        "value_sum": "Σ_{j ∈ J} E_{ego_loss} + E_{vru_j_gain}, where E is kinetic energy",
        "description": "Ensures ego vehicle never collides with pedestrians/cyclists",
    },
    "no_collisions_vehicles": {
        "title": "No collisions with other vehicles",
        "stl_formula": "G_{[T_1, T_2]}(∀i ∈ A · (p_0(t) ∩ p_i(t) = ∅))",
        "value_sum": "Σ_{i ∈ A} E_{ego_loss} + E_{vehicle_i_loss}, where E is kinetic energy",
        "description": "Ensures ego vehicle never collides with other vehicles",
    },
    "drivable_area": {
        "title": "Staying within the drivable area",
        "stl_formula": "G_{[T_1, T_2]}(p_0(t) ⊆ R_{driv}(t))",
        "value_sum": "max_t (||p_0(t) \\ R_{driv}(t)|| + d(p_0(t), R_{driv}(t))^2)",
        "description": "Ensures ego vehicle trajectory remains within drivable road region",
    },
}


def step4_convert_to_stl(categorized_rules, conn):
    """
    Convert driving rules to Signal Temporal Logic (STL) formulas.
    Each rule is mapped to an STL specification and value semantics (VS).
    Results are stored in the SQLite database.
    """
    prompt_text = (
        "You are converting driving rules into Signal Temporal Logic (STL) specifications.\n\n"
        f"{SIMULATION_CONTEXT}\n\n"
        "STL formulas and value semantics MUST be expressed purely over the\n"
        "observable signals listed above. Do not invent fields the benchmark\n"
        "does not expose (no traffic-light state, no turn signals, no intent,\n"
        "no weather, no perception confidence).\n\n"
        "Allowed symbol catalogue (map these to the observable state):\n"
        "   p_i(t)       — polygonal footprint of object i at time t\n"
        "   c_i(t)       — 2D center position of object i at time t\n"
        "   v_i(t)       — 2D velocity of object i at time t\n"
        "   a_i(t)       — acceleration of object i (derived from Δv)\n"
        "   θ_i(t)       — orientation/yaw of object i\n"
        "   type(i)      — object type (vehicle, pedestrian, bicycle)\n"
        "   R_driv       — drivable-area polygon from the map\n"
        "   L_k          — lane polygon / centerline / orientation for lane k\n"
        "   lane(i, t)   — lane occupancy of object i at time t\n"
        "   v_limit(L_k) — lane speed limit when available\n"
        "   d_poly(i, j) — polygon-to-polygon distance\n"
        "   d_cent(i, j) — center-to-center distance\n"
        "   A, V         — sets of nearby vehicles and VRUs (proximity-filtered)\n"
        "   Object 0 is always the ego vehicle.\n\n"
        "For each rule, provide:\n"
        "1. **stl_formula** — A formal STL specification using:\n"
        "   - G = always/globally operator\n"
        "   - F = eventually operator\n"
        "   - U = until operator\n"
        "   - [T_1, T_2] = time interval in seconds\n"
        "   - ∀/∃ = universal/existential quantifiers\n"
        "   - Only symbols from the catalogue above.\n\n"
        "2. **value_semantics** — A quantitative, non-negative measure of\n"
        "   violation severity, 0 when satisfied, larger when worse, so that\n"
        "   max/sum aggregation over the rollout is meaningful:\n"
        "   - For collision: kinetic energy loss (Joules)\n"
        "   - For boundary/lane: distance from valid region (meters)\n"
        "   - For clearance/TTC: shortfall below threshold (meters or seconds)\n"
        "   - For speed/comfort: excess over limit (m/s, m/s², m/s³)\n"
        "   - Formulated as a mathematical expression over the catalogue.\n\n"
        "Examples provided:\n"
        "Example 1: No collisions with VRUs\n"
        f"  STL: {STL_EXAMPLES['no_collisions_vru']['stl_formula']}\n"
        f"  VS:  {STL_EXAMPLES['no_collisions_vru']['value_sum']}\n\n"
        "Example 2: No collisions with vehicles\n"
        f"  STL: {STL_EXAMPLES['no_collisions_vehicles']['stl_formula']}\n"
        f"  VS:  {STL_EXAMPLES['no_collisions_vehicles']['value_sum']}\n\n"
        "Example 3: Staying within drivable area\n"
        f"  STL: {STL_EXAMPLES['drivable_area']['stl_formula']}\n"
        f"  VS:  {STL_EXAMPLES['drivable_area']['value_sum']}\n\n"
        f"Rules to convert:\n{json.dumps(categorized_rules, indent=2)}"
    )

    def call():
        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": prompt_text}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "stl_conversions",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "conversions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "rule": {"type": "string"},
                                        "category": {"type": "string"},
                                        "stl_formula": {"type": "string"},
                                        "value_semantics": {"type": "string"},
                                    },
                                    "required": [
                                        "rule",
                                        "category",
                                        "stl_formula",
                                        "value_semantics",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["conversions"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return json.loads(response.output_text)

    result = cached_api_call(
        "step4_convert_to_stl", call,
        fingerprint_payload={"prompt": prompt_text},
    )

    # Store STL conversions in database
    cursor = conn.cursor()
    for idx, conversion in enumerate(result.get("conversions", [])):
        rule_id = f"stl_{idx}_{conversion.get('category', 'unknown')}"
        cursor.execute("""
            INSERT OR REPLACE INTO stl_conversions
            (rule_id, rule_text, category, stl_formula, value_sum, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            rule_id,
            conversion.get("rule", ""),
            conversion.get("category", ""),
            conversion.get("stl_formula", ""),
            conversion.get("value_semantics", ""),
            conversion.get("notes", ""),
        ))
    conn.commit()

    return result


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def print_section(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_summary(extracted_rules, categorized, hierarchy, stl_conversions=None):
    rules = categorized["rules"]
    collision_rules = [r for r in rules if r["involves_collision"]]

    print_section("PIPELINE RESULTS")

    print(f"\nTotal rules extracted from PDF : {len(extracted_rules)}")
    print(f"Rules categorized             : {len(rules)}")
    print(f"Collision-involved rules      : {len(collision_rules)}")
    if stl_conversions:
        stl_count = len(stl_conversions.get("conversions", []))
        print(f"Rules converted to STL        : {stl_count}")

    print("\n--- Category distribution ---")
    for cat, count in Counter(r["category"] for r in rules).most_common():
        print(f"  {cat:30s} {count:3d}")

    print("\n--- Severity distribution ---")
    for sev, count in sorted(Counter(r["severity"] for r in rules).items()):
        label = {1: "minimal", 2: "low", 3: "moderate", 4: "severe", 5: "catastrophic"}
        print(f"  {sev} ({label.get(sev, '?'):12s}): {count:3d}")

    print_section("PRIORITY TIERS")
    for tier in hierarchy["tiers"]:
        print(f"\n  Tier {tier['tier']}: {tier['name']}")
        print(f"  {tier['description']}")
        print(f"  Ethical basis: {tier['ethical_reasoning']}")
        n = len(tier["rules"])
        for rule in tier["rules"][:5]:
            print(f"    • {rule}")
        if n > 5:
            print(f"    … and {n - 5} more")

    print_section("COLLISION SEVERITY ORDERING")
    for cs in hierarchy["collision_severity_ordering"]:
        print(f"  {cs['severity_rank']}. {cs['collision_type']}")
        print(f"     Typical Δv : {cs['typical_delta_v_kmh']}")
        print(f"     Harm       : {cs['expected_harm']}")

    print_section("PAIRWISE PRIORITY EDGES (sample)")
    edges = hierarchy["pairwise_priority_edges"]
    for edge in edges[:8]:
        print(f"  {edge['higher_priority_rule']}")
        print(f"    > {edge['lower_priority_rule']}")
        print(f"    Scenario: {edge['conflict_scenario']}")
        print()
    if len(edges) > 8:
        print(f"  … {len(edges) - 8} more edges in full output")

    if stl_conversions:
        print_section("STL CONVERSION EXAMPLES")
        conversions = stl_conversions.get("conversions", [])
        for i, conv in enumerate(conversions[:5]):
            print(f"\n  [{i+1}] {conv.get('rule', 'N/A')[:60]}")
            print(f"      STL: {conv.get('stl_formula', 'N/A')}")
            print(f"      VS:  {conv.get('value_semantics', 'N/A')[:60]}")
        if len(conversions) > 5:
            print(f"\n  … {len(conversions) - 5} more STL conversions in full output")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Rulebook Generator — Proof-of-Concept Pipeline")
    print("=" * 70)

    # Initialize SQLite database
    conn = init_database()
    print("\n✓ Database initialized at", DB_PATH)

    print("\n[0/4] Ensuring handbook PDF is uploaded …")
    file_id = ensure_file_uploaded(PDF_PATH)

    print("\n[1/4] Extracting simulation-relevant rules from PDF …")
    extracted = step1_extract_rules(file_id)
    rules = extracted["rules"]
    cache_step_to_db(conn, "step1_extract_rules", "Extract Rules", extracted)
    print(f"  → {len(rules)} rules extracted")

    print("\n[2/4] Categorizing rules and assessing severity …")
    categorized = step2_categorize_and_severity(rules)
    cat_rules = categorized["rules"]
    cache_step_to_db(conn, "step2_categorize_severity", "Categorize & Severity", categorized)
    collision_count = sum(1 for r in cat_rules if r["involves_collision"])
    print(f"  → {len(cat_rules)} rules categorized, {collision_count} collision-involved")

    print("\n[3/4] Building priority hierarchy …")
    hierarchy = step3_build_hierarchy(cat_rules)
    cache_step_to_db(conn, "step3_build_hierarchy", "Build Hierarchy", hierarchy)
    print(f"  → {len(hierarchy['tiers'])} tiers, "
          f"{len(hierarchy['collision_severity_ordering'])} collision types, "
          f"{len(hierarchy['pairwise_priority_edges'])} pairwise edges")

    print("\n[4/4] Converting rules to Signal Temporal Logic (STL) …")
    stl_conversions = step4_convert_to_stl(cat_rules, conn)
    cache_step_to_db(conn, "step4_stl_conversions", "STL Conversions", stl_conversions)
    stl_count = len(stl_conversions.get("conversions", []))
    print(f"  → {stl_count} rules converted to STL formulas")

    print_summary(rules, categorized, hierarchy, stl_conversions)

    output_path = Path("rulebook_output.json")
    full_output = {
        "metadata": {
            "source": "California Driver's Handbook",
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "database_path": str(DB_PATH),
            "pipeline_steps": [
                "extract_rules",
                "categorize_and_severity",
                "build_hierarchy",
                "convert_to_stl",
            ],
        },
        "extracted_rules": rules,
        "categorized_rules": cat_rules,
        "hierarchy": hierarchy,
        "stl_conversions": stl_conversions,
        "stl_examples": STL_EXAMPLES,
    }
    output_path.write_text(json.dumps(full_output, indent=2))
    print(f"\n✓ Full structured output saved to {output_path}")
    print(f"✓ All pipeline steps cached to {DB_PATH}")

    # Display example STL formulas
    print_section("STL EXAMPLE FORMULAS")
    for _, example in STL_EXAMPLES.items():
        print(f"\n  {example['title']}")
        print(f"  STL: {example['stl_formula']}")
        print(f"  VS:  {example['value_sum']}")

    conn.close()


if __name__ == "__main__":
    main()
