import argparse
import concurrent.futures
import os
import re
import json
import shutil
import subprocess
import threading
from pathlib import Path
from collections import Counter
from datetime import datetime

import openai
from git import GitCommandError, Repo
from database import (
    cached_api_call as db_cached_api_call,
    cache_step_to_db,
    init_database,
    retrieve_json_artifact,
    store_json_artifact,
    store_rule_scenario_manifest as db_store_rule_scenario_manifest,
)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_dotenv()

client = None


def openai_client() -> openai.Client:
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for PDF/rule generation.")
        client = openai.Client(api_key=api_key)
    return client

MODEL = "gpt-5.4"
IMPLEMENTATION_HARNESS = "pi"
IMPLEMENTATION_PROVIDER = "deepseek"
IMPLEMENTATION_MODEL = "deepseek-v4-flash"
IMPLEMENTATION_REASONING = "high"
PI_STREAM_EXTENSION_PATH = Path(".pi/extensions/stream-output/index.ts")
AGENT_DOCKER_IMAGE = "rulebook-generator-agents:latest"
AGENT_DOCKERFILE = Path("Dockerfile.rulebook-agents")
AGENT_WORKSPACE_ROOT = Path(".rulegen_agent_workspaces")
AGENT_DOCKER_CTX = Path(".rulegen-docker-ctx")
AGENT_LOG_DIR = Path("agent_logs")
PDF_PATH = Path("california-drivers-handbook.pdf")
DB_PATH = Path("rulebook_pipeline.db")
SCENIC_RULES_PATH = Path("/Users/jogramnaestjernshaugen/ScenicRules")
RULE_ID_OFFSET = 100  # generated rules start at ID 100 to avoid conflicts


def ensure_file_uploaded(pdf_path: Path, conn) -> str:
    """Return an OpenAI file_id for the handbook PDF.

    Uploads the PDF on first run and caches the resulting id in SQLite. On
    subsequent runs the cached id is reused, unless the file has been deleted
    server-side — in which case we re-upload.
    """
    cached_artifact = retrieve_json_artifact(conn, "openai_file_id")
    cached = cached_artifact.get("file_id") if cached_artifact else None
    if cached:
        try:
            openai_client().files.retrieve(cached)
            print(f"  [file_id cache hit: {cached}]")
            return cached
        except Exception as e:
            print(f"  [cached file_id {cached} unusable ({e}); re-uploading]")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    print(f"  [uploading {pdf_path} to OpenAI…]")
    with pdf_path.open("rb") as f:
        uploaded = openai_client().files.create(file=f, purpose="user_data")
    store_json_artifact(conn, "openai_file_id", {
        "file_id": uploaded.id,
        "pdf_path": str(pdf_path),
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"  [uploaded, file_id={uploaded.id}]")
    return uploaded.id


# ---------------------------------------------------------------------------
# Shared simulation context — reused across prompts so the model understands
# what the ScenicRules benchmark can actually observe and score.
# ---------------------------------------------------------------------------
SIMULATION_CONTEXT = """
ScenicRules benchmark — offline trajectory evaluation model:

This is an OFFLINE rulebook benchmark. Rules inspect a completed, pre-recorded
trajectory — they do not control the ego vehicle and do not run during planning.
A scenario produces a finite discrete sequence of world states (timesteps). Each
rule is evaluated once per timestep and returns a numeric score.

Rule function shape:

    def my_rule(handler, step, ...params):
        pool = handler(step)
        ego_state = pool.ego_state
        # ... inspect state ...
        return violation_score   # 0.0 if satisfied, positive if violated

The function must return 0 when the rule is satisfied at that step and a
positive value when violated. Violation magnitude should represent severity.
A Rule object then aggregates per-step scores over the trajectory using either:
  • max  — worst single-step violation (use for "did this ever happen?")
  • sum  — cumulative total (use for repeated infractions or sustained discomfort)

Available signals via pool = handler(step):

  Ego:
    pool.ego_state.position          — 2D numpy vector (x, y)
    pool.ego_state.velocity          — 2D numpy vector
    pool.ego_state.yaw               — orientation in radians
    pool.ego_state.polygon           — Shapely polygon footprint
    pool.ego_state.acceleration      — 2D numpy vector (finite difference of velocity)
    pool.ego_state.length / .width   — object dimensions

  Other actors:
    pool.vehicles_in_proximity       — cars/trucks within a radial threshold (PROXIMITY-FILTERED)
    pool.vrus_in_proximity           — pedestrians/bicycles within a radial threshold (PROXIMITY-FILTERED)
    pool.other_vehicle_states        — ALL other cars/trucks in the scenario (not filtered)
    pool.vru_states                  — ALL pedestrians/bicycles in the scenario (not filtered)

  Each actor state has: position, velocity, yaw, polygon, acceleration, length, width,
  and object_type ("Car", "Truck", "Pedestrian", or "Bicycle").

  Proximity warning: vehicles_in_proximity and vrus_in_proximity are filtered by
  radial distance + object radii. Use other_vehicle_states or vru_states for rules
  that must consider all actors regardless of distance.

  Road network (from Scenic):
    Drivable region polygon(s)
    Lane polygons, lane centerlines, lane orientations
    Lane speed limits (when available)
    Lane maneuvers (allowed next lanes / turn types)
    Intersection polygons
    Inferred lane membership for each object state
    correct_lanes   — lanes whose direction matches ego heading
    incorrect_lanes — lanes whose direction opposes ego heading

  NOT in the road network — these map features do not exist in the benchmark:
    • Crosswalk / pedestrian crossing polygons or markers
    • Stop-line geometry
    • Yield-line geometry
    • Bike-lane polygons or markings
    • Bus-stop or transit-zone polygons
    • Painted median or exclusion-zone polygons
    • Sidewalk polygons
    • Railroad crossing geometry
    • Posted sign locations or sign state
    Do not write rules that rely on any of the above. A pedestrian in the
    scenario is just an actor with a position and velocity — there is no
    map annotation indicating that it is "in a crosswalk".

  Ego trajectory:
    Ego trajectory as a Shapely LineString (full realized path)
    Future trajectory segment (from current step onward)
    Past trajectory segment (up to current step)
    Buffered trajectory variants based on ego width

  Geometry and kinematics:
    Collision = polygon intersection (Shapely)
    Distance = polygon-to-polygon Shapely distance (0 when touching/overlapping)
    Acceleration derived from consecutive velocity differences
    Jerk = ||a(t) − a(t−Δt)|| / Δt; time delta = 0.1 seconds

Time model: discrete 0.1-second steps; no sensor latency, uncertainty, or
perception error unless encoded separately in the trajectory or map.

NOT available — do not write rules that require:
  • Traffic light or stop sign state
  • Turn signals, brake lights, horn, eye contact, gestures, driver intent
  • Raw sensor data, occlusion, perception uncertainty, detection confidence
  • Weather, lighting, road surface, visibility
  • Legal doctrines or social norms not reducible to map + trajectory
  • Internal policy/network state of the driving agent
  • Fault attribution or what other agents "should have" intended
  • Continuous-time dynamics beyond finite-difference kinematics and polygon TTC
  • Any map feature not listed under "Road network" above — in particular:
    crosswalks, stop lines, bike lanes, sidewalks, safety zones, railroad
    crossings, bus stops, painted exclusion zones, or posted sign locations

Rule-writing guidance:
  • Prefer rules that produce a continuous numeric violation severity (not binary).
  • For each rule consider: Which actors are relevant? All or only proximity-filtered?
    Is it instantaneous, cumulative, or window-based? What threshold defines a
    violation? Should the trajectory score be max or sum?
  • Rules about "did this ever happen" → max aggregation.
  • Rules about "how much total discomfort / how many infractions" → sum aggregation.
  • Intersections may degrade lane-assignment accuracy; consider relaxing lane-based
    rules inside intersection polygons.
""".strip()


def cached_api_call(cache_key, fn, fingerprint_payload):
    """Cache API results keyed by (step, fingerprint of prompt+model).

    Any change to the prompt text, schema, or model invalidates the cache
    automatically — no need to manually delete files when prompts evolve.
    """
    return db_cached_api_call(DB_PATH, MODEL, cache_key, fn, fingerprint_payload)


def store_rule_scenario_manifest(conn, manifest: dict) -> None:
    db_store_rule_scenario_manifest(
        conn,
        manifest,
        SCENIC_RULES_PATH,
        IMPLEMENTATION_HARNESS,
        IMPLEMENTATION_PROVIDER,
        IMPLEMENTATION_MODEL,
        IMPLEMENTATION_REASONING,
    )


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
        "evaluated — by a Python function that inspects a completed ego "
        "trajectory step-by-step using the signals listed in the simulation "
        "context above. A rule is only useful if every value it needs is "
        "directly available from those signals.\n\n"
        "Focus on rules about:\n"
        "  • Collision avoidance with vehicles, pedestrians, bicycles\n"
        "    (evaluated via polygon intersection)\n"
        "  • Near-collision risk / time-to-collision with other actors\n"
        "  • Staying inside the drivable region polygon\n"
        "  • Correct side / correct direction of travel\n"
        "    (use correct_lanes / incorrect_lanes)\n"
        "  • Lane keeping, lane centering, lane-change execution\n"
        "  • Right-of-way expressible as geometric yielding behavior\n"
        "    (approach geometry, TTC, gap to conflicting actors)\n"
        "  • Speed limits when lane speed limit is available\n"
        "  • Following distance and lateral clearance (incl. passing\n"
        "    a bicyclist within the same or adjacent lane)\n"
        "  • Comfort properties inferable from motion (acceleration,\n"
        "    jerk, lateral acceleration)\n\n"
        "Exclude rules whose evaluation would require information not\n"
        "in the simulation context — in particular:\n"
        "  • Traffic light, stop sign, or yield sign state\n"
        "  • Turn signals, brake lights, horn, gestures, eye contact,\n"
        "    driver intent or attention\n"
        "  • Weather, lighting, visibility, road surface friction\n"
        "  • Licensing, registration, paperwork, DUI, equipment,\n"
        "    inspections, parking, passenger conduct\n"
        "  • Emergency-vehicle rules that depend on lights/sirens\n"
        "    (keep only the geometric yielding component, if any)\n"
        "  • Any rule requiring perception, occlusion reasoning, or\n"
        "    prediction of hidden agents\n\n"
        "Prefer rules that are scenario-agnostic and produce a continuous\n"
        "violation magnitude (e.g. clearance shortfall in metres, speed\n"
        "excess in m/s) rather than a binary pass/fail. Return each rule\n"
        "as a short, precise statement in the third person."
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
        response = openai_client().responses.create(
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
        "4. **aggregation** — how per-step scores should be combined over the "
        "trajectory: 'max' when the rule captures a worst-case event that only "
        "needs to occur once (e.g. collision, lane boundary exceedance), or "
        "'sum' when the rule captures cumulative or repeated infractions "
        "(e.g. sustained discomfort, repeated weaving, prolonged slow speed).\n"
        "5. **rationale** — one sentence explaining the severity rating and "
        "aggregation choice.\n\n"
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
                        "aggregation": {"type": "string", "enum": ["max", "sum"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["rule", "category", "severity",
                                 "involves_collision", "aggregation", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rules"],
        "additionalProperties": False,
    }

    def call():
        response = openai_client().responses.create(
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
        response = openai_client().responses.create(
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
        "Symbol catalogue — use ONLY these symbols:\n"
        "   p_0(t)           — ego polygon footprint (Shapely)\n"
        "   p_j(t)           — polygon footprint of actor j\n"
        "   c_0(t), c_j(t)  — 2D center positions\n"
        "   v_0(t), v_j(t)  — 2D velocity vectors\n"
        "   a_0(t), a_j(t)  — acceleration vectors (finite diff of v)\n"
        "   a_long,0(t)      — longitudinal acceleration (along heading)\n"
        "   a_lat,0(t)       — lateral acceleration (perpendicular to heading)\n"
        "   jerk_0(t)        — ||a_0(t)−a_0(t−Δt)||/Δt, Δt=0.1 s\n"
        "   θ_0(t)           — ego yaw (radians)\n"
        "   type(j)          — 'Car', 'Truck', 'Pedestrian', or 'Bicycle'\n"
        "   R_driv           — drivable-area polygon from the map\n"
        "   L_k              — lane polygon / centerline / orientation for lane k\n"
        "   lane(i,t)        — lane polygon occupied by object i at time t\n"
        "   correct_lanes(t) — lanes whose direction matches ego heading\n"
        "   incorrect_lanes(t) — lanes whose direction opposes ego heading\n"
        "   v_limit(L_k)     — speed limit for lane k (when available)\n"
        "   d_poly(i,j,t)    — Shapely polygon-to-polygon distance (0 when touching)\n"
        "   d_cent(i,j,t)    — center-to-center Euclidean distance\n"
        "   traj_0           — ego trajectory LineString (full realized path)\n"
        "   traj_future_0(t) — ego trajectory from step t onward\n"
        "   traj_past_0(t)   — ego trajectory up to step t\n"
        "   A_prox, V_prox   — proximity-filtered vehicles / VRUs\n"
        "   A_all, V_all     — ALL other vehicles / ALL VRUs (not filtered)\n"
        "   Object 0 is always the ego vehicle.\n\n"
        "Use A_all / V_all when a rule must consider every actor in the scene.\n"
        "Use A_prox / V_prox only when the rule is explicitly local.\n\n"
        "For each rule, provide:\n"
        "1. **stl_formula** — A formal STL specification using:\n"
        "   - G = always/globally operator\n"
        "   - F = eventually operator\n"
        "   - U = until operator\n"
        "   - [T_1, T_2] = time interval in seconds\n"
        "   - ∀/∃ = universal/existential quantifiers\n"
        "   - Only symbols from the catalogue above.\n\n"
        "2. **value_semantics** — A quantitative, non-negative per-step measure\n"
        "   of violation severity (0 when satisfied, larger when worse):\n"
        "   - Collision: kinetic energy loss in Joules (½mΔv²)\n"
        "   - Boundary/lane: area of ego polygon outside valid region (m²)\n"
        "     or signed distance from valid region (m)\n"
        "   - Clearance/TTC: shortfall below threshold (meters or seconds)\n"
        "   - Speed/comfort: excess over limit (m/s, m/s², or m/s³)\n"
        "   State whether the trajectory score should be max or sum of\n"
        "   per-step values, consistent with the aggregation field.\n\n"
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
        response = openai_client().responses.create(
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
# Step 5 — Implement each rule in ScenicRules on its own branch via Claude Code
# ---------------------------------------------------------------------------

def _rule_slug(text: str, max_len: int = 35) -> str:
    """Lowercase underscore slug from rule text, skipping stop-words."""
    stop = {
        "the", "driver", "does", "not", "a", "an", "or", "and", "when",
        "is", "in", "to", "of", "that", "with", "without", "for", "on",
        "if", "by", "at", "from", "into", "it", "its", "are", "be", "been",
        "so", "do", "no", "any", "too",
    }
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    words = [w for w in words if w not in stop]
    return "_".join(words)[:max_len].rstrip("_")


def _docker_env_args() -> list[str]:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for rule implementation.")
    return [
        "-e", "DEEPSEEK_API_KEY",
        "-e", "FORCE_COLOR=1",
        "-e", "TERM=xterm-256color",
    ]


def _ensure_agent_docker_image() -> None:
    """Build the Docker image used by implementation workers when needed."""
    if not AGENT_DOCKERFILE.exists():
        raise RuntimeError(f"Missing {AGENT_DOCKERFILE}; cannot run Docker isolation.")
    if not SCENIC_RULES_PATH.exists():
        raise RuntimeError(
            f"ScenicRules repo not found at {SCENIC_RULES_PATH}; "
            "update SCENIC_RULES_PATH in main.py."
        )
    inspect = subprocess.run(
        ["docker", "image", "inspect", AGENT_DOCKER_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if inspect.returncode == 0:
        return

    # Prepare a fixed, inspectable build context directory.
    ctx = AGENT_DOCKER_CTX
    ctx.mkdir(parents=True, exist_ok=True)

    shutil.copy2(AGENT_DOCKERFILE, ctx / AGENT_DOCKERFILE.name)

    (ctx / "docker").mkdir(parents=True, exist_ok=True)
    for docker_file in Path("docker").iterdir():
        if docker_file.is_file():
            shutil.copy2(docker_file, ctx / "docker" / docker_file.name)

    scenicrules_dst = ctx / "scenicrules"
    if scenicrules_dst.exists():
        shutil.rmtree(scenicrules_dst)
    print(f"  [docker] syncing {SCENIC_RULES_PATH} → {scenicrules_dst}", flush=True)
    shutil.copytree(
        SCENIC_RULES_PATH,
        scenicrules_dst,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".DS_Store",
        ),
    )
    print(
        f"  [docker] building {AGENT_DOCKER_IMAGE} "
        f"from context {ctx.resolve()}",
        flush=True,
    )
    result = subprocess.run(
        [
            "docker", "build",
            "-f", AGENT_DOCKERFILE.name,
            "-t", AGENT_DOCKER_IMAGE,
            ".",
        ],
        cwd=ctx,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker build failed (exit {result.returncode}); "
            f"inspect context at {ctx.resolve()}"
        )


def _invoke_deepseek_code_in_docker(
    prompt: str,
    workspace_repo: Path,
    log_path: Path,
    log_prefix: str = "         ",
) -> tuple[bool, str]:
    """
    Run Pi with DeepSeek inside Docker, with only one clone mounted read/write.
    Returns (success, output_text).
    """
    if not PI_STREAM_EXTENSION_PATH.exists():
        raise RuntimeError(f"Pi stream-output extension not found at {PI_STREAM_EXTENSION_PATH}")

    project_root = Path.cwd().resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "-v", f"{workspace_repo.resolve()}:/workspace:rw",
        "-v", f"{project_root.resolve()}:/rulebookgen:ro",
        "-w", "/workspace",
        "-e", "PI_CODING_AGENT_DIR=/tmp/pi-agent",
        "-e", "GIT_CONFIG_COUNT=1",
        "-e", "GIT_CONFIG_KEY_0=safe.directory",
        "-e", "GIT_CONFIG_VALUE_0=/workspace",
        *_docker_env_args(),
        AGENT_DOCKER_IMAGE,
        "pi",
        "--provider",
        IMPLEMENTATION_PROVIDER,
        "--model",
        IMPLEMENTATION_MODEL,
        "--thinking",
        IMPLEMENTATION_REASONING,
        "--extension",
        "/rulebookgen/.pi/extensions/stream-output/index.ts",
        "--stream=all",
        "--print",
        "--no-session",
        "--tools",
        "read,bash,edit,write,grep,find,ls",
        "--verbose",
        prompt,
    ]

    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        log_file.write("\n[command] " + " ".join(command[:-1]) + " <prompt omitted>\n\n")
    print(
        f"{log_prefix}[docker/pi] provider={IMPLEMENTATION_PROVIDER} "
        f"model={IMPLEMENTATION_MODEL} reasoning={IMPLEMENTATION_REASONING}"
    )
    try:
        assert process.stdout is not None
        with log_path.open("a") as log_file:
            for raw_line in process.stdout:
                log_file.write(raw_line)
                log_file.flush()
                line = raw_line.rstrip()
                if not line:
                    continue
                output_lines.append(line)
                print(f"{log_prefix}  [pi] {line}")
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    return_code = process.wait()
    with log_path.open("a") as log_file:
        log_file.write(f"\n[exit_code] {return_code}\n")
    return return_code == 0, "\n".join(output_lines)


def _branch_has_commits_ahead_of_main(branch_name: str) -> bool:
    """True if branch exists and has at least one commit not on main."""
    if not _branch_exists(branch_name):
        return False
    repo = Repo(SCENIC_RULES_PATH)
    try:
        count = repo.git.rev_list("--count", f"main..{branch_name}")
    except GitCommandError:
        return False
    return count.strip() not in ("", "0")


def _branch_exists(branch_name: str) -> bool:
    """True if a local branch exists."""
    repo = Repo(SCENIC_RULES_PATH)
    return any(head.name == branch_name for head in repo.heads)


def _prepare_rule_workspace(workspace_repo: Path, branch_name: str) -> tuple[bool, str]:
    """Create a fresh clone for one implementation worker."""
    if workspace_repo.exists():
        shutil.rmtree(workspace_repo)
    workspace_repo.parent.mkdir(parents=True, exist_ok=True)
    try:
        repo = Repo.clone_from(str(SCENIC_RULES_PATH), str(workspace_repo))
        repo.git.fetch("origin", "refs/heads/*:refs/remotes/origin/*")
        start_point = f"origin/{branch_name}" if _branch_exists(branch_name) else "origin/main"
        repo.git.checkout("-B", branch_name, start_point)
        return True, ""
    except GitCommandError as exc:
        return False, (exc.stderr or str(exc)).strip()


def _fetch_workspace_branch_into_repo(workspace_repo: Path, branch_name: str) -> None:
    """Copy the worker branch from its clone back into the ScenicRules repo."""
    repo = Repo(SCENIC_RULES_PATH)
    repo.git.fetch(
        "--force",
        str(workspace_repo.resolve()),
        f"{branch_name}:refs/heads/{branch_name}",
    )


def _commit_rule_workspace(
    workspace_repo: Path,
    category: str,
    idx: int,
    slug: str,
    severity: int,
    aggregation: str,
    rule_text: str,
) -> str:
    repo = Repo(workspace_repo)
    repo.git.add(
        "src/rulebook_benchmark/generated/",
        "src/evaluation/run_evaluation.py",
        "src/evaluation/assets/with_vru.graph",
        "src/evaluation/assets/no_vru.graph",
    )
    commit_msg = (
        f"rule({category}): implement rule_{idx:03d}_{slug}\n\n"
        f"Severity {severity}/5, aggregation={aggregation}\n"
        f"Rule: {rule_text[:120]}"
    )
    try:
        repo.git.commit("-m", commit_msg)
    except GitCommandError:
        return "empty"
    return "committed"
def _write_rule_scenario_manifest(entries: list[dict], conn=None) -> None:
    """Persist a machine-readable branch-to-scenario manifest in SQLite."""
    manifest = {
        "scenic_rules_repo": str(SCENIC_RULES_PATH),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "implementation_harness": IMPLEMENTATION_HARNESS,
        "implementation_provider": IMPLEMENTATION_PROVIDER,
        "implementation_model": IMPLEMENTATION_MODEL,
        "implementation_reasoning": IMPLEMENTATION_REASONING,
        "entries": entries,
    }
    if conn is not None:
        store_rule_scenario_manifest(conn, manifest)


def _branch_has_active_registration(branch_name: str, rule_id: int) -> bool:
    """True when the branch contains both implementation and active registration."""
    if not _branch_has_commits_ahead_of_main(branch_name):
        return False

    checks = [
        ("grep", "-q", f"rule_{rule_id}", f"{branch_name}:src/evaluation/run_evaluation.py"),
        ("grep", "-q", f"^{rule_id}$", f"{branch_name}:src/evaluation/assets/with_vru.graph"),
    ]
    repo = Repo(SCENIC_RULES_PATH)
    for command, *args in checks:
        try:
            getattr(repo.git, command)(*args)
        except GitCommandError:
            return False
    return True


def _build_implementation_prompt(
    *,
    idx: int,
    rule: dict,
    rule_id: int,
    slug: str,
    stl_info: dict,
) -> str:
    rule_text = rule["rule"]
    category = rule["category"]
    aggregation = rule["aggregation"]
    severity = rule["severity"]
    stl_formula = stl_info.get("stl_formula", "N/A")
    value_semantics = stl_info.get("value_semantics", "N/A")
    return (
        "You are working in the ScenicRules benchmark repository. Implement only "
        "this one generated rule and make no unrelated changes.\n\n"
        "First read src/rulebook_benchmark/rule_functions.py, "
        "src/evaluation/run_evaluation.py, src/evaluation/cfgs/eval.yaml, and the "
        "active graph file under src/evaluation/assets/.\n\n"
        f"Create or update src/rulebook_benchmark/generated/rule_{idx:03d}_{slug}.py "
        "and src/rulebook_benchmark/generated/__init__.py.\n\n"
        f"Rule text: {rule_text}\n"
        f"Category: {category}\n"
        f"Severity: {severity}/5\n"
        f"Aggregation: {aggregation}\n"
        f"STL formula: {stl_formula}\n"
        f"Value semantics: {value_semantics}\n"
        f"Rule ID: {rule_id}\n\n"
        "The generated Python file must define one rule function with signature "
        "(handler, step, **params). It must return 0.0 when satisfied and a positive "
        "float when violated. Define a Rule object named "
        f"rule_{rule_id} = Rule(fn, {aggregation}, 'generated_{idx:03d}_{slug}', "
        f"{rule_id}, **params). Use the existing Rule API and helper style.\n\n"
        "Register the generated rule so it is active immediately: import it in "
        "src/evaluation/run_evaluation.py, add it to ruleset, add it to "
        "rule_id_to_rule, and update the active graph file so the #rules section "
        f"includes {rule_id}. Add acyclic lowest-priority edges by connecting current "
        f"terminal graph nodes to {rule_id}. For non-VRU rules, also update "
        "no_vru.graph if appropriate.\n\n"
        "Do not create or modify Scenic scenarios in this step. Scenic scenario "
        "generation is handled outside the agent pipeline.\n\n"
        "Before finishing, run "
        f"`python3 -m py_compile src/evaluation/run_evaluation.py "
        f"src/rulebook_benchmark/generated/rule_{idx:03d}_{slug}.py` and fix errors. "
        "If a one-sample evaluation is feasible using an existing scenario, run it "
        "and fix syntax/import failures."
    )


def step5_implement_rules(
    categorized_rules: list,
    stl_conversions: dict,
    conn,
    parallel_workers: int = 1,
) -> list:
    """
    For each categorized rule, create a git branch in the ScenicRules repo and
    invoke Pi with DeepSeek to implement the rule by reading the real
    codebase and writing the implementation file directly.

    Caching: a branch that already has commits ahead of main is skipped.
    """
    stl_by_rule = {c["rule"]: c for c in stl_conversions.get("conversions", [])}
    n = len(categorized_rules)
    if parallel_workers < 1:
        raise ValueError("--parallel-workers must be at least 1")
    worker_count = min(parallel_workers, n) if n else 1
    results_by_idx: dict[int, dict] = {}
    fetch_lock = threading.Lock()

    print(f"\n  Target repo   : {SCENIC_RULES_PATH}")
    print(f"  Workspace dir : {AGENT_WORKSPACE_ROOT}")
    print(f"  Agent logs    : {AGENT_LOG_DIR}")
    print(f"  Docker image  : {AGENT_DOCKER_IMAGE}")
    print(f"  Rules to impl : {n}")
    print(f"  Parallel jobs : {worker_count}")
    print(f"  OpenAI model  : {MODEL}")
    print(f"  Impl harness  : {IMPLEMENTATION_HARNESS}")
    print(f"  Impl provider : {IMPLEMENTATION_PROVIDER}")
    print(f"  Impl model    : {IMPLEMENTATION_MODEL}")
    print(f"  Reasoning     : {IMPLEMENTATION_REASONING}")
    print()
    AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for old_log in AGENT_LOG_DIR.glob("*.log"):
        old_log.unlink()
    try:
        _ensure_agent_docker_image()
    except Exception as exc:
        (AGENT_LOG_DIR / "pipeline.log").write_text(
            "[status] setup_error\n"
            f"[docker_image] {AGENT_DOCKER_IMAGE}\n"
            f"[error] {exc}\n"
        )
        raise
    AGENT_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    def worker(idx: int, rule: dict) -> tuple[int, dict]:
        rule_text = rule["rule"]
        category = rule["category"]
        aggregation = rule["aggregation"]
        severity = rule["severity"]
        rule_id = RULE_ID_OFFSET + idx
        slug = _rule_slug(rule_text)
        branch_name = f"rule/{idx:03d}-{category}-{slug}"[:80]
        log_path = AGENT_LOG_DIR / f"{idx:03d}-{slug}.log"
        log_prefix = f"  [{idx + 1:>2}/{n}] "

        display = rule_text[:70] + "…" if len(rule_text) > 70 else rule_text
        print(f"{log_prefix}{display}", flush=True)
        print(f"{log_prefix}category={category} severity={severity}/5 aggregation={aggregation}", flush=True)

        if _branch_has_active_registration(branch_name, rule_id):
            print(f"{log_prefix}[cache hit] {branch_name}", flush=True)
            return idx, {
                "rule": rule_text, "category": category,
                "branch": branch_name,
                "status": "cached",
                "agent_log_path": "",
            }

        log_path.write_text(
            "\n".join([
                f"[rule_index] {idx}",
                f"[rule_id] {rule_id}",
                f"[rule] {rule_text}",
                f"[category] {category}",
                f"[severity] {severity}",
                f"[aggregation] {aggregation}",
                f"[branch] {branch_name}",
                f"[model] {IMPLEMENTATION_PROVIDER}/{IMPLEMENTATION_MODEL}",
                f"[reasoning] {IMPLEMENTATION_REASONING}",
                "[status] starting",
                "",
            ]),
        )

        workspace_repo = AGENT_WORKSPACE_ROOT / f"{idx:03d}-{slug}" / "repo"
        ok, err = _prepare_rule_workspace(workspace_repo, branch_name)
        if not ok:
            print(f"{log_prefix}[ERROR preparing workspace: {err}]", flush=True)
            with log_path.open("a") as log_file:
                log_file.write(f"\n[status] workspace_prepare_error\n[error] {err}\n")
            return idx, {
                "rule": rule_text, "category": category,
                "branch": branch_name,
                "status": "error",
                "agent_log_path": str(log_path),
            }

        prompt = _build_implementation_prompt(
            idx=idx,
            rule=rule,
            rule_id=rule_id,
            slug=slug,
            stl_info=stl_by_rule.get(rule_text, {}),
        )

        print(f"{log_prefix}[invoking Pi/DeepSeek in Docker]", flush=True)
        success, _ = _invoke_deepseek_code_in_docker(
            prompt,
            workspace_repo,
            log_path,
            log_prefix,
        )

        if not success:
            print(f"{log_prefix}[ERROR: Pi/DeepSeek exited non-zero]", flush=True)
            return idx, {
                "rule": rule_text, "category": category,
                "branch": branch_name,
                "status": "error",
                "agent_log_path": str(log_path),
            }

        try:
            status = _commit_rule_workspace(
                workspace_repo,
                category,
                idx,
                slug,
                severity,
                aggregation,
                rule_text,
            )
            if status == "committed":
                with fetch_lock:
                    _fetch_workspace_branch_into_repo(workspace_repo, branch_name)
        except GitCommandError as exc:
            err = exc.stderr or str(exc)
            print(f"{log_prefix}[ERROR committing/fetching branch: {err}]", flush=True)
            with log_path.open("a") as log_file:
                log_file.write(f"\n[status] commit_or_fetch_error\n[error] {err}\n")
            status = "error"

        print(f"{log_prefix}[{status}] branch={branch_name}", flush=True)
        with log_path.open("a") as log_file:
            log_file.write(f"\n[status] {status}\n")
        return idx, {
            "rule": rule_text, "category": category,
            "branch": branch_name,
            "status": status,
            "agent_log_path": str(log_path),
        }

    if n:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(worker, idx, rule): idx
                for idx, rule in enumerate(categorized_rules)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    _, entry = future.result()
                except Exception as exc:
                    rule = categorized_rules[idx]
                    rule_id = RULE_ID_OFFSET + idx
                    slug = _rule_slug(rule["rule"])
                    category = rule["category"]
                    branch_name = f"rule/{idx:03d}-{category}-{slug}"[:80]
                    entry = {
                        "rule": rule["rule"],
                        "category": category,
                        "branch": branch_name,
                        "status": "error",
                        "error": str(exc),
                        "agent_log_path": str(AGENT_LOG_DIR / f"{idx:03d}-{slug}.log"),
                    }
                    log_path = AGENT_LOG_DIR / f"{idx:03d}-{slug}.log"
                    exception_text = (
                        "\n[status] worker_exception\n"
                        f"[error] {exc}\n"
                    )
                    if log_path.exists():
                        with log_path.open("a") as log_file:
                            log_file.write(exception_text)
                    else:
                        log_path.write_text(
                            f"[rule_index] {idx}\n"
                            f"[rule_id] {rule_id}\n"
                            f"[rule] {rule['rule']}\n"
                            f"[category] {category}\n"
                            f"[branch] {branch_name}\n"
                            f"{exception_text}"
                        )
                    print(f"  [{idx + 1:>2}/{n}] [ERROR: {exc}]", flush=True)
                results_by_idx[idx] = entry
                _write_rule_scenario_manifest(
                    [results_by_idx[i] for i in sorted(results_by_idx)],
                    conn,
                )

    results = [results_by_idx[i] for i in sorted(results_by_idx)]
    implemented_count = sum(1 for r in results if r["status"] == "committed")
    skipped_count = sum(1 for r in results if r["status"] == "cached")

    print()
    print(f"  Implemented (new branches) : {implemented_count}")
    print(f"  Skipped (already cached)   : {skipped_count}")
    print(f"  Total                      : {n}")

    _write_rule_scenario_manifest(results, conn)
    print("  Rule manifest              : SQLite rule_scenario_manifest")

    return results


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
    parser = argparse.ArgumentParser(description="Generate ScenicRules rulebook branches.")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of isolated Docker/Pi implementation workers to run in parallel.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Rulebook Generator — Pipeline")
    print("=" * 70)

    # Initialize SQLite database
    conn = init_database(DB_PATH)
    print("\n✓ Database initialized at", DB_PATH)

    print("\n[0/5] Ensuring handbook PDF is uploaded …")
    file_id = ensure_file_uploaded(PDF_PATH, conn)

    print("\n[1/5] Extracting simulation-relevant rules from PDF …")
    extracted = step1_extract_rules(file_id)
    rules = extracted["rules"]
    cache_step_to_db(conn, "step1_extract_rules", "Extract Rules", extracted)
    print(f"  → {len(rules)} rules extracted")

    print("\n[2/5] Categorizing rules and assessing severity …")
    categorized = step2_categorize_and_severity(rules)
    cat_rules = categorized["rules"]
    cache_step_to_db(conn, "step2_categorize_severity", "Categorize & Severity", categorized)
    collision_count = sum(1 for r in cat_rules if r["involves_collision"])
    print(f"  → {len(cat_rules)} rules categorized, {collision_count} collision-involved")

    print("\n[3/5] Building priority hierarchy …")
    hierarchy = step3_build_hierarchy(cat_rules)
    cache_step_to_db(conn, "step3_build_hierarchy", "Build Hierarchy", hierarchy)
    print(f"  → {len(hierarchy['tiers'])} tiers, "
          f"{len(hierarchy['collision_severity_ordering'])} collision types, "
          f"{len(hierarchy['pairwise_priority_edges'])} pairwise edges")

    print("\n[4/5] Converting rules to Signal Temporal Logic (STL) …")
    stl_conversions = step4_convert_to_stl(cat_rules, conn)
    cache_step_to_db(conn, "step4_stl_conversions", "STL Conversions", stl_conversions)
    stl_count = len(stl_conversions.get("conversions", []))
    print(f"  → {stl_count} rules converted to STL formulas")

    print("\n[5/5] Implementing rules as Python code in ScenicRules …")
    impl_results = step5_implement_rules(
        cat_rules,
        stl_conversions,
        conn,
        parallel_workers=args.parallel_workers,
    )
    cache_step_to_db(conn, "step5_implement_rules", "Implement Rules", impl_results)
    committed = sum(1 for r in impl_results if r["status"] == "committed")
    cached = sum(1 for r in impl_results if r["status"] == "cached")
    print(f"  → {committed} branches created, {cached} already active")

    print_summary(rules, categorized, hierarchy, stl_conversions)

    full_output = {
        "metadata": {
            "source": "California Driver's Handbook",
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "database_path": str(DB_PATH),
            "implementation_harness": IMPLEMENTATION_HARNESS,
            "implementation_provider": IMPLEMENTATION_PROVIDER,
            "implementation_model": IMPLEMENTATION_MODEL,
            "implementation_reasoning": IMPLEMENTATION_REASONING,
            "parallel_workers": args.parallel_workers,
            "pipeline_steps": [
                "extract_rules",
                "categorize_and_severity",
                "build_hierarchy",
                "convert_to_stl",
                "implement_rules",
            ],
        },
        "extracted_rules": rules,
        "categorized_rules": cat_rules,
        "hierarchy": hierarchy,
        "stl_conversions": stl_conversions,
        "stl_examples": STL_EXAMPLES,
        "implemented_rules": impl_results,
        "implemented_rule_manifest": {
            "entries": impl_results,
        },
    }
    store_json_artifact(conn, "rulebook_output", full_output)
    print(f"\n✓ Full structured output stored in {DB_PATH}")

    # Display example STL formulas
    print_section("STL EXAMPLE FORMULAS")
    for _, example in STL_EXAMPLES.items():
        print(f"\n  {example['title']}")
        print(f"  STL: {example['stl_formula']}")
        print(f"  VS:  {example['value_sum']}")

    conn.close()


if __name__ == "__main__":
    main()
