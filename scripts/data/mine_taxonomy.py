#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import yaml

from common import repo_path, utc_now, write_json


def build_taxonomy() -> Dict[str, Any]:
    return {
        "created_at": utc_now(),
        "source": "abstracted_from_RISE_GRADE_KRIS_without_copying_samples",
        "benchmark_families": {
            "RISE_like": {
                "weight_pilot": 0.40,
                "families": {
                    "temporal_reasoning": [
                        "aging_decay",
                        "growth",
                        "seasonal_change",
                        "charging_discharging",
                        "settling_drying",
                    ],
                    "causal_reasoning": [
                        "force_deformation",
                        "thermal_phase_change",
                        "fluid_dynamics",
                        "chemical_effect",
                        "biological_response",
                    ],
                    "spatial_reasoning": [
                        "viewpoint_change",
                        "3d_rotation",
                        "occlusion",
                        "relative_position",
                        "object_rearrangement",
                    ],
                    "logical_reasoning": [
                        "math_symbolic",
                        "board_game",
                        "maze_path",
                        "rule_based_pattern",
                        "counting_constraint",
                    ],
                },
            },
            "GRADE_like": {
                "weight_pilot": 0.35,
                "domains": [
                    "math",
                    "physics",
                    "chemistry",
                    "biology",
                    "computer_science",
                    "economics",
                    "history",
                    "geography",
                    "music",
                    "sports",
                ],
            },
            "KRIS_like": {
                "weight_pilot": 0.25,
                "knowledge_types": ["factual", "conceptual", "procedural"],
                "families": [
                    "anomaly_correction",
                    "multi_element_composition",
                    "temporal_prediction",
                    "viewpoint_change",
                    "rule_based_reasoning",
                    "entity_attribute_edit",
                ],
            },
        },
    }


def build_checklist_templates() -> Dict[str, Any]:
    return {
        "global": [
            {"id": "preserve", "question": "Are unrelated regions, identity, lighting, viewpoint, and background preserved?", "weight": 0.25},
            {"id": "visual_quality", "question": "Is the edited image visually plausible and free of obvious artifacts?", "weight": 0.20},
            {"id": "no_extra", "question": "Does the edit avoid unrelated objects, text, or style changes?", "weight": 0.15},
        ],
        "cognitive": [
            {"id": "reasoning", "question": "Does the edit reflect the required reasoning or knowledge instead of a superficial visual change?", "weight": 0.25},
            {"id": "evidence", "question": "Is each non-obvious factual or symbolic claim supported by image observation, search, or solver evidence?", "weight": 0.15},
        ],
        "grade_style": [
            {"id": "answer_correct", "question": "Is the discipline-specific answer or state correct?", "weight": 0.45},
            {"id": "rubric_visible", "question": "Is the required result visible and readable in the edited image?", "weight": 0.20},
        ],
        "rise_style": [
            {"id": "target_state", "question": "Does the image show the correct target state implied by the instruction?", "weight": 0.40},
            {"id": "source_consistency", "question": "Does the edited result preserve the source object's identity and context?", "weight": 0.25},
        ],
        "kris_style": [
            {"id": "knowledge_plausibility", "question": "Is the result plausible under the required factual, conceptual, or procedural knowledge?", "weight": 0.40},
            {"id": "relation_binding", "question": "Are multi-object relations and constraints maintained?", "weight": 0.20},
        ],
    }


def build_knowledge_bank() -> Dict[str, Any]:
    return {
        "math_linear_equation": {
            "queries": ["linear equation solve ax plus b equals c", "how to solve one variable linear equations"],
            "facts": ["A one-variable linear equation ax + b = c is solved by x = (c - b) / a."],
        },
        "physics_refraction": {
            "queries": ["light ray bends toward normal entering denser medium", "Snell law refraction diagram"],
            "facts": ["When light enters a denser transparent medium from air, the refracted ray bends toward the normal."],
        },
        "chemistry_litmus": {
            "queries": ["blue litmus paper acid turns red", "acid base indicator litmus color"],
            "facts": ["Blue litmus paper turns red in acidic solution."],
        },
        "biology_lifecycle": {
            "queries": ["butterfly life cycle caterpillar chrysalis", "larva pupa butterfly stages"],
            "facts": ["A caterpillar forms a chrysalis before emerging as an adult butterfly."],
        },
        "cs_bst": {
            "queries": ["binary search tree insertion rule", "BST insert smaller left greater right"],
            "facts": ["In a binary search tree, smaller values go to the left subtree and larger values to the right subtree."],
        },
        "economics_demand_shift": {
            "queries": ["demand increase shifts curve right", "supply demand graph demand increase"],
            "facts": ["An increase in demand shifts the demand curve to the right."],
        },
        "temporal_fruit_decay": {
            "queries": ["fruit ripening browning spots over time", "banana peel brown spots ripening"],
            "facts": ["Many fruits darken, soften, and develop brown or black spots as they ripen or decay."],
        },
        "thermal_ice_melt": {
            "queries": ["ice melts into water under heat", "solid ice phase change heat"],
            "facts": ["Heating ice above its melting point changes it from solid ice to liquid water."],
        },
        "spatial_occlusion": {
            "queries": ["occlusion object behind another object visibility", "visual spatial relation behind in front"],
            "facts": ["An object placed behind another object should be partially hidden by the front object."],
        },
        "traffic_light_order": {
            "queries": ["traffic light color order red yellow green", "standard traffic signal vertical order"],
            "facts": ["A standard vertical traffic light places red at the top, yellow in the middle, and green at the bottom."],
        },
    }


def write_yaml(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    tax_dir = repo_path("data", "taxonomy")
    write_yaml(tax_dir / "benchmark_taxonomy.yaml", build_taxonomy())
    write_yaml(tax_dir / "checklist_templates.yaml", build_checklist_templates())
    write_json(tax_dir / "knowledge_bank_v0.json", build_knowledge_bank())
    print("wrote data/taxonomy/benchmark_taxonomy.yaml, checklist_templates.yaml, knowledge_bank_v0.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

