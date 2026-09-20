"""
Sanity tests for analysis.py.

Run from the project root with:
    pytest -v

These tests check flood classification, critical-time calculation,
population impact, summary KPIs, and dashboard alert messages.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import (  # noqa: E402
    classify,
    time_to_critical,
    affected_population,
    summarize,
    banner_message,
)


# ---------------------------------------------------------------------------
# Test 1: Safe depths are classified correctly

def test_1_safe_depths():
    depth = np.array([
        [0.05, 0.10],
        [0.12, 0.14]
    ])

    result = classify(depth)

    assert np.all(result == 0)


# ---------------------------------------------------------------------------
# Test 2: Warning depths are classified correctly

def test_2_warning_depths():
    depth = np.array([
        [0.15, 0.20],
        [0.25, 0.29]
    ])

    result = classify(depth)

    assert np.all(result == 1)


# ---------------------------------------------------------------------------
# Test 3: Critical depths are classified correctly

def test_3_critical_depths():
    depth = np.array([
        [0.30, 0.40],
        [0.50, 0.60]
    ])

    result = classify(depth)

    assert np.all(result == 2)


# ---------------------------------------------------------------------------
# Test 4: Mixed depths produce the expected classifications

def test_4_mixed_classification():
    depth = np.array([
        [0.05, 0.20],
        [0.35, 0.10]
    ])

    result = classify(depth)

    expected = np.array([
        [0, 1],
        [2, 0]
    ])

    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# Test 5: Time to critical is calculated correctly

def test_5_time_to_critical():
    depth = np.array([
        [
            [0.05, 0.10],
            [0.05, 0.10]
        ],
        [
            [0.10, 0.20],
            [0.10, 0.15]
        ],
        [
            [0.20, 0.30],
            [0.10, 0.40]
        ]
    ])

    time_h = np.array([0.0, 1.0, 2.0])

    eta = time_to_critical(depth, time_h)

    # Cell [0, 0] becomes critical after the available
    # simulation period, so it should not have an earlier
    # critical time than the cells that actually reach it.
    assert eta.shape == (2, 2)

    # These two cells reach critical at t = 2 h.
    assert eta[0, 1] == pytest.approx(2.0)
    assert eta[1, 1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 6: Affected population is calculated from critical regions

def test_6_affected_population():
    depth = np.array([
        [
            [0.10, 0.20],
            [0.35, 0.40]
        ]
    ])

    population = np.array([
        [100, 200],
        [300, 400]
    ])

    affected = affected_population(depth, population)

    # Warning cell gets weight 0.25
    # Critical cells get weight 1.0
    expected = (200 * 0.25) + 300 + 400

    assert affected == pytest.approx(expected)

# ---------------------------------------------------------------------------
# Test 7: summarize() returns the expected main sections

def test_7_summary_contains_expected_sections():
    depth = np.array([
        [
            [0.05, 0.10],
            [0.05, 0.10]
        ],
        [
            [0.20, 0.30],
            [0.10, 0.40]
        ]
    ])

    time_h = np.array([0.0, 1.0])

    result = {
        "depth": depth,
        "time_h": time_h,
        "rain_mm_h": np.array([50.0, 50.0]),
    }

    city = {
        "elevation": np.zeros((2, 2)),
        "drainage": np.zeros((2, 2)),
        "population": np.array([
            [100, 200],
            [300, 400]
        ])
    }

    summary = summarize(result, city)

    assert "kpis" in summary
    assert "classification" in summary
    assert "eta" in summary
    assert "warning_table" in summary


# ---------------------------------------------------------------------------
# Test 8: Summary identifies critical regions

def test_8_summary_detects_critical_regions():
    depth = np.array([
        [
            [0.05, 0.10],
            [0.05, 0.10]
        ],
        [
            [0.20, 0.30],
            [0.10, 0.40]
        ]
    ])

    result = {
        "depth": depth,
        "time_h": np.array([0.0, 1.0]),
        "rain_mm_h": np.array([50.0, 50.0]),
    }

    city = {
        "elevation": np.zeros((2, 2)),
        "drainage": np.zeros((2, 2)),
        "population": np.ones((2, 2)) * 100
    }

    summary = summarize(result, city)

    assert summary["kpis"]["critical_regions"] == 2


# ---------------------------------------------------------------------------
# Test 9: banner_message() produces a critical alert

def test_9_banner_message_critical():
    summary = {
        "kpis": {
            "critical_regions": 2,
            "warning_regions": 0,
            "first_critical_h": 0.5,
        },
        "warning_table": pd.DataFrame({
            "Region": ["R0_0", "R0_1"],
            "Lead time (h)": [0.5, 1.0],
        })
    }

    message = banner_message(summary)

    assert "CRITICAL ALERT" in message


# ---------------------------------------------------------------------------
# Test 10: A simulation with no critical region produces a non-critical message

def test_10_no_critical_region():
    depth = np.array([
        [
            [0.02, 0.05],
            [0.03, 0.08]
        ],
        [
            [0.05, 0.10],
            [0.08, 0.12]
        ]
    ])

    result = {
        "depth": depth,
        "time_h": np.array([0.0, 1.0]),
        "rain_mm_h": np.array([10.0, 10.0]),
    }

    city = {
        "elevation": np.zeros((2, 2)),
        "drainage": np.zeros((2, 2)),
        "population": np.ones((2, 2)) * 100
    }

    summary = summarize(result, city)

    assert summary["kpis"]["critical_regions"] == 0

    message = banner_message(summary)

    assert "CRITICAL ALERT" not in message