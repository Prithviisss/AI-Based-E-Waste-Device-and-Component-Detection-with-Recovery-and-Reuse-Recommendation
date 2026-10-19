"""
tests/test_class_map.py
------------------------
Unit tests for src/class_map.yaml integrity.
Covers: all dataset2 classes present, uniqueness, battery dual-source notes,
        unmapped_policy field, heatsink→aluminium mapping.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASS_MAP_PATH = Path(__file__).resolve().parents[1] / "src" / "class_map.yaml"


@pytest.fixture(scope="module")
def class_map():
    with open(CLASS_MAP_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def mappings(class_map):
    return class_map.get("mappings", [])


class TestClassMapStructure:
    def test_file_exists(self):
        assert CLASS_MAP_PATH.exists(), f"class_map.yaml not found at {CLASS_MAP_PATH}"

    def test_unmapped_policy_present(self, class_map):
        assert "unmapped_policy" in class_map

    def test_unmapped_policy_value(self, class_map):
        assert class_map["unmapped_policy"] == "report_in_output"

    def test_mappings_is_list(self, mappings):
        assert isinstance(mappings, list)
        assert len(mappings) > 0

    def test_all_entries_have_required_fields(self, mappings):
        for entry in mappings:
            assert "source" in entry, f"Missing 'source' in {entry}"
            assert "label"  in entry, f"Missing 'label' in {entry}"
            assert "kb_key" in entry, f"Missing 'kb_key' in {entry}"


class TestUniqueness:
    def test_no_duplicate_source_label_pairs(self, mappings):
        seen = set()
        for entry in mappings:
            key = (entry["source"], entry["label"])
            assert key not in seen, f"Duplicate entry: {key}"
            seen.add(key)


class TestBatteryDualSource:
    def test_battery_dataset1_present(self, mappings):
        ds1_battery = [e for e in mappings if e["source"] == "dataset1" and e["label"] == "battery"]
        assert len(ds1_battery) == 1

    def test_battery_dataset2_present(self, mappings):
        ds2_battery = [e for e in mappings if e["source"] == "dataset2" and e["label"] == "battery"]
        assert len(ds2_battery) == 1

    def test_battery_notes_are_distinct(self, mappings):
        batteries = [e for e in mappings if e["label"] == "battery" and e.get("note")]
        notes = [e["note"] for e in batteries]
        assert len(notes) == len(set(notes)), "Battery notes from dataset1 and dataset2 must be distinct"

    def test_battery_both_map_to_battery_kb(self, mappings):
        batteries = [e for e in mappings if e["label"] == "battery"]
        for b in batteries:
            assert b["kb_key"] == "battery"


class TestKeyMappings:
    def test_heatsink_maps_to_aluminium(self, mappings):
        heatsink = [e for e in mappings if e["label"] == "heatsink"]
        assert len(heatsink) >= 1
        for h in heatsink:
            assert h["kb_key"] == "aluminium"

    def test_heatsink_has_note(self, mappings):
        heatsink = [e for e in mappings if e["label"] == "heatsink"]
        for h in heatsink:
            assert h.get("note"), "heatsink must have a note explaining the aluminium inference"

    def test_dataset3_ram_present(self, mappings):
        ram = [e for e in mappings if e["source"] == "dataset3" and e["label"] == "ram"]
        assert len(ram) == 1
        assert ram[0]["kb_key"] == "ram"

    def test_dataset3_ssd_present(self, mappings):
        ssd = [e for e in mappings if e["source"] == "dataset3" and e["label"] == "SSD"]
        assert len(ssd) == 1
        assert ssd[0]["kb_key"] == "ssd"

    def test_fallback_disk_maps_to_storage_drive(self, mappings):
        disk = [e for e in mappings if e["source"] == "fallback" and e["label"] == "DISK"]
        assert len(disk) == 1
        assert disk[0]["kb_key"] == "storage_drive"


class TestDataset1Coverage:
    EXPECTED_LABELS = [
        "battery", "computer", "keyboard", "microwave", "mobile",
        "mouse", "PCB", "player", "printer", "speaker", "washing_machine", "TV",
    ]

    def test_all_12_classes_present(self, mappings):
        ds1_labels = {e["label"] for e in mappings if e["source"] == "dataset1"}
        for lbl in self.EXPECTED_LABELS:
            assert lbl in ds1_labels, f"Dataset1 label '{lbl}' missing from class_map.yaml"
