"""
Dataset adapters.

WHY THIS PACKAGE EXISTS
-----------------------
The rest of the project speaks ONE vocabulary: `patient_id`, `length_of_stay`,
`readmitted_30d`, and so on. Real datasets never use your vocabulary - they use
theirs. An adapter is the translator that sits between them.

Adding support for a new dataset means writing ONE new file in this folder. No
other file in the project changes. That is the whole point.

Available adapters:
    uci_diabetes  - Diabetes 130-US Hospitals, 1999-2008 (the project default)
"""

from __future__ import annotations

from . import uci_diabetes

ADAPTERS = {
    "uci_diabetes": uci_diabetes,
}

__all__ = ["ADAPTERS", "uci_diabetes"]
