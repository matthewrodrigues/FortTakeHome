"""Textual insights (§8.6-8.7).

Phase 5 ships the **execution narrative** (form half only): the form-subscore
decomposition rendered as measured-change prose, largest deviations first. The effort
narrative (RIR/RPE) arrives in Phase 8. Kept import-light (no streamlit/plotly) so the CLI
demo depends only on core packages.
"""

from wristset.insights.execution import execution_narrative

__all__ = ["execution_narrative"]
