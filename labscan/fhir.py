"""Выгрузка в FHIR R4: Patient + Observation в одном transaction Bundle.

FHIR — общий язык медицинских систем; если у МИС клиента есть FHIR-API,
интеграция сводится к POST этого Bundle. Если нет — адаптер переводит его
в формат МИС (см. mis.py).
"""

from __future__ import annotations

import uuid
from datetime import date

from .catalog import BY_CODE

UCUM = {"g/L": "g/L", "mmol/L": "mmol/L", "umol/L": "umol/L", "U/L": "U/L", "mg/L": "mg/L", "%": "%",
        "mm/h": "mm/h", "fL": "fL", "mIU/L": "m[IU]/L", "10^9/L": "10*9/L", "10^12/L": "10*12/L"}
INTERPRETATION = {"H": ("H", "High"), "L": ("L", "Low"), "N": ("N", "Normal")}


def patient_resource(full_name: str, birth: date | None, pid: str) -> dict:
    parts = full_name.split()
    res = {"resourceType": "Patient", "id": pid,
           "name": [{"text": full_name, "family": parts[0] if parts else None, "given": parts[1:]}]}
    if birth:
        res["birthDate"] = birth.isoformat()
    return res


def observation_resource(obs: dict, patient_ref: str, collected: date | None) -> dict:
    a = BY_CODE.get(obs["code"])
    coding = [{"system": "http://loinc.org", "code": obs["loinc"], "display": a.name_ru if a else obs["name"]}]
    res = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                  "code": "laboratory"}]}],
        "code": {"coding": coding, "text": obs["name"]},
        "subject": {"reference": patient_ref},
        "valueQuantity": {"value": obs["value"], "unit": obs["unit"],
                          "system": "http://unitsofmeasure.org", "code": UCUM.get(obs["unit"], obs["unit"])},
    }
    if collected:
        res["effectiveDateTime"] = collected.isoformat()
    if obs.get("ref_low") is not None or obs.get("ref_high") is not None:
        rr = {}
        if obs.get("ref_low") is not None:
            rr["low"] = {"value": obs["ref_low"], "unit": obs["unit"]}
        if obs.get("ref_high") is not None:
            rr["high"] = {"value": obs["ref_high"], "unit": obs["unit"]}
        res["referenceRange"] = [rr]
    if obs.get("flag") in INTERPRETATION:
        code, display = INTERPRETATION[obs["flag"]]
        res["interpretation"] = [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
            "code": code, "display": display}]}]
    return res


def to_bundle(full_name: str, birth: date | None, collected: date | None, observations: list[dict],
              patient_id: str) -> dict:
    pref = f"Patient/{patient_id}"
    entries = [{"resource": patient_resource(full_name, birth, patient_id),
                "request": {"method": "PUT", "url": pref}}]
    for o in observations:
        r = observation_resource(o, pref, collected)
        entries.append({"resource": r, "request": {"method": "POST", "url": "Observation"}})
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}
