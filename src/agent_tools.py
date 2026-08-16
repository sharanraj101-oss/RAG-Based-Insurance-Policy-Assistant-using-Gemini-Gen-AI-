import json
from typing import Dict, Any

# Simulated Live Claim Database
SIMULATED_CLAIMS_DB = {
    "CLM-1001": {
        "claim_id": "CLM-1001",
        "policy_number": "POL-884920",
        "claimant_name": "Aarav Sharma",
        "policy_type": "Health Insurance",
        "status": "APPROVED",
        "approved_amount": "INR 85,000",
        "hospital_name": "Apollo Super Specialty Hospital",
        "last_updated": "2026-08-05",
        "remarks": "Claim processed and disbursement pending within 2 business days."
    },
    "CLM-1002": {
        "claim_id": "CLM-1002",
        "policy_number": "POL-332114",
        "claimant_name": "Priya Patel",
        "policy_type": "Motor Insurance",
        "status": "UNDER_INSPECTION",
        "approved_amount": "Pending Surveyor Assessment",
        "garage_name": "Cyber Auto Works, Bengaluru",
        "last_updated": "2026-08-07",
        "remarks": "Surveyor has inspected vehicle. Awaiting final repair estimate."
    },
    "CLM-1003": {
        "claim_id": "CLM-1003",
        "policy_number": "POL-554209",
        "claimant_name": "Rohan Verma",
        "policy_type": "Life & Term Insurance",
        "status": "DOCUMENTS_REQUIRED",
        "approved_amount": "INR 50,00,000",
        "hospital_name": "Fortis Healthcare",
        "last_updated": "2026-08-08",
        "remarks": "Original death certificate received. Awaiting cancelled cheque for account transfer."
    }
}

def lookup_claim_status(claim_id: str) -> Dict[str, Any]:
    """
    Looks up live claim status from insurer database.
    """
    clean_id = claim_id.strip().upper()
    if clean_id in SIMULATED_CLAIMS_DB:
        return {
            "found": True,
            "data": SIMULATED_CLAIMS_DB[clean_id]
        }
    else:
        return {
            "found": False,
            "message": f"No active claim record found for Claim ID '{clean_id}'. Available test IDs: CLM-1001, CLM-1002, CLM-1003."
        }

def calculate_insurance_premium(age: int, sum_insured: float, plan_type: str = "health", zero_dep: bool = False) -> Dict[str, Any]:
    """
    Calculates estimated annual insurance premium based on risk factors.
    """
    base_rate = 0.015 if plan_type.lower() == "health" else 0.02
    
    if age < 30:
        age_multiplier = 0.9
    elif age < 45:
        age_multiplier = 1.1
    elif age < 60:
        age_multiplier = 1.4
    else:
        age_multiplier = 1.8
        
    base_premium = sum_insured * base_rate * age_multiplier
    rider_cost = 2500.0 if zero_dep else 0.0
    
    total_premium = base_premium + rider_cost
    gst_amount = total_premium * 0.18
    final_payable = total_premium + gst_amount
    
    return {
        "age": age,
        "sum_insured": f"INR {sum_insured:,.2f}",
        "plan_type": plan_type.capitalize(),
        "base_premium": f"INR {base_premium:,.2f}",
        "rider_cost": f"INR {rider_cost:,.2f}",
        "gst_18_percent": f"INR {gst_amount:,.2f}",
        "total_annual_premium": f"INR {final_payable:,.2f}"
    }

if __name__ == "__main__":
    print("Testing Claim Lookup:")
    print(json.dumps(lookup_claim_status("CLM-1001"), indent=2))
    
    print("\nTesting Premium Calculator:")
    print(json.dumps(calculate_insurance_premium(35, 1000000, "health", True), indent=2))
