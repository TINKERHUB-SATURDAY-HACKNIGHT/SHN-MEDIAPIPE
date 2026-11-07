from datetime import datetime


class PantryMemory:
    URGENCY_ORDER = {
        "EAT_TODAY": 1,
        "USE_SOON": 2,
        "STORE_LONGER": 3,
    }

    URGENCY_LABELS = {
        "EAT_TODAY": "Eat Today",
        "USE_SOON": "Use in 2 to 3 Days",
        "STORE_LONGER": "Can Stay Longer",
    }

    def __init__(self):
        self.items = []

    def add_item(self, scan_result: dict):
        entry = {
            "id": len(self.items) + 1,
            "fruit_name": scan_result.get("fruit_name", "Fruit"),
            "surface_description": scan_result.get("surface_description", ""),
            "urgency": scan_result.get("urgency", "USE_SOON"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.items.append(entry)
        return entry

    def get_sorted_items(self):
        return sorted(
            self.items,
            key=lambda x: self.URGENCY_ORDER.get(x.get("urgency", "USE_SOON"), 2),
        )

    def get_spoken_recap(self) -> str:
        if not self.items:
            return "Your pantry scan list is currently empty. Hold a fruit and pinch your thumb and index finger to scan."

        sorted_items = self.get_sorted_items()
        eat_first = [
            item["fruit_name"]
            for item in sorted_items
            if item["urgency"] == "EAT_TODAY"
        ]
        use_soon = [
            item["fruit_name"]
            for item in sorted_items
            if item["urgency"] == "USE_SOON"
        ]
        keep_longer = [
            item["fruit_name"]
            for item in sorted_items
            if item["urgency"] == "STORE_LONGER"
        ]

        parts = [f"You have scanned {len(self.items)} items so far."]
        if eat_first:
            parts.append(f"Use first today: {', '.join(eat_first)}.")
        if use_soon:
            parts.append(f"Use within two to three days: {', '.join(use_soon)}.")
        if keep_longer:
            parts.append(f"Can stay longer: {', '.join(keep_longer)}.")

        return " ".join(parts)