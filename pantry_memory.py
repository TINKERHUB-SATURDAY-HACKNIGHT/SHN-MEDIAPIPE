from datetime import datetime


class PantryMemory:
    URGENCY_ORDER = {
        "EAT_TODAY": 1,
        "USE_SOON": 2,
        "STORE_LONGER": 3,
    }

    def __init__(self):
        self.items = []

    def add_item(self, scan_result: dict) -> dict:
        entry = {
            "id": len(self.items) + 1,
            "fruit_name": scan_result.get("fruit_name", "Fruit"),
            "surface_description": scan_result.get("surface_description", ""),
            "urgency": scan_result.get("urgency", "USE_SOON"),
            "shelf_life_days": scan_result.get("shelf_life_days", "2-3 days"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.items.append(entry)
        return entry

    def get_sorted_items(self) -> list:
        return sorted(
            self.items,
            key=lambda x: self.URGENCY_ORDER.get(x.get("urgency", "USE_SOON"), 2),
        )

    def get_comparison_summary(self) -> str:
        """Short suffix telling the user how the latest fruit compares to previously scanned fruits."""
        if len(self.items) < 2:
            return "Scan another fruit to hear which one to use first and which to use last."
        sorted_list = self.get_sorted_items()
        first_item = sorted_list[0]["fruit_name"]
        last_item = sorted_list[-1]["fruit_name"]
        return f"Priority order: Use {first_item} first, and use {last_item} last."

    def get_spoken_recap(self) -> str:
        if not self.items:
            return "No fruits scanned yet. Hold a fruit inside the center box to start the 10 second inspection."

        sorted_items = self.get_sorted_items()
        first_item = sorted_items[0]
        last_item = sorted_items[-1]

        if len(sorted_items) == 1:
            return (
                f"You have scanned one item: {first_item['fruit_name']}, "
                f"status {first_item['shelf_life_days']}."
            )

        eat_today = [i["fruit_name"] for i in sorted_items if i["urgency"] == "EAT_TODAY"]
        use_soon = [i["fruit_name"] for i in sorted_items if i["urgency"] == "USE_SOON"]
        store_longer = [i["fruit_name"] for i in sorted_items if i["urgency"] == "STORE_LONGER"]

        msg = [f"Out of {len(sorted_items)} fruits scanned:"]
        if eat_today:
            msg.append(f"Use FIRST today: {', '.join(eat_today)}.")
        if use_soon:
            msg.append(f"Use in two to three days: {', '.join(use_soon)}.")
        if store_longer:
            msg.append(f"Use LAST, can stay longer: {', '.join(store_longer)}.")
        if not eat_today or not store_longer:
            msg.append(f"Use {first_item['fruit_name']} first, and {last_item['fruit_name']} last.")

        return " ".join(msg)