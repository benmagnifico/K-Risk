import os
import json
import random
import pandas as pd


#HighD的system message
highd_msg = """"

You are ChatGPT, a large language model trained by OpenAI. You are now acting as a mature driving assistant. And your task is to make the decision that assures the safety.

You have access to the same information as a real human driver, which includes the following:
1. Description of the environment
2. Your own vehicle's position, dynamic information, and legally permissible actions in that lane.
3. Vehicles in your current lane and adjacent lanes, including their position, dynamic information, and legally permissible actions in that lane.

This scenario takes place at a highway. The positive x-coordinate points east, while the positive y-coordinate points south. The top-left corner is the origin (0,0). The velocity components follow the same convention: a negative vx indicates movement toward the west, while a negative vy indicates movement toward the north.

Your available actions include:
1. IDLE: Remain in the current lane with the current speed (Action ID: 1)
2. Turn Left: Change to the lane on the left of the current lane (Action ID: 2)
3. Turn Right: Change to the lane on the right of the current lane (Action ID: 3)
4. Acceleration: Increase vehicle speed (Action ID: 4)
5. Deceleration: Reduce vehicle speed (Action ID: 5)

A clear action ID must be chosen at the end of your reasoning process.
"""

# === Risk Reminder Messages ===
risk_reminders = {
    "preceding_id": {
        "brake_label": "is decelerating. Prepare to slow down and maintain a safe distance.",
        "left_turn_label": "is shifting left. Avoid overtaking from the left; anticipate lane changes.",
        "right_turn_label": "is shifting right. Avoid overtaking from the right; anticipate lane changes."
    },

    "following_id": {
        "acc_label": "is accelerating, possibly preparing to overtake. Maintain speed and avoid unnecessary braking.",
        "left_turn_label": "is shifting left, possibly overtaking. Avoid changing to the left lane.",
        "right_turn_label": "is shifting right, possibly overtaking. Avoid changing to the right lane."
    },

    "left_alongside_id": {
        "brake_label": "is decelerating. Avoid lane changes to the left.",
        "right_turn_label": "is shifting toward your lane. Stay alert for possible intrusion.",
        "acc_label": "is accelerating. Avoid lane changes to the left."
    },

    "right_alongside_id": {
        "brake_label": "is decelerating. Avoid lane changes to the right.",
        "left_turn_label": "is shifting toward your lane. Stay alert for possible intrusion.",
        "acc_label": "is accelerating. Avoid lane changes to the right."
    },

    "left_preceding_id": {
        "brake_label": "is decelerating. Be ready to adjust speed and avoid overtaking from the left.",
        "right_turn_label": "is shifting toward your lane. Prepare for a potential cut-in.",
    },

    "right_preceding_id": {
        "brake_label": "is decelerating. Be ready to adjust speed and avoid overtaking from the right.",
        "left_turn_label": "is shifting toward your lane. Prepare for a potential cut-in.",
    },

    "left_following_id": {
        "acc_label": "is accelerating, possibly preparing to overtake. Avoid changing to the left lane.",
        "right_turn_label": "is shifting toward your lane. Maintain speed and avoid unnecessary braking.",
    },

    "right_following_id": {
        "acc_label": "is accelerating, possibly preparing to overtake. Avoid changing to the right lane.",
        "left_turn_label": "is shifting toward your lane. Maintain speed and avoid unnecessary braking.",
    }
}

# === Lane Info for HighD ===
lane_info = {
    "lane_counts": {
        4: ["1", "2", "3", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"],
        6: ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44",
            "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57"]
    },
    "four_lane_info": {
        2: {"direction": "east-west", "left_adjacent_lane": [3], "right_adjacent_lane": None, "available_actions": ["Move straight", "change left to lane 1"]},
        3: {"direction": "east-west", "left_adjacent_lane": None, "right_adjacent_lane": [2], "available_actions": ["Move straight", "change right to lane 2"]},
        5: {"direction": "west-east", "left_adjacent_lane": None, "right_adjacent_lane": [5], "available_actions": ["Move straight", "change right to lane 6"]},
        6: {"direction": "west-east", "left_adjacent_lane": [6], "right_adjacent_lane": None, "available_actions": ["Move straight", "change left to lane 5"]}
    },
    "six_lane_info": {
        2: {"direction": "east-west", "left_adjacent_lane": [3], "right_adjacent_lane": None, "available_actions": ["Move straight", "change left to lane 3"]},
        3: {"direction": "east-west", "left_adjacent_lane": [4], "right_adjacent_lane": [2], "available_actions": ["Move straight", "change right to lane 2", "change left to lane 4"]},
        4: {"direction": "east-west", "left_adjacent_lane": None, "right_adjacent_lane": [3], "available_actions": ["Move straight", "change right to lane 3"]},
        6: {"direction": "west-east", "left_adjacent_lane": None, "right_adjacent_lane": [7], "available_actions": ["Move straight", "change right to lane 7"]},
        7: {"direction": "west-east", "left_adjacent_lane": [6], "right_adjacent_lane": [8], "available_actions": ["Move straight", "change right to lane 8", "change left to lane 6"]},
        8: {"direction": "west-east", "left_adjacent_lane": [7], "right_adjacent_lane": None, "available_actions": ["Move straight", "change left to lane 7"]}
    }
}



def generate_highd(json_path, frame_step=5, start_frame=None, end_frame=None):
    """
    Generate a risk scene description from a JSON file using HighD-specific configuration.
    Allows frame range control using start_frame and end_frame.
    """
    lane_counts = lane_info["lane_counts"]
    four_lane_info = lane_info["four_lane_info"]
    six_lane_info = lane_info["six_lane_info"]

    filename = os.path.basename(json_path)
    try:
        parts = filename.replace(".json", "").split("_")
        track_number = int(parts[1])
        ego_id = int(parts[2])
    except Exception:
        return f"Error: Could not extract track_number and ego_id from filename: {filename}"

    track_lane_mapping = {
        int(track): lane_count
        for lane_count, tracks in lane_counts.items()
        for track in tracks
    }

    num_lanes = track_lane_mapping.get(track_number, None)
    if num_lanes == 4:
        active_lane_info = four_lane_info
    elif num_lanes == 6:
        active_lane_info = six_lane_info
    else:
        return f"Error: Lane count for track {track_number} not supported."

    if not os.path.exists(json_path):
        return f"Error: JSON file not found: {json_path}"

    with open(json_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        df = pd.DataFrame(raw)
    elif isinstance(raw, dict) and "frames" in raw:
        rows = []
        for frame in raw["frames"]:
            frame_id = frame.get("frame_id")
            for v in frame.get("vehicles", []):
                v["frame"] = frame_id
                rows.append(v)
        df = pd.DataFrame(rows)
    else:
        return f"Error: Unknown JSON format structure."

    df.rename(columns={"car_id": "id", "car_center_x": "x", "car_center_y": "y", "frame_id": "frame"}, inplace=True)
    if "frame" not in df.columns or "id" not in df.columns:
        return "Error: Missing essential fields 'frame' or 'id'."

    # === NEW: Filter frame range ===
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]


    frames = sorted(df["frame"].unique())[::frame_step]
    descriptions = []

    position_alias = {
        "precedingId": "preceding in your lane",
        "followingId": "following you",
        "leftPrecedingId": "preceding in the left lane",
        "leftFollowingId": "following in the left lane",
        "leftAlongsideId": "to your left",
        "rightPrecedingId": "preceding in the right lane",
        "rightFollowingId": "following in the right lane",
        "rightAlongsideId": "to your right"
    }

    adverb_phrases = {
        "acc": "accelerating at a higher-than-average rate",
        "brake": "braking harder than usual",
        "yaw_left": "moving left at a higher-than-average yaw rate",
        "yaw_right": "moving right at a higher-than-average yaw rate"
    }

    categories = {
        "precedingId": "driving ahead in your lane",
        "followingId": "driving behind in your lane",
        "leftPrecedingId": "driving ahead on the lane to your left",
        "leftFollowingId": "driving behind on the lane to your left",
        "leftAlongsideId": "driving alongside on the lane to your left",
        "rightPrecedingId": "driving ahead on the lane to your right",
        "rightFollowingId": "driving behind on the lane to your right",
        "rightAlongsideId": "driving alongside on the lane to your right"
    }

    for frame in frames:
        frame_df = df[df["frame"] == frame]
        ego_row = frame_df[frame_df["id"] == ego_id]
        if ego_row.empty:
            continue

        ego_row = ego_row.iloc[0]
        ego_lane = ego_row.get("laneId", "Unknown")
        ego_lane_details = active_lane_info.get(ego_lane, {})
        ego_actions = ego_lane_details.get("available_actions", ["No available actions"])
        ego_acc = ego_row.get("acc_rate_1s", "N/A")
        ego_acc = f"{float(ego_acc):.2f}" if pd.notna(ego_acc) and ego_acc != "N/A" else "N/A"
        ego_class = ego_row.get("vehicle_class", "N/A")

        paragraph = f"""
        --------------------------------------------
        \nFrame {frame}:
        The ego vehicle is driving on lane {ego_lane}, a {ego_lane_details.get("direction", "Unknown")} lane.
        The legal actions on this lane are: {', '.join(f'({i+1}) {action}' for i, action in enumerate(ego_actions))}.
        Position and dynamic information (for longitudinal and lateral risk assessment):
        - Position: ({ego_row['x']:.2f}, {ego_row['y']:.2f}) m
        - Longitudinal velocity: {ego_row.get('xVelocity', 0):.2f} m/s, Lateral velocity: {ego_row.get('yVelocity', 0):.2f} m/s
        - Acceleration: {ego_acc} m/s²
        - Vehicle Class: {ego_class}
        """

        surrounding_descriptions = []
        reminders = []

        for col, description in categories.items():
            sur_id = ego_row.get(col)
            if pd.isna(sur_id) or sur_id == -1:
                continue

            sur_row = frame_df[frame_df["id"] == int(sur_id)]
            if sur_row.empty:
                continue
            sur_row = sur_row.iloc[0]

            sur_lane = sur_row.get("laneId", "Unknown")
            sur_lane_details = active_lane_info.get(sur_lane, {})
            sur_actions = sur_lane_details.get("available_actions", ["No available actions"])
            sur_acc = sur_row.get("acc_rate_1s", "N/A")
            sur_acc = f"{float(sur_acc):.2f}" if pd.notna(sur_acc) and sur_acc != "N/A" else "N/A"
            sur_class = sur_row.get("vehicle_class", "N/A")

            surrounding_descriptions.append(f"""
            Vehicle {sur_id} is {description}: Lane {sur_lane}, a {sur_lane_details.get("direction", "Unknown")} lane.
            The legal actions on this lane are: {', '.join(f'({i+1}) {action}' for i, action in enumerate(sur_actions))}.
            Position and dynamic information:
            - Position: ({sur_row['x']:.2f}, {sur_row['y']:.2f}) m
            - Longitudinal velocity: {sur_row.get('xVelocity', 0):.2f} m/s, Lateral velocity: {sur_row.get('yVelocity', 0):.2f} m/s
            - Acceleration: {sur_acc} m/s²
            - Vehicle Class: {sur_class}
            """)

            behavior_keys = {
                "brake_label": sur_row.get("brake_high", False),
                "acc_label": sur_row.get("acc_high", False),
                "left_turn_label": sur_row.get("yaw_left", False),
                "right_turn_label": sur_row.get("yaw_right", False)
            }

            for behavior, triggered in behavior_keys.items():
                if triggered and behavior in risk_reminders.get(col, {}):
                    position_text = position_alias.get(col, "in your surroundings")
                    adverb_clause = adverb_phrases.get(behavior, "")
                    reminder = risk_reminders[col][behavior]
                    message = f"Vehicle {int(sur_id)} {position_text} is {adverb_clause}. {reminder.strip()}"
                    if message not in reminders:
                        reminders.append(message)

        if reminders:
            paragraph += "\nRisk Reminder(s):\n- " + "\n- ".join(reminders)
        if surrounding_descriptions:
            paragraph += "\nSurrounding Vehicles:\n" + "\n".join(surrounding_descriptions)
        else:
            paragraph += "\nNo surrounding vehicles detected."

        descriptions.append(paragraph)

    return "\n\n".join(descriptions)