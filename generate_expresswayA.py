
import pandas as pd
import json
import re
import os

#ExpresswayA的system message
expresswayA_msg = """"

You are ChatGPT, a large language model trained by OpenAI. You are now acting as a mature driving assistant. And your task is to make the decision that assures the safety.

You have access to the same information as a real human driver, which includes the following:
1. Description of the environment
2. Your own vehicle's position, dynamic information, and legally permissible actions in that lane.
3. Vehicles in your current lane and adjacent lanes, including their position, dynamic information, and legally permissible actions in that lane.

This scenario takes place at an expressway with exits. The positive x-coordinate points east, while the positive y-coordinate points south. The top-left corner is the origin (0,0). The velocity components follow the same convention: a negative vx indicates movement toward the west, while a negative vy indicates movement toward the north.

Your available actions include:
1. IDLE: Remain in the current lane with the current speed (Action ID: 1)
2. Turn Left: Change to the lane on the left of the current lane (Action ID: 2)
3. Turn Right: Change to the lane on the right of the current lane (Action ID: 3)
4. Acceleration: Increase vehicle speed (Action ID: 4)
5. Deceleration: Reduce vehicle speed (Action ID: 5)

A clear action ID must be chosen at the end of your reasoning process.
"""

#ExpresswayA Dictionary
lane_info_expresswayA = {
    0: {
        "direction": "east-west",
        "left_adjacent_lane": [3],
        "right_adjacent_lane": None,
        "front_lane":[1],
        "available_actions": ["Move straight to lane 1, change to left lane 3."]
    },
    1: {
        "direction": "east-west",
        "left_adjacent_lane": [3],
        "right_adjacent_lane": None,
        "front_lane":[2],
        "available_actions": ["Move straight to lane 2, change to left lane 3."]
    },
    2: {
        "direction": "east-west",
        "left_adjacent_lane": [3],
        "right_adjacent_lane": None,
        "available_actions": ["Move straight and exit freeway, change to left lane 3 (illegal)."]
    },
    3: {
        "direction": "east-west",
        "left_adjacent_lane": [4],
        "right_adjacent_lane": [1],
        "available_actions": [
            "Move straight, change to left lane 4, change to right lane 1"]
    },
    4: {
        "direction": "east-west",
        "left_adjacent_lane": [5],
        "right_adjacent_lane": [3],
        "available_actions": ["Move straight, change to left lane 5, change to right lane 3"]
    },
    5: {
        "direction": "east-west",
        "left_adjacent_lane": None,
        "right_adjacent_lane": [4],
        "available_actions": ["Move straight, change to right lane 4"]
    },
    6: {
        "direction": "west-east",
        "left_adjacent_lane": None,
        "right_adjacent_lane": [7],
        "available_actions": ["Move straight, change to right lane 7"]
    },
    7: {
        "direction": "west-east",
        "left_adjacent_lane": [6],
        "right_adjacent_lane": [8],
        "available_actions": ["Move straight, change to left lane 6, change to right lane 8"]
    },
    8: {
        "direction": "west-east",
        "left_adjacent_lane": [7],
        "right_adjacent_lane": [10],
        "available_actions": ["Move straight, change to left lane 7, change to right lane 10"]
    },
    9: {
        "direction": "west-east",
        "left_adjacent_lane": [8],
        "right_adjacent_lane": None,
        "front_lane":[10],
        "available_actions": ["Move straight to lane 10, change to left lane 8."]
    },
    10: {
        "direction": "west-east",
        "left_adjacent_lane": [8],
        "right_adjacent_lane": None,
        "available_actions": ["Move straight to lane 11, change to left lane 8"]
    },
    11: {
        "direction": "west-east",
        "left_adjacent_lane": [8],
        "right_adjacent_lane": None,
        "available_actions": ["Move straight and exit freeway, change to left lane 8 (illegal)"]
    },
}


def get_risk_reminders(vehicle_position, vehicle_row):
    reminders = []
    maneuvers = risk_reminders.get(vehicle_position, {})

    for maneuver, message in maneuvers.items():
        if vehicle_row.get(maneuver):
            reminders.append(message)

    return reminders

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


def generate_expresswayA(json_path, lane_info=lane_info_expresswayA, frame_step=5, start_frame=None, end_frame=None):
    filename = os.path.basename(json_path)
    ego_match = re.search(r'car[_]?(\d+)', filename)
    if ego_match:
        ego_id = int(ego_match.group(1))
    else:
        raise ValueError("Ego vehicle ID (carXXXX or car_XXXX) not found in filename.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["frame_id"] = df["frame_id"].astype(int)
    df["car_id"] = df["car_id"].astype(int)

    # Set the default start and end frames if not provided
    if start_frame is None:
        start_frame = df["frame_id"].min()
    if end_frame is None:
        end_frame = df["frame_id"].max()

    all_descriptions = []

    # Process frames from start_frame to end_frame with the specified step size
    for frame_id in range(start_frame, end_frame + 1, frame_step):
        frame_df = df[df["frame_id"] == frame_id]
        ego_row = frame_df[frame_df["car_id"] == ego_id]

        if ego_row.empty:
            continue

        ego_row = ego_row.iloc[0]
        ego_x, ego_y, ego_speed, ego_course = (
            ego_row["car_center_x"], ego_row["car_center_y"], ego_row["speed"], ego_row["course"]
        )

        ego_lane = ego_row["lane_id"] if pd.notna(ego_row["lane_id"]) else "Unknown"
        lane_details = lane_info.get(ego_lane, {})
        available_actions = lane_details.get("available_actions", ["No available actions"])
        available_actions_str = ", ".join(available_actions) if isinstance(available_actions, list) else available_actions

        paragraph = f"""
================================================================================
Frame {frame_id}:
================================================================================

Ego Vehicle {ego_id} is currently traveling in lane {ego_lane}, classified as a {lane_details.get("direction", "Unknown direction")} direction lane.
The available legal maneuvers for this lane include: {available_actions_str}.
At the current timestamp, the vehicle is located at coordinates ({ego_x:.2f}, {ego_y:.2f}), moving at a speed of {ego_speed:.2f} m/s.
The vehicle's heading angle is {ego_course:.2f} degrees.
"""

        categories = {
            "preceding_id": "driving directly ahead in your lane",
            "following_id": "driving behind in your lane",
            "left_preceding_id": "ahead in your adjacent left lane",
            "left_following_id": "behind in your adjacent left lane",
            "left_alongside_id": "laterally aligned in your adjacent left lane",
            "right_preceding_id": "ahead in your adjacent right lane",
            "right_following_id": "behind in your adjacent right lane",
            "right_alongside_id": "laterally aligned in your adjacent right lane"
        }

        surrounding_descriptions = []
        all_safety_reminders = []

        for col_name, description in categories.items():
            vehicle_id = ego_row.get(col_name)
            if pd.notna(vehicle_id):
                try:
                    vehicle_id = int(vehicle_id)
                    vehicle_row = frame_df[frame_df["car_id"] == vehicle_id]

                    if not vehicle_row.empty:
                        vehicle_row = vehicle_row.iloc[0]
                        v_lane = vehicle_row.get("lane_id", "Unknown")
                        v_lane_details = lane_info.get(v_lane, {})
                        v_actions = v_lane_details.get("available_actions", ["No available actions"])
                        v_actions_str = ", ".join(v_actions) if isinstance(v_actions, list) else v_actions

                        desc = f"""
Vehicle {vehicle_id} is {description}: lane {v_lane} — a {v_lane_details.get("direction", "Unknown direction")} lane.
The legal actions for this lane include: {v_actions_str}.
The vehicle is located at ({vehicle_row['car_center_x']:.2f}, {vehicle_row['car_center_y']:.2f}), moving at a speed of {vehicle_row['speed']:.2f} m/s.
The vehicle's heading angle is {vehicle_row['course']:.2f} degrees.
"""
                        reminder_lines = get_risk_reminders(col_name, vehicle_row)
                        if reminder_lines:
                            position_label = col_name.replace("_id", "").replace("_", " ").capitalize()
                            for msg in reminder_lines:
                                full_msg = f"⚠️ {position_label} vehicle {vehicle_id} {msg}"
                                all_safety_reminders.append(full_msg)

                        surrounding_descriptions.append(desc)
                except (ValueError, TypeError):
                    continue

        if surrounding_descriptions:
            paragraph += "\n".join(surrounding_descriptions)
        else:
            paragraph += "\nNo surrounding vehicles were detected in the vicinity during this frame."

        if ego_row.get("collision_0_1s") or ego_row.get("collision_1_2s"):
            all_safety_reminders.append("\n[⚠️ Time-To-Collision (TTC) Risk Assessment]")

            if ego_row.get("collision_0_1s"):
                ttc_01 = ego_row.get("TTC_0_1s", "N/A")
                cid_01 = ego_row.get("collision_ids_0_1s", "N/A")
                all_safety_reminders.append(
                    f"⚠️ Vehicle {cid_01} is on an imminent collision course within 0–1 seconds. Estimated TTC: {ttc_01}."
                )
                for key in categories:
                    if pd.notna(ego_row.get(key)) and str(cid_01) in str(ego_row.get(key)):
                        natural_key = key.replace("_id", "").replace("_", " ")
                        all_safety_reminders.append(
                            f"Vehicle {cid_01} is located in the {natural_key} position; take immediate caution."
                        )

            if ego_row.get("collision_1_2s"):
                ttc_12 = ego_row.get("TTC_1_2s", "N/A")
                cid_12 = ego_row.get("collision_ids_1_2s", "N/A")
                all_safety_reminders.append(
                    f"⚠️ Vehicle {cid_12} poses a collision risk within 1–2 seconds. Estimated TTC: {ttc_12}."
                )

        if all_safety_reminders:
            paragraph += "\n\nSafety Reminders:\n" + "\n".join(all_safety_reminders)

        all_descriptions.append(paragraph)

    return "\n\n".join(all_descriptions)
