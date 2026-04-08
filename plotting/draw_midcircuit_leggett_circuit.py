#!/usr/bin/env python3
"""
Draw Leggett test circuit diagrams with mid-circuit measurements.
Shows sequential reuse of 2 qubits for 6 or 12 measurement sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

# Circuit drawing parameters
QUBIT_SPACING = 1.4
GATE_WIDTH = 0.55
GATE_HEIGHT = 0.45
TIME_STEP = 1.0
SEQUENCE_GAP = 0.8


def get_measurement_gates(vec):
    """Get the optimal gate sequence to measure along vec direction.

    Returns list of (gate_name, angle) tuples.
    - XZ plane (y≈0): single Ry rotation
    - YZ plane (x≈0): single Rx rotation
    - General/XY plane: Rz then Ry
    """
    x, y, z = vec

    # Z-axis: no rotation needed
    if abs(x) < 1e-10 and abs(y) < 1e-10:
        return []

    # XZ plane: use Ry only (signed angle)
    if abs(y) < 1e-10:
        angle = -np.arctan2(x, z)
        if abs(angle) < 1e-10:
            return []
        return [('Ry', angle)]

    # YZ plane: use Rx only
    if abs(x) < 1e-10:
        angle = np.arctan2(y, z)
        if abs(angle) < 1e-10:
            return []
        return [('Rx', angle)]

    # General case (XY plane or other): use Rz then Ry
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    rz = -phi
    ry = -theta
    gates = []
    if abs(rz) > 1e-10:
        gates.append(('Rz', rz))
    if abs(ry) > 1e-10:
        gates.append(('Ry', ry))
    return gates


def draw_qubit_line(ax, y, x_start, x_end, label=None):
    """Draw a horizontal qubit line."""
    ax.plot([x_start, x_end], [y, y], 'k-', linewidth=1, zorder=1)
    if label:
        ax.text(x_start - 0.3, y, label, ha='right', va='center', fontsize=10, fontweight='bold')


def draw_gate(ax, x, y, label, width=GATE_WIDTH, height=GATE_HEIGHT, color='#E8F4FD', edgecolor='#2E86AB'):
    """Draw a gate box."""
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor=edgecolor, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)


def draw_control_target(ax, x, y_control, y_target):
    """Draw a CNOT gate."""
    # Control dot
    ax.plot(x, y_control, 'ko', markersize=8, zorder=3)
    # Target circle with plus
    circle = Circle((x, y_target), 0.15, facecolor='white', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.plot([x, x], [y_target - 0.15, y_target + 0.15], 'k-', linewidth=1.5, zorder=4)
    ax.plot([x - 0.15, x + 0.15], [y_target, y_target], 'k-', linewidth=1.5, zorder=4)
    # Vertical line connecting
    ax.plot([x, x], [y_control, y_target], 'k-', linewidth=1.5, zorder=2)


def draw_measurement(ax, x, y, label=None):
    """Draw a measurement symbol with optional classical bit label."""
    # Meter box
    rect = FancyBboxPatch((x - 0.25, y - 0.2), 0.5, 0.4,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor='#F5F5F5', edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    # Arc
    arc_x = np.linspace(x - 0.15, x + 0.15, 20)
    arc_y = y - 0.05 + 0.1 * np.sin(np.linspace(0, np.pi, 20))
    ax.plot(arc_x, arc_y, 'k-', linewidth=1, zorder=4)
    # Arrow
    ax.annotate('', xy=(x + 0.1, y + 0.1), xytext=(x - 0.05, y - 0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=1), zorder=4)
    if label:
        ax.text(x, y - 0.4, label, ha='center', va='top', fontsize=7, color='#666')


def draw_reset(ax, x, y):
    """Draw a reset symbol."""
    rect = FancyBboxPatch((x - 0.25, y - 0.2), 0.5, 0.4,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor='#E0E0E0', edgecolor='#333333', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, "|0⟩", ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)


def format_angle(radians):
    """Format angle for display, return None if effectively zero."""
    degrees = np.degrees(radians)
    if abs(degrees) < 0.01:  # Effectively zero
        return None
    # Express as fraction of pi if close to common values
    pi_frac = radians / np.pi
    if abs(pi_frac - 1.0) < 0.01 or abs(pi_frac + 1.0) < 0.01:
        return "π"
    elif abs(pi_frac - 0.5) < 0.01:
        return "π/2"
    elif abs(pi_frac + 0.5) < 0.01:
        return "-π/2"
    elif abs(pi_frac - 0.25) < 0.01:
        return "π/4"
    elif abs(pi_frac + 0.25) < 0.01:
        return "-π/4"
    else:
        return f"{degrees:.1f}°"


def draw_midcircuit_leggett(num_sequences, phi_deg, title=None, filename=None):
    """
    Draw Leggett test circuit with mid-circuit measurements.

    Args:
        num_sequences: Number of measurement sequences (6 for +phi only, 12 for ±phi)
        phi_deg: Angle in degrees
        title: Plot title
        filename: Output filename
    """
    phi_rad = np.radians(phi_deg)

    # Alice's measurement directions
    a1 = np.array([1, 0, 0])  # X-axis
    a2 = np.array([0, 1, 0])  # Y-axis
    a3 = np.array([0, 0, 1])  # Z-axis

    # Bob's measurement directions for +phi
    b1_pos = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime_pos = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2_pos = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime_pos = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3_pos = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime_pos = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # Bob's measurement directions for -phi
    b1_neg = np.array([np.cos(-phi_rad/2), np.sin(-phi_rad/2), 0])
    b1_prime_neg = np.array([np.cos(-phi_rad/2), -np.sin(-phi_rad/2), 0])
    b2_neg = np.array([0, np.cos(-phi_rad/2), np.sin(-phi_rad/2)])
    b2_prime_neg = np.array([0, np.cos(-phi_rad/2), -np.sin(-phi_rad/2)])
    b3_neg = np.array([np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])
    b3_prime_neg = np.array([-np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])

    # Measurement pairs for +phi
    pairs_pos = [
        (a1, b1_pos, "C(a₁,b₁)", f"+{phi_deg}°"),
        (a1, b1_prime_pos, "C(a₁,b₁')", f"+{phi_deg}°"),
        (a2, b2_pos, "C(a₂,b₂)", f"+{phi_deg}°"),
        (a2, b2_prime_pos, "C(a₂,b₂')", f"+{phi_deg}°"),
        (a3, b3_pos, "C(a₃,b₃)", f"+{phi_deg}°"),
        (a3, b3_prime_pos, "C(a₃,b₃')", f"+{phi_deg}°"),
    ]

    pairs_neg = [
        (a1, b1_neg, "C(a₁,b₁)", f"-{phi_deg}°"),
        (a1, b1_prime_neg, "C(a₁,b₁')", f"-{phi_deg}°"),
        (a2, b2_neg, "C(a₂,b₂)", f"-{phi_deg}°"),
        (a2, b2_prime_neg, "C(a₂,b₂')", f"-{phi_deg}°"),
        (a3, b3_neg, "C(a₃,b₃)", f"-{phi_deg}°"),
        (a3, b3_prime_neg, "C(a₃,b₃')", f"-{phi_deg}°"),
    ]

    # Select pairs based on num_sequences
    if num_sequences == 6:
        measurement_pairs = pairs_pos
    else:  # 12
        measurement_pairs = pairs_pos + pairs_neg

    # Calculate figure width based on number of sequences
    sequence_width = 8.0  # Width per sequence block (increased for spacing)
    x_end = num_sequences * sequence_width + 2

    # Figure size
    fig_height = 5
    fig_width = min(48, max(20, num_sequences * 4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')

    y_a = QUBIT_SPACING  # Alice's qubit
    y_b = 0              # Bob's qubit

    # Draw qubit lines
    draw_qubit_line(ax, y_a, 0, x_end, "q0")
    draw_qubit_line(ax, y_b, 0, x_end, "q1")

    # Draw each measurement sequence
    x = 0.8
    for seq_idx, (a_vec, b_vec, corr_label, phi_label) in enumerate(measurement_pairs):
        x_start = x

        # Singlet state preparation: X on Bob, H on Alice, CNOT, Z on Bob
        draw_gate(ax, x, y_b, "X", color='#FFE4E1', edgecolor='#CD5C5C')

        x += TIME_STEP
        draw_gate(ax, x, y_a, "H", color='#E8F4FD', edgecolor='#2E86AB')

        x += TIME_STEP
        draw_control_target(ax, x, y_a, y_b)

        x += TIME_STEP
        draw_gate(ax, x, y_b, "Z", color='#E8FFE8', edgecolor='#228B22')

        # Measurement rotations
        gates_a = get_measurement_gates(a_vec)
        gates_b = get_measurement_gates(b_vec)

        x += TIME_STEP

        # Alice's rotations
        x_a = x
        for gate_name, angle in gates_a:
            label = format_angle(angle)
            if label:
                draw_gate(ax, x_a, y_a, gate_name, color='#FFF8DC', edgecolor='#DAA520')
                ax.text(x_a, y_a + 0.38, label, ha='center', va='bottom', fontsize=7, color='#666')
                x_a += TIME_STEP

        # Bob's rotations
        x_b = x
        for gate_name, angle in gates_b:
            label = format_angle(angle)
            if label:
                draw_gate(ax, x_b, y_b, gate_name, color='#FFF8DC', edgecolor='#DAA520')
                ax.text(x_b, y_b - 0.38, label, ha='center', va='top', fontsize=7, color='#666')
                x_b += TIME_STEP

        # Move x forward
        x = max(x_a, x_b, x + TIME_STEP)

        # Mid-circuit measurement
        draw_measurement(ax, x, y_a)
        draw_measurement(ax, x, y_b)

        # Add sequence label above
        x_mid = (x_start + x) / 2
        ax.text(x_mid, y_a + 0.9, f"Seq {seq_idx + 1}",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')
        ax.text(x_mid, y_a + 0.55, f"{corr_label} ({phi_label})",
                ha='center', va='bottom', fontsize=9, color='#666')

        x += TIME_STEP + 0.2

        # Reset for next sequence (except last)
        if seq_idx < len(measurement_pairs) - 1:
            draw_reset(ax, x, y_a)
            draw_reset(ax, x, y_b)
            x += TIME_STEP + SEQUENCE_GAP

            # Draw vertical separator
            ax.axvline(x=x - SEQUENCE_GAP/2, color='#CCC', linestyle=':', linewidth=1, zorder=0)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(-1.0, x + 1.0)
    ax.set_ylim(-1.0, y_a + 1.8)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")

    plt.close()


def draw_wavy_continuation(ax, x, y_top, y_bottom):
    """Draw a wavy line to indicate continuation from above."""
    # Create wavy line points
    num_waves = 3
    y_points = np.linspace(y_top, y_bottom, 50)
    x_points = x + 0.15 * np.sin(np.linspace(0, num_waves * 2 * np.pi, 50))
    ax.plot(x_points, y_points, color='#666', linewidth=2, linestyle='-', zorder=5)


def draw_midcircuit_stacked(num_sequences, phi_deg, sequences_per_row=6, title=None, filename=None):
    """
    Draw Leggett test circuit with mid-circuit measurements in stacked rows.

    Args:
        num_sequences: Total number of measurement sequences (e.g., 12)
        phi_deg: Angle in degrees
        sequences_per_row: Number of sequences per row (default 6)
        title: Plot title
        filename: Output filename
    """
    phi_rad = np.radians(phi_deg)

    # Alice's measurement directions
    a1 = np.array([1, 0, 0])  # X-axis
    a2 = np.array([0, 1, 0])  # Y-axis
    a3 = np.array([0, 0, 1])  # Z-axis

    # Bob's measurement directions for +phi
    b1_pos = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime_pos = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2_pos = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime_pos = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3_pos = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime_pos = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # Bob's measurement directions for -phi
    b1_neg = np.array([np.cos(-phi_rad/2), np.sin(-phi_rad/2), 0])
    b1_prime_neg = np.array([np.cos(-phi_rad/2), -np.sin(-phi_rad/2), 0])
    b2_neg = np.array([0, np.cos(-phi_rad/2), np.sin(-phi_rad/2)])
    b2_prime_neg = np.array([0, np.cos(-phi_rad/2), -np.sin(-phi_rad/2)])
    b3_neg = np.array([np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])
    b3_prime_neg = np.array([-np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])

    # Measurement pairs for +phi
    pairs_pos = [
        (a1, b1_pos, "C(a₁,b₁)", f"+{phi_deg}°"),
        (a1, b1_prime_pos, "C(a₁,b₁')", f"+{phi_deg}°"),
        (a2, b2_pos, "C(a₂,b₂)", f"+{phi_deg}°"),
        (a2, b2_prime_pos, "C(a₂,b₂')", f"+{phi_deg}°"),
        (a3, b3_pos, "C(a₃,b₃)", f"+{phi_deg}°"),
        (a3, b3_prime_pos, "C(a₃,b₃')", f"+{phi_deg}°"),
    ]

    pairs_neg = [
        (a1, b1_neg, "C(a₁,b₁)", f"-{phi_deg}°"),
        (a1, b1_prime_neg, "C(a₁,b₁')", f"-{phi_deg}°"),
        (a2, b2_neg, "C(a₂,b₂)", f"-{phi_deg}°"),
        (a2, b2_prime_neg, "C(a₂,b₂')", f"-{phi_deg}°"),
        (a3, b3_neg, "C(a₃,b₃)", f"-{phi_deg}°"),
        (a3, b3_prime_neg, "C(a₃,b₃')", f"-{phi_deg}°"),
    ]

    measurement_pairs = pairs_pos + pairs_neg
    num_rows = (num_sequences + sequences_per_row - 1) // sequences_per_row

    # Calculate dimensions
    sequence_width = 8.0
    row_width = sequences_per_row * sequence_width + 2
    row_height = 4.5  # Height per row including spacing

    # Figure size
    fig_height = num_rows * row_height + 1
    fig_width = 26
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')

    # Draw each row
    for row_idx in range(num_rows):
        # Y positions for this row (top row first)
        row_y_offset = (num_rows - 1 - row_idx) * row_height
        y_a = row_y_offset + QUBIT_SPACING
        y_b = row_y_offset

        # Get sequences for this row
        start_seq = row_idx * sequences_per_row
        end_seq = min(start_seq + sequences_per_row, num_sequences)
        row_pairs = measurement_pairs[start_seq:end_seq]

        # Draw qubit lines
        x_line_start = 0 if row_idx == 0 else 0.8
        # Only show labels on first row
        if row_idx == 0:
            draw_qubit_line(ax, y_a, x_line_start, row_width, "q0")
            draw_qubit_line(ax, y_b, x_line_start, row_width, "q1")
        else:
            draw_qubit_line(ax, y_a, x_line_start, row_width, None)
            draw_qubit_line(ax, y_b, x_line_start, row_width, None)
            # Just show ... to indicate continuation
            ax.text(0.4, (y_a + y_b) / 2, "...", fontsize=16, ha='center', va='center', color='#666')

        # Draw each sequence in this row
        x = 1.2
        for local_idx, (a_vec, b_vec, corr_label, phi_label) in enumerate(row_pairs):
            seq_idx = start_seq + local_idx
            x_start = x

            # Singlet state preparation
            draw_gate(ax, x, y_b, "X", color='#FFE4E1', edgecolor='#CD5C5C')
            x += TIME_STEP
            draw_gate(ax, x, y_a, "H", color='#E8F4FD', edgecolor='#2E86AB')
            x += TIME_STEP
            draw_control_target(ax, x, y_a, y_b)
            x += TIME_STEP
            draw_gate(ax, x, y_b, "Z", color='#E8FFE8', edgecolor='#228B22')

            # Measurement rotations
            gates_a = get_measurement_gates(a_vec)
            gates_b = get_measurement_gates(b_vec)

            x += TIME_STEP

            # Alice's rotations
            x_a = x
            for gate_name, angle in gates_a:
                label = format_angle(angle)
                if label:
                    draw_gate(ax, x_a, y_a, gate_name, color='#FFF8DC', edgecolor='#DAA520')
                    ax.text(x_a, y_a + 0.38, label, ha='center', va='bottom', fontsize=7, color='#666')
                    x_a += TIME_STEP

            # Bob's rotations
            x_b = x
            for gate_name, angle in gates_b:
                label = format_angle(angle)
                if label:
                    draw_gate(ax, x_b, y_b, gate_name, color='#FFF8DC', edgecolor='#DAA520')
                    ax.text(x_b, y_b - 0.38, label, ha='center', va='top', fontsize=7, color='#666')
                    x_b += TIME_STEP

            x = max(x_a, x_b, x + TIME_STEP)

            # Mid-circuit measurement
            draw_measurement(ax, x, y_a)
            draw_measurement(ax, x, y_b)

            # Sequence label
            x_mid = (x_start + x) / 2
            ax.text(x_mid, y_a + 0.9, f"Seq {seq_idx + 1}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')
            ax.text(x_mid, y_a + 0.55, f"{corr_label} ({phi_label})",
                    ha='center', va='bottom', fontsize=9, color='#666')

            x += TIME_STEP + 0.2

            # Reset only if NOT the last sequence in this row AND not the last sequence overall
            is_last_in_row = (local_idx == len(row_pairs) - 1)
            is_last_overall = (seq_idx == num_sequences - 1)

            if not is_last_in_row and not is_last_overall:
                draw_reset(ax, x, y_a)
                draw_reset(ax, x, y_b)
                x += TIME_STEP + SEQUENCE_GAP
                # Separator
                ax.axvline(x=x - SEQUENCE_GAP/2, color='#CCC', linestyle=':', linewidth=1, zorder=0)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(-0.5, row_width + 2.5)
    ax.set_ylim(-1.2, num_rows * row_height + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")

    plt.close()


def draw_midcircuit_compact(num_sequences, phi_deg, sequences_per_row=2, title=None, filename=None):
    """
    Draw a compact stacked version showing the mid-circuit measurement pattern.
    Shows 2 sequences per row with ... continuation between rows.

    Args:
        num_sequences: Number of measurement sequences (6 or 12)
        phi_deg: Angle in degrees
        sequences_per_row: Sequences per row (default 2)
        title: Plot title
        filename: Output filename
    """
    phi_rad = np.radians(phi_deg)

    # Alice's measurement directions
    a1 = np.array([1, 0, 0])  # X-axis
    a2 = np.array([0, 1, 0])  # Y-axis
    a3 = np.array([0, 0, 1])  # Z-axis

    # Bob's measurement directions for +phi
    b1_pos = np.array([np.cos(phi_rad/2), np.sin(phi_rad/2), 0])
    b1_prime_pos = np.array([np.cos(phi_rad/2), -np.sin(phi_rad/2), 0])
    b2_pos = np.array([0, np.cos(phi_rad/2), np.sin(phi_rad/2)])
    b2_prime_pos = np.array([0, np.cos(phi_rad/2), -np.sin(phi_rad/2)])
    b3_pos = np.array([np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])
    b3_prime_pos = np.array([-np.sin(phi_rad/2), 0, np.cos(phi_rad/2)])

    # Bob's measurement directions for -phi
    b1_neg = np.array([np.cos(-phi_rad/2), np.sin(-phi_rad/2), 0])
    b1_prime_neg = np.array([np.cos(-phi_rad/2), -np.sin(-phi_rad/2), 0])
    b2_neg = np.array([0, np.cos(-phi_rad/2), np.sin(-phi_rad/2)])
    b2_prime_neg = np.array([0, np.cos(-phi_rad/2), -np.sin(-phi_rad/2)])
    b3_neg = np.array([np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])
    b3_prime_neg = np.array([-np.sin(-phi_rad/2), 0, np.cos(-phi_rad/2)])

    # Measurement pairs
    pairs_pos = [
        (a1, b1_pos, "C(a₁,b₁)", f"+{phi_deg}°"),
        (a1, b1_prime_pos, "C(a₁,b₁')", f"+{phi_deg}°"),
        (a2, b2_pos, "C(a₂,b₂)", f"+{phi_deg}°"),
        (a2, b2_prime_pos, "C(a₂,b₂')", f"+{phi_deg}°"),
        (a3, b3_pos, "C(a₃,b₃)", f"+{phi_deg}°"),
        (a3, b3_prime_pos, "C(a₃,b₃')", f"+{phi_deg}°"),
    ]

    pairs_neg = [
        (a1, b1_neg, "C(a₁,b₁)", f"-{phi_deg}°"),
        (a1, b1_prime_neg, "C(a₁,b₁')", f"-{phi_deg}°"),
        (a2, b2_neg, "C(a₂,b₂)", f"-{phi_deg}°"),
        (a2, b2_prime_neg, "C(a₂,b₂')", f"-{phi_deg}°"),
        (a3, b3_neg, "C(a₃,b₃)", f"-{phi_deg}°"),
        (a3, b3_prime_neg, "C(a₃,b₃')", f"-{phi_deg}°"),
    ]

    if num_sequences == 6:
        measurement_pairs = pairs_pos
    else:
        measurement_pairs = pairs_pos + pairs_neg

    num_rows = (num_sequences + sequences_per_row - 1) // sequences_per_row
    row_height = 3.8
    row_width = 16

    # Figure size
    fig_height = num_rows * row_height + 1
    fig_width = 13
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')

    # Draw each row
    for row_idx in range(num_rows):
        row_y_offset = (num_rows - 1 - row_idx) * row_height
        y_a = row_y_offset + QUBIT_SPACING
        y_b = row_y_offset

        start_seq = row_idx * sequences_per_row
        end_seq = min(start_seq + sequences_per_row, num_sequences)
        row_pairs = measurement_pairs[start_seq:end_seq]

        # Draw qubit lines
        x_line_end = row_width - 2
        if row_idx == 0:
            draw_qubit_line(ax, y_a, 0, x_line_end, "q0")
            draw_qubit_line(ax, y_b, 0, x_line_end, "q1")
        else:
            draw_qubit_line(ax, y_a, 0.8, x_line_end, None)
            draw_qubit_line(ax, y_b, 0.8, x_line_end, None)
            ax.text(0.4, (y_a + y_b) / 2, "...", fontsize=16, ha='center', va='center', color='#666')

        # Fixed column positions for alignment (relative to sequence start)
        # Col 0: H/X, Col 1: CNOT, Col 2: Z, Col 3-4: Rotations, Col 5: Measure, Col 6: Reset
        SEQ_WIDTH = 7 * TIME_STEP

        # Draw sequences in this row
        for local_idx, (a_vec, b_vec, corr_label, phi_label) in enumerate(row_pairs):
            seq_idx = start_seq + local_idx
            x_base = 1.2 + local_idx * SEQ_WIDTH

            # Column positions
            x_hx = x_base
            x_cnot = x_base + TIME_STEP
            x_z = x_base + 2 * TIME_STEP
            x_rot1 = x_base + 3 * TIME_STEP
            x_rot2 = x_base + 4 * TIME_STEP
            x_meas = x_base + 5 * TIME_STEP
            x_reset = x_base + 6 * TIME_STEP

            # Singlet prep - X and H aligned vertically
            draw_gate(ax, x_hx, y_a, "H", color='#E8F4FD', edgecolor='#2E86AB')
            draw_gate(ax, x_hx, y_b, "X", color='#FFE4E1', edgecolor='#CD5C5C')
            draw_control_target(ax, x_cnot, y_a, y_b)
            draw_gate(ax, x_z, y_b, "Z", color='#E8FFE8', edgecolor='#228B22')

            # Measurement rotations
            gates_a = get_measurement_gates(a_vec)
            gates_b = get_measurement_gates(b_vec)

            # Alice's rotations (at fixed positions)
            x_a = x_rot1
            for gate_name, angle in gates_a:
                label = format_angle(angle)
                if label:
                    draw_gate(ax, x_a, y_a, gate_name, color='#FFF8DC', edgecolor='#DAA520')
                    ax.text(x_a, y_a + 0.38, label, ha='center', va='bottom', fontsize=7, color='#666')
                    x_a += TIME_STEP

            # Bob's rotations (at fixed positions)
            x_b = x_rot1
            for gate_name, angle in gates_b:
                label = format_angle(angle)
                if label:
                    draw_gate(ax, x_b, y_b, gate_name, color='#FFF8DC', edgecolor='#DAA520')
                    ax.text(x_b, y_b - 0.38, label, ha='center', va='top', fontsize=7, color='#666')
                    x_b += TIME_STEP

            # Measurement - at fixed column position
            draw_measurement(ax, x_meas, y_a)
            draw_measurement(ax, x_meas, y_b)

            # Sequence label
            x_mid = (x_hx + x_meas) / 2
            ax.text(x_mid, y_a + 0.9, f"Seq {seq_idx + 1}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444')
            ax.text(x_mid, y_a + 0.55, f"{corr_label} ({phi_label})",
                    ha='center', va='bottom', fontsize=9, color='#666')

            # Reset after each sequence (except the very last one overall)
            is_last_overall = (seq_idx == num_sequences - 1)

            if not is_last_overall:
                # Reset symbols at fixed column position
                draw_reset(ax, x_reset, y_a)
                draw_reset(ax, x_reset, y_b)

    ax.set_xlim(-0.5, row_width)
    ax.set_ylim(-0.8, num_rows * row_height - 0.8)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")

    plt.close()


if __name__ == "__main__":
    # Compact diagrams (recommended for papers)
    draw_midcircuit_compact(
        num_sequences=6,
        phi_deg=30,
        title="Mid-Circuit Measurement Leggett Test: 6 Sequences (φ=+30° only)",
        filename="leggett_midcircuit_6seq_compact.png"
    )

    draw_midcircuit_compact(
        num_sequences=12,
        phi_deg=30,
        title="Mid-Circuit Measurement Leggett Test: 12 Sequences (φ=±30°)",
        filename="leggett_midcircuit_12seq_compact.png"
    )

    # Full detailed diagram for 6 sequences (single row)
    draw_midcircuit_leggett(
        num_sequences=6,
        phi_deg=30,
        title="Mid-Circuit Measurement Leggett Test: 6 Sequences Detail (φ=+30°)",
        filename="leggett_midcircuit_6seq_full.png"
    )

    # Stacked diagram for 12 sequences (2 rows of 6)
    draw_midcircuit_stacked(
        num_sequences=12,
        phi_deg=30,
        sequences_per_row=6,
        title="Mid-Circuit Measurement Leggett Test: 12 Sequences (φ=±30°)",
        filename="leggett_midcircuit_12seq_stacked.png"
    )

    print("\nAll mid-circuit measurement circuit diagrams generated!")
